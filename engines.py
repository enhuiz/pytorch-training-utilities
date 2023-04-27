import time
import warnings
from typing import Any, Protocol

import torch
import torch.distributed
from deepspeed import DeepSpeedEngine
from deepspeed.runtime.utils import clip_grad_norm_
from torch import Tensor
from torch.distributed import all_reduce

from .config import Config
from .distributed import fix_unset_envs
from .utils import dispatch_attribute, flatten_dict, gather_attribute

Stats = dict[str, float]


class Engine(DeepSpeedEngine):
    def __init__(self, *args, **kwargs):
        fix_unset_envs()
        super().__init__(None, *args, **kwargs)
        self._frozen_params = set()
        self._fp32_grad_norm = None

    def freeze_(self):
        for p in self.module.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                self._frozen_params.add(p)

    def unfreeze_(self):
        for p in self._frozen_params:
            p.requires_grad_(True)
        self._frozen_params.clear()

    def freeze(self):
        warnings.warn("freeze is deprecated, use freeze_ instead", DeprecationWarning)
        self.freeze_()

    def unfreeze(self):
        warnings.warn("freeze is deprecated, use freeze_ instead", DeprecationWarning)
        self.unfreeze_()

    @property
    def global_step(self):
        return self.global_steps

    def gather_attribute(self, *args, **kwargs):
        return gather_attribute(self.module, *args, **kwargs)

    def dispatch_attribute(self, *args, **kwargs):
        return dispatch_attribute(self.module, *args, **kwargs)

    def clip_fp32_gradients(self):
        self._fp32_grad_norm = clip_grad_norm_(
            parameters=self.module.parameters(),
            max_norm=self.gradient_clipping(),
            mpu=self.mpu,
        )

    def get_grad_norm(self):
        grad_norm = self.get_global_grad_norm()
        if grad_norm is None:
            grad_norm = self._fp32_grad_norm
        return grad_norm


class TrainFeeder(Protocol):
    def __call__(
        self, *, engines: "Engines", batch: Any, name: str
    ) -> None | tuple[Tensor, Stats]:
        ...


class Engines(dict[str, Engine]):
    def setup(self, cfg: Config):
        self._cfg = cfg
        self._global_step = 0

    @property
    def cfg(self) -> Config:
        return self._cfg

    @property
    def config(self):
        return self._cfg

    @property
    def global_step(self):
        return self._global_step

    def gather_attribute(self, *args, **kwargs):
        ret = {}
        for engine in self.values():
            ret |= engine.gather_attribute(*args, **kwargs)
        return ret

    def dispatch_attribute(self, *args, **kwargs):
        for engine in self.values():
            engine.dispatch_attribute(*args, **kwargs)

    def save_checkpoint(self, tag="default"):
        self.cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
        for name, engine in self.items():
            engine.save_checkpoint(self.cfg.ckpt_dir / name, tag=tag)

    def load_checkpoint(self, tag=None):
        for name, engine in self.items():
            load_dir = self.cfg.ckpt_dir / name
            engine.load_checkpoint(
                tag=tag,
                load_dir=load_dir,
                load_module_strict=self.cfg.strict_loading,
                load_optimizer_states=self.cfg.strict_loading,
            )
        self._update_global_step()

    def _update_global_step(self):
        for engine in self.values():
            self._global_step = max(self._global_step, engine.global_step)

    def eval(self):
        for engine in self.values():
            engine.eval()

    def train(self):
        for engine in self.values():
            engine.train()

    def step(self, feeder: TrainFeeder, batch):
        total_elapsed_time = 0

        stats: Any = dict()

        for name, engine in self.items():
            torch.cuda.synchronize()
            start_time = time.time()

            n_ooms = torch.zeros([], device=self.cfg.device)

            try:
                maybe_loss_and_engine_stats = feeder(
                    engines=self, batch=batch, name=name
                )

                if maybe_loss_and_engine_stats is None:
                    # Here we allow skip optimizers. It's useful when, for example,
                    # skipping discriminators in the begining of GAN training.
                    continue

                loss, engine_stats = maybe_loss_and_engine_stats

                engine.backward(loss)
                engine.step()

                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                total_elapsed_time += elapsed_time

                stats.update(
                    flatten_dict(
                        {
                            name: dict(
                                loss=loss.item(),
                                lr=engine.get_lr()[0],
                                grad_norm=engine.get_grad_norm(),
                                elapsed_time=elapsed_time,
                                engine_step=engine.global_step,
                                **engine_stats,
                            )
                        }
                    ),
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.cfg.save_on_oom:
                    n_ooms += 1
                else:
                    raise e

            all_reduce(n_ooms)

            if n_ooms.item() > 0:
                self.save_checkpoint()
                raise RuntimeError("Out of memory!")

        self._update_global_step()
        stats["elapsed_time"] = total_elapsed_time
        stats["wall_time"] = time.time()
        stats["global_step"] = self.global_step

        return stats
