import logging
import time
from typing import Any, Protocol

import torch
from deepspeed import DeepSpeedEngine
from torch import Tensor, nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from .config import Config
from .distributed import is_local_leader
from .utils import flatten_dict

Stats = dict[str, float]

_logger = logging.getLogger(__name__)


class Engine(DeepSpeedEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(None, *args, **kwargs)

    @property
    def global_step(self):
        return self.global_steps


class TrainStepFn(Protocol):
    def __call__(
        self,
        *,
        engines: "Engines",
        batch: Any,
        name: str,
    ) -> None | tuple[Tensor, Stats]:
        ...


class Engines(nn.ModuleDict):
    def setup(self, cfg: Config):
        self._cfg = cfg
        if is_local_leader():
            cfg.dump()
            _logger.info(cfg)

    @property
    def cfg(self):
        return self._cfg

    @property
    def config(self):
        return self._cfg

    @property
    def global_step(self):
        values = set()
        for engine in self.values():
            values.add(engine.global_step)
        if len(values) > 1:
            raise ValueError(
                "Multiple global steps detected, maybe errors in the checkpoints?"
            )
        return next(iter(values))

    def save_checkpoint(self):
        self.cfg.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        for name, engine in self.items():
            engine.save_checkpoint(self.cfg.ckpt_path / f"engine-{name}")

    def load_checkpoint(self):
        for name, engine in self.items():
            engine.load_checkpoint(
                self.cfg.ckpt_path / f"engine-{name}",
                load_module_strict=False,
            )

    def eval(self):
        for engine in self.values():
            engine.eval()

    def train(self):
        for engine in self.values():
            engine.train()

    def step(self, fn: TrainStepFn, batch):
        total_elapsed_time = 0
        stats: Any = dict(global_step=self.global_step)

        for name, engine in self.items():
            torch.cuda.synchronize()
            start_time = time.time()

            try:
                maybe_loss_substats = fn(engines=self, batch=batch, name=name)

                if maybe_loss_substats is None:
                    # Here we allow skip optimizers. It's useful when, for example,
                    # skipping discriminators in the begining of GAN training.
                    continue

                loss, substats = maybe_loss_substats

                engine.backward(loss)

                # For monitoring purpose
                grad_norm = clip_grad_norm_(engine.parameters(), max_norm=1e9)

                engine.step()
            except RuntimeError as e:
                if "out of memory" in str(e) and self.cfg.save_on_oom:
                    self.save_checkpoint()
                raise e

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            total_elapsed_time += elapsed_time

            stats.update(
                flatten_dict(
                    {
                        name: dict(
                            loss=loss.item(),
                            lr=engine.get_lr(),
                            grad_norm=grad_norm.item(),
                            elapsed_time=elapsed_time,
                            **substats,
                        )
                    }
                ),
            )

        stats["elapsed_time"] = total_elapsed_time
        stats["wall_time"] = time.time()

        return stats
