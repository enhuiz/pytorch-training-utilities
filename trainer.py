import json
import logging
import random
import selectors
import sys
import time
from dataclasses import asdict, dataclass, replace
from functools import cache
from itertools import zip_longest
from pathlib import Path
from typing import Any, Protocol, TypeVar, overload

import humanize
import torch
from torch import Tensor, nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .config import Config
from .logging import setup_logging
from .utils import flatten_dict

_logger = logging.getLogger(__name__)

T = TypeVar("T")
Model = TypeVar("Model", bound=nn.Module)
Model_co = TypeVar("Model_co", bound=nn.Module, covariant=True)
Model_contra = TypeVar("Model_contra", bound=nn.Module, contravariant=True)


@dataclass
class State:
    iteration: int = 0

    def asdict(self):
        return asdict(self)


_cfg: Config
_state: State


def get_cfg():
    try:
        return _cfg
    except:
        raise RuntimeError(
            "Trainer has not been setup, please call trainer.setup() first."
        )


def get_state():
    try:
        return replace(_state)
    except:
        return None


def get_iteration():
    try:
        return _state.iteration
    except:
        return None


def setup(cfg: Config):
    # Register cfg
    global _cfg
    _cfg = cfg

    # Dump cfg
    cfg.dump()

    # Config logging
    setup_logging(cfg.log_dir)
    _logger.info(cfg)


def load_state_dict_non_strict(model, state_dict):
    model_state_dict = model.state_dict()
    provided = set(state_dict)
    required = set(model_state_dict)
    agreed = provided & required
    for k in list(agreed):
        if model_state_dict[k].shape != state_dict[k].shape:
            agreed.remove(k)
            provided.remove(k)
    state_dict = {k: state_dict[k] for k in agreed}
    if diff := provided - required:
        _logger.warning(
            f"Extra parameters are found. "
            f"Provided but not required parameters: \n{diff}."
        )
    if diff := required - provided:
        _logger.warning(
            f"Some parameters are missing. "
            f"Required but not provided parameters: \n{diff}."
        )
    model.load_state_dict(state_dict, strict=False)


@cache
def load_ckpt(path: Path | str):
    return torch.load(path, map_location="cpu")


def load_model(path: Path | str, model: Model, strict=True) -> tuple[Model, State]:
    path = Path(path)
    if path.exists():
        ckpt = load_ckpt(path)
        if strict:
            model.load_state_dict(ckpt["model"])
        else:
            load_state_dict_non_strict(model, ckpt["model"])
        state = State(**ckpt["state"])
        _logger.info(f"{path} loaded.")
    else:
        _logger.warn(f"{path} does not exist, skip loading.")
        state = State()
    return model, state


class ModelLoader(Protocol[Model_co]):
    def __call__(self, path: Path) -> tuple[Model_co, State]:
        ...


class OptimizerFactory(Protocol[Model_contra]):
    def __call__(self, model: Model_contra) -> Optimizer:
        ...


class SchedulerFactory(Protocol):
    def __call__(self, optimizer: Optimizer) -> Scheduler:
        ...


StepStats = dict[str, float]


class TrainStep(Protocol[Model_contra]):
    def __call__(
        self,
        *,
        model: Model_contra,
        batch: Any,
        state: State,
        optimizer_idx: int,
    ) -> None | tuple[Tensor, StepStats]:
        ...


class EvalFn(Protocol[Model_contra]):
    def __call__(self, *, model: Model_contra, state: State):
        ...


class Logger(Protocol):
    def __call__(
        self,
        *,
        data: dict,
    ):
        ...


class ModelSaver(Protocol[Model_co]):
    def __call__(
        self,
        path: Path,
        model: nn.Module,
        state: State,
        optimizers: list[Optimizer],
        schedulers: list[Scheduler],
        **kwargs,
    ):
        ...


def save_model(
    path: Path | str,
    model: nn.Module,
    state: State,
    optimizers: list[Optimizer],
    schedulers: list[Scheduler],
    **kwargs,
):
    torch.save(
        dict(
            model=model.state_dict(),
            optimizers=[optimizer.state_dict() for optimizer in optimizers],
            schedulers=[scheduler.state_dict() for scheduler in schedulers],
            state=state.asdict(),
            **kwargs,
        ),
        path,
    )
    _logger.info(f"{path} saved.")


def _load_optimizers_and_schedulers(
    *,
    path: Path,
    optimizers: list[Optimizer],
    schedulers: list[Scheduler],
):
    path = Path(path)

    if not path.exists():
        return

    ckpt = load_ckpt(path)

    try:
        for optimizer, state_dict in zip(optimizers, ckpt["optimizers"]):
            optimizer.load_state_dict(state_dict)
        _logger.info(f"Optimizers loaded from {path}.")
    except Exception as e:
        _logger.warn(f"Loading optimizers from {path} failed due to {str(e)}.")

    try:
        for scheduler, state_dict in zip(schedulers, ckpt["schedulers"]):
            scheduler.load_state_dict(state_dict)
        _logger.info(f"Schedulers loaded from {path}.")
    except Exception as e:
        _logger.warn(f"Loading schedulers from {path} failed due to {str(e)}.")


@cache
def _get_stdin_selector():
    selector = selectors.DefaultSelector()
    selector.register(fileobj=sys.stdin, events=selectors.EVENT_READ)
    return selector


def _non_blocking_input():
    s = ""
    selector = _get_stdin_selector()
    events = selector.select(timeout=0)
    for key, _ in events:
        s = key.fileobj.readline().strip()
        _logger.info(f'Get stdin "{s}".')
    shared = [s]
    return shared[0]


def _make_infinite_epochs(dl):
    while True:
        _logger.info("New epoch starts.")
        yield from dl


@overload
def to_device(x: list[T], device: str) -> list[T]:
    ...


@overload
def to_device(x: dict[str, T], device: str) -> dict[str, T]:
    ...


@overload
def to_device(x: T, device: str) -> T:
    ...


def to_device(x, device):
    if isinstance(x, list):
        x = [to_device(xi, device) for xi in x]
    elif isinstance(x, dict):
        x = {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, Tensor):
        x = x.to(device)
    return x


def train(
    model_loader: ModelLoader,
    optimizer_factories: list[OptimizerFactory],
    train_dl: DataLoader,
    train_step: TrainStep,
    eval_fn: EvalFn,
    scheduler_factories: list[SchedulerFactory | None] = [],
    logger: Logger = lambda data: _logger.info(json.dumps(data, indent=2, default=str)),
    model_saver: ModelSaver = save_model,
):
    cfg = get_cfg()

    # Set up random seeds
    random.seed(0)
    torch.manual_seed(0)

    # Load model
    model, state = model_loader(cfg.ckpt_path)
    model = model.to(cfg.device)
    model.train()

    # Setup state
    global _state
    _state = state

    # Prepare optimizers and schedulers
    optimizers: list[Optimizer] = []
    schedulers: list[Scheduler] = []
    for optimizer_factory, scheduler_factory in zip_longest(
        optimizer_factories, scheduler_factories
    ):
        optimizer = optimizer_factory(model)
        optimizers.append(optimizer)
        if scheduler_factory is not None:
            scheduler = scheduler_factory(optimizer)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        schedulers.append(scheduler)

    _load_optimizers_and_schedulers(
        path=cfg.ckpt_path,
        optimizers=optimizers,
        schedulers=schedulers,
    )

    # Release cached ckpt
    load_ckpt.cache_clear()

    events = []

    # Pre-loop command
    command = _non_blocking_input()
    if command in ["eval", "eval_quit"]:
        model.eval()
        eval_fn(model=model, state=state)
        model.train()
    if command in ["quit", "eval_quit"]:
        return

    # Training loop
    for batch in _make_infinite_epochs(train_dl):
        if state.iteration >= cfg.max_iter:
            break

        state.iteration += 1

        total_elapsed_time = 0
        batch = to_device(batch, cfg.device)

        log_dict: Any = dict(iteration=state.iteration)

        for optimizer_idx, (optimizer, scheduler) in enumerate(
            zip(optimizers, schedulers)
        ):
            torch.cuda.synchronize()
            start_time = time.time()

            maybe_loss_stats = train_step(
                model=model,
                batch=batch,
                state=state,
                optimizer_idx=optimizer_idx,
            )

            if maybe_loss_stats is None:
                # Here we allow skip optimizers. It's useful when, for example,
                # skipping discriminators in the begining of GAN training.
                continue

            loss, stats = maybe_loss_stats

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                max_norm=cfg.max_grad_norm or 1e9,
            )
            optimizer.step()
            scheduler.step()

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            total_elapsed_time += elapsed_time

            log_dict.update(
                flatten_dict(
                    {
                        f"{optimizer_idx}": dict(
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            grad_norm=grad_norm.item(),
                            elapsed_time=elapsed_time,
                            **stats,
                        )
                    }
                ),
            )

        log_dict["elapsed_time"] = total_elapsed_time
        log_dict["wall_time"] = time.time()
        logger(data=log_dict)

        command = _non_blocking_input()

        if "@" in command:
            what, when = command.split("@")
            try:
                events.append((what, int(when)))
                _logger.info(f"Event {command} registered.")
            except Exception as e:
                _logger.error(e)
            command = ""

        # Commands are the current command plus the triggered (i.e. iteration >= trigger point) events
        events = [e for e in events if e[1] >= state.iteration]
        commands = [command] + [e[0] for e in events if e[1] == state.iteration]

        for command in commands:
            if command in ["event show", "event"]:
                msg = "Events:\n" + "\n".join(["@".join(map(str, e)) for e in events])
                _logger.info(msg)

            if command == "event clear":
                events.clear()

            if "time" in command:
                target_iter = cfg.max_iter
                if " to " in command:
                    try:
                        target_iter = int(command.split(" to ")[-1])
                    except Exception as e:
                        _logger.error(e)
                remaining_iters = target_iter - state.iteration + 1
                remaining_time = int(remaining_iters * total_elapsed_time)
                _logger.info(humanize.precisedelta(remaining_time))

            ckpt_every = cfg.ckpt_every or cfg.eval_every
            if state.iteration % ckpt_every == 0 or command in ["save", "quit"]:
                cfg.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                model_saver(
                    path=cfg.ckpt_path,
                    model=model,
                    state=state,
                    optimizers=optimizers,
                    schedulers=schedulers,
                )

            if state.iteration % cfg.eval_every == 0 or command in ["eval"]:
                model.eval()
                eval_fn(model=model, state=state)
                model.train()

            if command in ["quit"]:
                return
