import re
import logging
import sys
import time
import torch
import torch.nn as nn
import random
import selectors
import humanize
import pandas as pd
from itertools import zip_longest
from functools import cache
from pathlib import Path
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol, TypeVar, overload

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


def load_model(path: Path | str, model: Model, strict=True) -> tuple[Model, State]:
    path = Path(path)
    if path.exists():
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=strict)
        state = State(**ckpt["state"])
        _logger.info(f"{path} loaded.")
    else:
        _logger.warn(f"{path} does not exist, skip loading.")
        state = State()
    return model, state


def config_logging(log_dir: str | Path | None = "log", log_level="info"):
    handlers: list[Any] = []
    if log_dir is not None:
        filename = Path(log_dir) / f"log.txt"
        filename.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename, mode="a")
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    handlers.append(console_handler)
    logging.basicConfig(
        level=logging.getLevelName(log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


class ModelFactory(Protocol[Model_co]):
    def __call__(self, path: Path) -> tuple[Model_co, State]:
        ...


class OptimizerFactory(Protocol[Model_contra]):
    def __call__(self, model: Model_contra) -> Optimizer:
        ...


class SchedulerFactory(Protocol):
    def __call__(self, optimizer: Optimizer) -> Scheduler:
        ...


class TrainStep(Protocol[Model_contra]):
    def __call__(
        self,
        *,
        model: Model_contra,
        batch: Any,
        state: State,
        optimizer_idx: int,
    ) -> tuple[Tensor | None, dict[str, float]]:
        ...


class EvalFn(Protocol[Model_contra]):
    def __call__(self, *, model: Model_contra, device: str, state: State):
        ...


class Logger(Protocol):
    def __call__(
        self,
        *,
        data: dict,
    ):
        ...


def _save_ckpt(
    *,
    path: Path,
    model: nn.Module,
    state: State,
    optimizers: list[Optimizer],
    schedulers: list[Scheduler],
):
    torch.save(
        dict(
            model=model.state_dict(),
            optimizers=[optimizer.state_dict() for optimizer in optimizers],
            schedulers=[scheduler.state_dict() for scheduler in schedulers],
            state=state.asdict(),
        ),
        path,
    )
    _logger.info(f"{path} saved.")


def _load_optimizers_and_schedulers_(
    *,
    path: Path,
    optimizers: list[Optimizer],
    schedulers: list[Scheduler],
):
    path = Path(path)

    if not path.exists():
        return

    ckpt = torch.load(path, map_location="cpu")

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


def _get_named_modules(module, attrname):
    for name, module in module.named_modules():
        if hasattr(module, attrname):
            yield name, module


def _flatten(d):
    records = pd.json_normalize(d).to_dict(orient="records")
    return records[0] if records else {}


def gather_attribute(module, attrname, delete=True, prefix=True):
    ret = {}
    for name, module in _get_named_modules(module, attrname):
        ret[name] = getattr(module, attrname)
        if delete:
            delattr(module, attrname)
    if prefix:
        ret = {attrname: ret}
    ret = _flatten(ret)
    # remove consecutive dots
    ret = {re.sub(r"\.+", ".", k): v for k, v in ret.items()}
    return ret


def dispatch_attribute(
    module,
    attrname,
    value,
    filter_fn: Callable[[nn.Module], bool] | None = None,
):
    for _, module in _get_named_modules(module, attrname):
        if filter_fn is None or filter_fn(module):
            setattr(module, attrname, value)


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
    model_factory: ModelFactory,
    optimizer_factories: list[OptimizerFactory],
    train_dl: DataLoader,
    train_step: TrainStep,
    eval_fn: EvalFn,
    ckpt_path: Path,
    max_iter: int = 10_000,
    eval_every: int = 1_000,
    scheduler_factories: list[SchedulerFactory | None] = [],
    save_every: int | None = None,
    device: str = "cuda" if torch.cuda.is_available else "cpu",
    logger: Logger = lambda data: _logger.info(str(data)),
    max_grad_norm: float = 10,
):
    save_every = save_every or eval_every

    # Set up random seeds
    random.seed(0)
    torch.manual_seed(0)

    # Load model
    model, state = model_factory(ckpt_path)
    model = model.to(device)
    model.train()

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

    _load_optimizers_and_schedulers_(
        path=ckpt_path,
        optimizers=optimizers,
        schedulers=schedulers,
    )

    events = []

    # Training loop
    for state.iteration, batch in zip(
        range(state.iteration + 1, max_iter + 1),
        _make_infinite_epochs(train_dl),
    ):
        total_elapsed_time = 0
        batch = to_device(batch, device)

        log_data: Any = dict(iteration=state.iteration)
        for optimizer_idx, (optimizer, scheduler) in enumerate(
            zip(optimizers, schedulers)
        ):
            torch.cuda.synchronize()
            start_time = time.time()

            loss, stats = train_step(
                model=model,
                batch=batch,
                state=state,
                optimizer_idx=optimizer_idx,
            )

            if loss is None:
                # Here we allow skip optimizers, it's useful if we want to skip discriminators
                # in the begining of GAN training.
                continue

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(
                sum([group["params"] for group in optimizer.param_groups], []),
                max_norm=max_grad_norm,
            )
            optimizer.step()
            scheduler.step()

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            total_elapsed_time += elapsed_time

            log_data.update(
                _flatten(
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

        log_data["elapsed_time"] = total_elapsed_time
        logger(data=log_data)

        command = _non_blocking_input()

        if "@" in command:
            what, when = command.split("@")
            try:
                events.append((what, int(when)))
                _logger.info(f"Event {command} registered.")
            except Exception as e:
                _logger.error(e)
            command = ""

        # commands are the current command plus the triggerd events
        events = [e for e in events if e[1] >= state.iteration]
        commands = [command] + [e[0] for e in events if e[1] == state.iteration]

        for command in commands:
            if command == "event show":
                msg = "Events:\n" + "\n".join(["@".join(map(str, e)) for e in events])
                _logger.info(msg)

            if command == "event clear":
                events.clear()

            if "time" in command:
                tgt_iter = max_iter
                if " to " in command:
                    try:
                        tgt_iter = int(command.split(" to ")[-1])
                    except Exception as e:
                        _logger.error(e)
                remaining_time = int((tgt_iter - state.iteration) * total_elapsed_time)
                _logger.info(humanize.precisedelta(remaining_time))

            if state.iteration % save_every == 0 or command in ["save", "quit"]:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                _save_ckpt(
                    path=ckpt_path,
                    model=model,
                    state=state,
                    optimizers=optimizers,
                    schedulers=schedulers,
                )

            if state.iteration % eval_every == 0 or command == "eval":
                model.eval()
                eval_fn(model=model, state=state, device=device)
                model.train()

            if command == "quit":
                return
