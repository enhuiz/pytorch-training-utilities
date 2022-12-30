import json
import logging
import random
import selectors
import sys
from functools import cache
from typing import Protocol

import humanize
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .distributed import local_leader_only
from .engines import Engine, Engines, TrainStepFn

_logger = logging.getLogger(__name__)
_engines: Engines


def get_global_step():
    try:
        return _engines.global_step
    except:
        return None


def get_cfg():
    try:
        return _engines.cfg
    except:
        raise RuntimeError(
            "Trainer has not been setup, please call trainer.setup() first."
        )


get_iteration = get_global_step


class EnginesLoader(Protocol):
    def __call__(self) -> Engines:
        ...


def load_engines(engines: dict[str, Engine] | nn.ModuleDict, config: Config):
    engines = Engines({**engines})
    engines.setup(config)
    engines.load_checkpoint()
    return engines


class EvalFn(Protocol):
    def __call__(self, *, engines: Engines):
        ...


class Logger(Protocol):
    def __call__(self, *, data: dict):
        ...


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


@local_leader_only
def logger(data):
    return _logger.info(json.dumps(data, indent=2, default=str))


def train(
    engines_loader: EnginesLoader,
    train_dl: DataLoader,
    train_step_fn: TrainStepFn,
    eval_fn: EvalFn,
    logger: Logger = logger,
):
    # Set up random seeds
    random.seed(0)
    torch.manual_seed(0)

    engines = engines_loader()
    cfg = engines.cfg

    # Setup global engines
    global _engines
    _engines = engines

    events = []

    # Pre-loop command
    command = _non_blocking_input()
    if command in ["eval", "eval_quit"]:
        engines.eval()
        eval_fn(engines=engines)
        engines.train()
    if command in ["quit", "eval_quit"]:
        return

    # Training loop
    for batch in _make_infinite_epochs(train_dl):
        if engines.global_step >= cfg.max_iter:
            break

        stats = engines.step(fn=train_step_fn, batch=batch)
        total_elapsed_time = stats.get("total_elapsed_time", 0)
        logger(data=stats)

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
        events = [e for e in events if e[1] >= engines.global_step]
        commands = [command] + [e[0] for e in events if e[1] == engines.global_step]

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
                remaining_iters = target_iter - engines.global_step + 1
                remaining_time = int(remaining_iters * total_elapsed_time)
                _logger.info(humanize.precisedelta(remaining_time))

            save_every = cfg.save_model_every or cfg.eval_every

            saving_commands = ["save"]

            if cfg.save_on_quit:
                saving_commands.append("quit")

            if engines.global_step % save_every == 0 or command in saving_commands:
                engines.save_checkpoint()

            if engines.global_step % cfg.eval_every == 0 or command in ["eval"]:
                engines.eval()
                eval_fn(engines=engines)
                engines.train()

            if command in ["quit"]:
                return
