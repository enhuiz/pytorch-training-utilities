import json
import logging
from logging import StreamHandler
from pathlib import Path

from coloredlogs import ColoredFormatter

from .config import Config
from .distributed import global_leader_only, global_rank, local_rank

_logger = logging.getLogger(__name__)


@global_leader_only
def setup_logging(cfg: Config):
    handlers = []
    stdout_handler = StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    formatter = ColoredFormatter(
        f"%(asctime)s - %(name)s - %(levelname)s - GR={global_rank()};LR={local_rank()} - \n%(message)s"
    )
    stdout_handler.setFormatter(formatter)
    handlers.append(stdout_handler)
    if cfg.log_dir is not None:
        filename = Path(cfg.log_dir) / f"log.txt"
        filename.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename, mode="a")
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - \n%(message)s",
        handlers=handlers,
    )


class Logger:
    def __init__(self, cfg: Config):
        pass

    @global_leader_only(default=None)
    def __call__(self, data: dict):
        raise NotImplementedError


class DefaultLogger(Logger):
    def __init__(self, cfg: Config):
        setup_logging(cfg)

    @global_leader_only(default=None)
    def __call__(self, data: dict):
        return _logger.info(json.dumps(data, indent=2, default=str))


class WandbLogger(Logger):
    @global_leader_only(default=None)
    def __init__(self, cfg: Config):
        import wandb

        self.wandb = wandb
        wandb.init(
            project=Path.cwd().name,
            group=cfg.cfg_name,
            config=cfg.as_dict(),
            name=cfg.run_name,
        )

    @global_leader_only(default=None)
    def __call__(self, data: dict):
        self.wandb.log(data, step=data["global_step"])


class WandbWithDefaultLogger(Logger):
    @global_leader_only(default=None)
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.loggers = [DefaultLogger(cfg), WandbLogger(cfg)]

    @global_leader_only(default=None)
    def __call__(self, data: dict):
        for logger in self.loggers:
            logger(data)
