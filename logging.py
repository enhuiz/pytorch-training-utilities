import logging
from logging import StreamHandler
from pathlib import Path

from coloredlogs import ColoredFormatter


def setup_logging(log_dir: str | Path | None = "log", log_level="info"):
    handlers = []
    stdout_handler = StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - \n%(message)s"
    )
    stdout_handler.setFormatter(formatter)
    handlers.append(stdout_handler)
    if log_dir is not None:
        filename = Path(log_dir) / f"log.txt"
        filename.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename, mode="a")
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging.getLevelName(log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - \n%(message)s",
        handlers=handlers,
    )
