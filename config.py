import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

console = Console()


def _as_rich_table(d, key="Key", value="Value"):
    table = Table()
    table.add_column(key, justify="left", style="cyan")
    table.add_column(value, justify="left", style="magenta")
    for k, v in d.items():
        table.add_row(k, str(v))
    return table


def _print_rich_table(d, key="Key", value="Value"):
    console.print(_as_rich_table(d, key, value))


@dataclass(frozen=True)
class Config:
    cfg_name: str = "my-cfg"
    log_root: Path = Path("logs")
    ckpt_root: Path = Path("ckpts")

    device: str = "cuda"

    max_iter: int = 100_000
    max_grad_norm: float | None = None

    eval_every: int = 1_000
    save_artifacts_every: int | None = 100
    save_ckpt_every: int | None = None
    strict_loading: bool = True

    save_on_oom: bool = True
    save_on_quit: bool = True

    @property
    def relpath(self):
        return Path(self.cfg_name)

    @property
    def cfg_relpath(self):
        return None

    @property
    def ckpt_dir(self):
        return self.ckpt_root / self.relpath

    @property
    def log_dir(self):
        return self.log_root / self.relpath / str(self.start_time)

    @cached_property
    def start_time(self):
        return int(time.time())

    @cached_property
    def git_commit(self):
        try:
            cmd = "git rev-parse HEAD"
            return subprocess.check_output(cmd.split()).decode("utf8").strip()
        except:
            return ""

    @cached_property
    def git_status(self):
        try:
            cmd = "git status"
            return subprocess.check_output(cmd.split()).decode("utf8").strip()
        except:
            return ""

    def as_dict(self):
        data = {k: getattr(self, k) for k in dir(self) if not k.startswith("__")}
        data = {k: v for k, v in data.items() if not callable(v)}
        return data

    def as_json(self):
        data = self.as_dict()
        return json.dumps(data, indent=2, default=str)

    def save(self, path=None):
        """
        Save config to a json file, by default in the log directory.
        """
        if path is None:
            path = self.log_dir / "cfg.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.as_json())

    @staticmethod
    def _is_cfg_argv(s):
        return "=" in s and "--" not in s

    @classmethod
    def from_cli(cls):
        cli_cfg = OmegaConf.from_cli([s for s in sys.argv if cls._is_cfg_argv(s)])

        # Replace argv to ensure there are no omegaconf options, for compatibility with argparse.
        sys.argv = [s for s in sys.argv if not cls._is_cfg_argv(s)]

        if cli_cfg.get("help"):
            _print_rich_table(asdict(cls()), key="Key", value="Default")
            exit()

        if "yaml" in cli_cfg:
            yaml_cfg = OmegaConf.load(cli_cfg.yaml)
            yaml_path = Path(cli_cfg.yaml).absolute()
            cfg_name = Path(*yaml_path.relative_to(Path.cwd()).parts[1:])
            cfg_name = cfg_name.with_suffix("")
            yaml_cfg.setdefault("cfg_name", cfg_name)
            cli_cfg.pop("yaml")
        else:
            yaml_cfg = {}

        obj = cls(**dict(OmegaConf.merge(cls, yaml_cfg, cli_cfg)))

        return obj

    def __post_init__(self):
        if self.cfg_relpath is not None:
            raise RuntimeError("cfg_relpath is deprecated.")

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.as_json()

    def print(self):
        _print_rich_table(self.as_dict(), key="Key", value="Value")


if __name__ == "__main__":
    cfg = Config.from_cli()
    cfg.print()
