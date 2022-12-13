import sys
import time
import json
import subprocess
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass, asdict
from functools import cached_property


@dataclass
class HParams:
    project: str = "project"
    experiment: str = "experiment"
    log_root: Path = Path("log")
    ckpt_root: Path = Path("ckpt")

    @property
    def relative_path(self):
        return Path(self.project, self.experiment)

    @property
    def ckpt_path(self):
        return (self.ckpt_root / self.relative_path).with_suffix(".ckpt")

    @property
    def log_dir(self):
        return self.log_root / self.relative_path / str(self.start_time)

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

    def __str__(self):
        return self.dumps()

    def __repr__(self):
        return str(self)

    def dumps(self):
        data = {k: getattr(self, k) for k in dir(self) if not k.startswith("__")}
        data = {k: v for k, v in data.items() if not callable(v)}
        return json.dumps(data, indent=2, default=str)

    def dump(self, path=None):
        if path is None:
            path = self.log_dir / "hparams.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.dumps())

    @classmethod
    def from_cli(cls):
        cli_hp = OmegaConf.from_cli([s for s in sys.argv if "=" in s])
        # Replace argv to ensure there are no omegaconf options, for compatibility with argparse.
        sys.argv = [s for s in sys.argv if "=" not in s]

        if cli_hp.get("help"):
            print(f"Configurable hyperparameters with their default values:")
            print(json.dumps(asdict(cls()), indent=2, default=str))
            exit()

        if "yaml" in cli_hp:
            yaml_hp = OmegaConf.load(cli_hp.yaml)
            yaml_path = Path(cli_hp.yaml).absolute()
            experiment = Path(*yaml_path.relative_to(Path.cwd()).parts[1:])
            experiment = experiment.with_suffix("")
            yaml_hp.setdefault("experiment", experiment)
            cli_hp.pop("yaml")
        else:
            yaml_hp = {}

        obj = cls(**dict(OmegaConf.merge(cls, yaml_hp, cli_hp)))

        return obj


if __name__ == "__main__":
    hp = HParams.from_cli()
    print(hp)
