import sys
import inspect
import pandas as pd
from textwrap import shorten
from omegaconf import OmegaConf
from pathlib import Path


def _get_class_name(self):
    module = inspect.getmodule(self)
    if module is None:
        module_name = "unknown"
    else:
        module_name = module.__name__
    return f"{module_name}.{self.__name__}"


def _d2md(d: dict, sort=True, tag="value"):
    items = d.items()
    items = sorted(items) if sort else items
    rows = [{"name": k, tag: shorten(str(v), width=60)} for k, v in items]
    df = pd.DataFrame(rows).sort_values("name")
    assert isinstance(df, pd.DataFrame)
    return df.to_markdown(index=False, tablefmt="psql")


def init_hp(cls):
    cli_hp = OmegaConf.from_cli([s for s in sys.argv if "=" in s])
    # replace argv to make sure there is no omegaconf options
    sys.argv = [s for s in sys.argv if "=" not in s]

    if cli_hp.get("help"):
        print(f"Configurable hyperparams (use xxx=yyy to configure):")
        print(_d2md(vars(cls()), tag="default"))
        exit()

    if "config" in cli_hp:
        cfg_hp = OmegaConf.load(cli_hp.config)
        cfg_path = Path(cli_hp.config)
        if cfg_path.parts[0] != "config":
            raise ValueError("Config must come from the config folder.")
        cfg_root = getattr(cls, "cfg_root", "config")
        cfg_hp.setdefault("name", cfg_path.relative_to(cfg_root).with_suffix(""))
        cli_hp.pop("config")
    else:
        cfg_hp = {}

    def __repr__(self):
        md = _d2md({k: getattr(self, k) for k in dir(self) if not k.startswith("__")})
        return f"{_get_class_name(type(self))}\n{md}"

    cls.__repr__ = __repr__
    obj = cls(**dict(OmegaConf.merge(cls, cfg_hp, cli_hp)))

    return obj
