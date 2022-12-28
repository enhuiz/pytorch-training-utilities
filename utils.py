import re
from typing import Callable

import pandas as pd
from torch import nn


def flatten_dict(d):
    records = pd.json_normalize(d).to_dict(orient="records")
    return records[0] if records else {}


def _get_named_modules(module, attrname):
    for name, module in module.named_modules():
        if hasattr(module, attrname):
            yield name, module


def gather_attribute(module, attrname, delete=True, prefix=True):
    ret = {}
    for name, module in _get_named_modules(module, attrname):
        ret[name] = getattr(module, attrname)
        if delete:
            delattr(module, attrname)
    if prefix:
        ret = {attrname: ret}
    ret = flatten_dict(ret)
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


def load_state_dict_non_strict(model, state_dict, logger=None):
    model_state_dict = model.state_dict()
    provided = set(state_dict)
    required = set(model_state_dict)
    agreed = provided & required
    for k in list(agreed):
        if model_state_dict[k].shape != state_dict[k].shape:
            agreed.remove(k)
            provided.remove(k)
    state_dict = {k: state_dict[k] for k in agreed}
    if logger is not None and (diff := provided - required):
        logger.warning(
            f"Extra parameters are found. "
            f"Provided but not required parameters: \n{diff}."
        )
    if logger is not None and (diff := required - provided):
        logger.warning(
            f"Some parameters are missing. "
            f"Required but not provided parameters: \n{diff}."
        )
    model.load_state_dict(state_dict, strict=False)
