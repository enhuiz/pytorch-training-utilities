"""
Inspired by https://github.com/k2-fsa/icefall/blob/master/icefall/diagnostics.py
"""

from collections import defaultdict

import pandas as pd
import torch
from torch import nn


class Diagnostic:
    def __init__(self, module: nn.Module):
        self._module = module
        self._handlers = []
        self._history = defaultdict(lambda: defaultdict(lambda: 0.0))

    def _accumulate(self, name, value):
        if value.dim() == 0:
            value = value[None, None]
        elif value.dim() == 1:
            value = value[None]
        else:
            value = value.flatten(1)

        self._history[name]["min"] += value.min(1).values.mean().item()
        self._history[name]["max"] += value.max(1).values.mean().item()
        self._history[name]["mean"] += value.mean().item()
        self._history[name]["var"] += value.var(1).mean().item()
        self._history[name]["cnt"] += 1

    def hook(self):
        # If not released, don't hook
        if len(self._handlers) > 0:
            return

        for name, module in self._module.named_modules():

            def forward_hook(m, i, o, name=name or "top"):
                if not isinstance(o, tuple):
                    o = (o,)

                for i, oi in enumerate(o):
                    if oi is None:
                        continue

                    self._accumulate(f"{name}/out_{i}", oi)

            self._handlers.append(module.register_forward_hook(forward_hook))

            def backward_hook(m, i, o, name=name or "top"):
                if not isinstance(o, tuple):
                    o = (o,)

                for i, oi in enumerate(o):
                    if oi is None:
                        continue

                    self._accumulate(f"{name}/grad_{i}", oi)

            self._handlers.append(module.register_full_backward_hook(backward_hook))

        for name, param in self._module.named_parameters():
            if not param.requires_grad:
                continue

            def hook(grad, name=name, param=param):
                self._accumulate(f"{name}/param", param)
                self._accumulate(f"{name}/grad", grad)

            self._handlers.append(param.register_hook(hook))

    def release(self):
        for handler in self._handlers:
            handler.remove()
        self._handlers.clear()
        self._history.clear()

    @property
    def dataframe(self):
        df = pd.DataFrame(self._history).T
        for col in df.columns:
            df[col] /= df["cnt"]
        if "cnt" in df:
            del df["cnt"]
        return df

    @property
    def to_csv(self):
        return self.dataframe.to_csv

    @property
    def to_markdown(self):
        return self.dataframe.to_markdown


if __name__ == "__main__":
    model = nn.Linear(10, 10)
    diagnostic = Diagnostic(model)
    model.bias.requires_grad_(False)
    diagnostic.hook()
    model(torch.randn(10)).sum().backward()
    model(torch.randn(10)).sum().backward()
    model(torch.randn(10)).sum().backward()
    print(diagnostic.to_markdown())
    diagnostic.release()
