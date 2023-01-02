"""
Inspired by https://github.com/k2-fsa/icefall/blob/master/icefall/diagnostics.py
"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from . import artifacts
from .distributed import global_leader_only

_logger = logging.getLogger(__name__)


class Diagnostic:
    def __init__(
        self,
        module: nn.Module,
        tag="module",
        max_pca_dim=512,
        percentiles=np.linspace(0, 1, 10),
    ):
        self._module = module
        self._handlers = []
        self._history: defaultdict[str, defaultdict[str, int | Tensor]]
        self._history = defaultdict(lambda: defaultdict(lambda: 0))
        self._tag = tag
        self._max_pca_dim = max_pca_dim
        self._percentiles = percentiles

    @staticmethod
    def _get_type(name):
        if "grad" in name:
            return "grad"
        elif "output" in name:
            return "output"
        elif "param" in name:
            return "param"
        raise NotImplementedError(name)

    @property
    def dataframe(self):
        df = pd.DataFrame(self._history)
        for col in df.columns:
            if col not in ["min", "max", "size"]:
                df[col] /= df["cnt"]

        rows = []

        for stats in df.columns:
            if stats in ["size", "cnt"]:
                continue

            for name, s in df.iterrows():
                v = s[stats]

                if not isinstance(v, Tensor):
                    v = None

                rows.append(
                    dict(
                        name=name,
                        type=self._get_type(name),
                        stats=stats,
                        size=s["size"],
                        norm=v if v is None else v.norm().item(),
                        mean=v if v is None else v.mean().item(),
                    )
                    | (
                        {
                            f"p{i}": v
                            for i, v in enumerate(np.percentile(v, self._percentiles))
                        }
                        if v is not None
                        else {}
                    )
                )

        df = pd.DataFrame(rows)

        if len(df) > 0:
            df = df.sort_values(["type", "stats", "name"])

        return df

    def _accumulate_by_dim(self, name, x, dim):
        x = x.transpose(dim, -1)
        while x.dim() <= 1:
            x = x.unsqueeze(0)
        x = x.flatten(0, -2)  # (n d)

        size = x.shape[-1]

        a = self._history["abs"][name]
        if isinstance(a, Tensor) and len(a) != size:
            # Size mismatch, skip.
            return

        if size < self._max_pca_dim and len(x) > 1:
            self._history["pca"][name] += torch.pca_lowrank(x)[1].cpu()

        self._history["abs"][name] += x.abs().sum(0).cpu()
        self._history["pos"][name] += x.clamp_min(0).sum(0).cpu()
        self._history["val"][name] += x.sum(0).cpu()
        self._history["var"][name] += x.pow(2).sum(0).cpu()
        self._history["cnt"][name] += len(x)

        a = self._history["min"][name]
        b = x.min(0).values.cpu()
        if not isinstance(a, Tensor):
            a = torch.full_like(b, float("inf"))
        self._history["min"][name] = torch.minimum(a, b)

        a = self._history["max"][name]
        b = x.max(0).values.cpu()
        if not isinstance(a, Tensor):
            a = torch.full_like(b, -float("inf"))
        self._history["max"][name] = torch.maximum(a, b)
        self._history["size"][name] = size

    @torch.no_grad()
    def _accumulate(self, name, x):
        for d in range(x.dim()):
            self._accumulate_by_dim(name + f"/dim_{d}", x, d)

    @global_leader_only
    def attach(self):
        # If not detached, don't attach
        if len(self._handlers) > 0:
            return

        _logger.info(f"Attaching diagnostic for module {self._tag}.")

        for name, module in self._module.named_modules():

            def forward_hook(m, i, o, name=name or self._tag):
                if not isinstance(o, tuple):
                    o = (o,)

                for i, oi in enumerate(o):
                    if not isinstance(oi, Tensor):
                        continue

                    self._accumulate(f"{name}/output/{i}", oi)

            self._handlers.append(module.register_forward_hook(forward_hook))

            def backward_hook(m, i, o, name=name or self._tag):
                if not isinstance(o, tuple):
                    o = (o,)

                for i, oi in enumerate(o):
                    if not isinstance(oi, Tensor):
                        continue

                    self._accumulate(f"{name}/grad/{i}", oi)

            self._handlers.append(module.register_full_backward_hook(backward_hook))

        for name, param in self._module.named_parameters():
            if not param.requires_grad:
                continue

            def hook(grad, name=name, param=param):
                self._accumulate(f"{name}/param", param)
                self._accumulate(f"{name}/grad", grad)

            self._handlers.append(param.register_hook(hook))

    @global_leader_only
    def detach(self):
        for handler in self._handlers:
            handler.remove()
        self._handlers.clear()
        self._history.clear()

    @global_leader_only
    def save(self, detach=True):
        path = artifacts.get_path(f"diagnostic/{self._tag}", ".csv")
        self.to_csv(path, index=False)
        if detach:
            self.detach()

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
    diagnostic.attach()
    model(torch.randn(10)).sum().backward()
    model(torch.randn(10)).sum().backward()
    model(torch.randn(10)).sum().backward()
    print(diagnostic.to_markdown(index=False, floatfmt=".3g"))
