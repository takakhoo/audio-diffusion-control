"""Low-rank adapters whose strength is set at call time. One adapter per slider.

A SliderBank wraps chosen nn.Linear layers of the transformer. Every wrapped layer
computes

    y = W x + sum_k scale_k * (alpha_k / rank_k) * B_k A_k x

where scale_k is read from the bank on each forward pass. A scale can be a float or
a per-sample tensor, so one batch can hold the same clip at several slider positions.
"""

from __future__ import annotations

import json
import math
import re
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor, nn

# Layer names differ by backbone: Stable Audio blocks are transformer_blocks.N.{attn1,attn2,ff},
# ACE-Step blocks are layers.N.{self_attn,cross_attn,mlp}. Each preset matches both.
_BLOCK = r"(transformer_blocks|layers)\.\d+\."
_ATTN = r"(to_q|to_k|to_v|to_out\.0)$"
_SELF, _CROSS = rf"(attn1|self_attn)\.{_ATTN}", rf"(attn2|cross_attn)\.{_ATTN}"
_FF = r"(ff\.net\.(0\.proj|2)|mlp\.(gate_proj|up_proj|down_proj))$"
TARGETS = {
    "xattn": _BLOCK + _CROSS,
    "self": _BLOCK + _SELF,
    "attn": rf"{_BLOCK}({_SELF}|{_CROSS})",
    "ff": _BLOCK + _FF,
    "noxattn": rf"{_BLOCK}({_SELF}|{_FF})",
    "all": rf"{_BLOCK}({_SELF}|{_CROSS}|{_FF})",
}


class SliderLinear(nn.Module):
    def __init__(self, base: nn.Linear, scales: dict):
        super().__init__()
        self.base = base
        self.down = nn.ModuleDict()
        self.up = nn.ModuleDict()
        self.gain: dict[str, float] = {}
        self._scales = scales

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        for name, scale in self._scales.items():
            if name not in self.down:
                continue
            if isinstance(scale, Tensor):
                # Classifier-free guidance stacks copies of the batch; tile the scales to match.
                scale = scale.repeat(x.shape[0] // scale.shape[0]).view(-1, *([1] * (x.dim() - 1)))
                scale = scale.to(x.dtype)
            elif scale == 0:
                continue
            out = out + self.up[name](self.down[name](x)) * (scale * self.gain[name])
        return out


class SliderBank:
    """Attaches named sliders to a transformer and controls their scales."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.scales: dict[str, float | Tensor] = {}
        self.meta: dict[str, dict] = {}
        self._layers: dict[str, SliderLinear] = {}

    def _wrap(self, path: str) -> SliderLinear:
        if path in self._layers:
            return self._layers[path]
        parent_path, _, leaf = path.rpartition(".")
        parent = self.model.get_submodule(parent_path)
        layer = SliderLinear(getattr(parent, leaf), self.scales)
        setattr(parent, leaf, layer)
        self._layers[path] = layer
        return layer

    def add(self, name: str, rank: int = 4, alpha: float = 1.0, targets: str = "all", **meta) -> list[nn.Parameter]:
        if name in self.meta:
            raise ValueError(f"slider {name!r} already attached")
        pattern = re.compile(TARGETS.get(targets, targets))
        paths = []
        for path, module in list(self.model.named_modules()):
            if path.endswith(".base"):
                path, module = path[: -len(".base")], module
            if isinstance(module, nn.Linear) and pattern.search(path):
                paths.append(path)
        if not paths:
            raise ValueError(f"no linear layers match {targets!r}")
        params = []
        for path in paths:
            layer = self._wrap(path)
            base = layer.base
            down = nn.Linear(base.in_features, rank, bias=False)
            up = nn.Linear(rank, base.out_features, bias=False)
            nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
            nn.init.zeros_(up.weight)
            layer.down[name] = down.to(base.weight.device)
            layer.up[name] = up.to(base.weight.device)
            layer.gain[name] = alpha / rank
            params += [layer.down[name].weight, layer.up[name].weight]
        self.meta[name] = dict(rank=rank, alpha=alpha, targets=targets, **meta)
        self.scales[name] = 0.0
        return params

    def parameters(self, name: str) -> list[nn.Parameter]:
        out = []
        for layer in self._layers.values():
            if name in layer.down:
                out += [layer.down[name].weight, layer.up[name].weight]
        return out

    def set(self, **scales: float | Tensor) -> None:
        for name, value in scales.items():
            if name not in self.meta:
                raise KeyError(name)
            self.scales[name] = value

    def reset(self) -> None:
        for name in self.scales:
            self.scales[name] = 0.0

    @contextmanager
    def at(self, **scales: float | Tensor):
        before = dict(self.scales)
        self.set(**scales)
        try:
            yield self
        finally:
            self.scales.update(before)

    def gated(self, predict, scales: dict[str, float | Tensor], start: float = 1.0, end: float = 0.0):
        """Wrap a predictor so the sliders are only active for end <= t <= start.

        Leaving the first, noisiest steps untouched keeps the layout of the clip
        (the same idea as the SDEdit-style start step in Concept Sliders).
        """

        def wrapped(z: Tensor, t: Tensor) -> Tensor:
            if end <= float(t) <= start:
                with self.at(**scales):
                    return predict(z, t)
            return predict(z, t)

        return wrapped

    def state_dict(self, name: str) -> dict[str, Tensor]:
        sd = {}
        for path, layer in self._layers.items():
            if name in layer.down:
                sd[f"{path}.down"] = layer.down[name].weight.detach().cpu()
                sd[f"{path}.up"] = layer.up[name].weight.detach().cpu()
        return sd

    def save(self, name: str, file: str | Path) -> None:
        from safetensors.torch import save_file

        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(name), str(file), metadata={"slider": json.dumps(self.meta[name])})

    def load(self, name: str, file: str | Path) -> dict:
        from safetensors import safe_open

        with safe_open(str(file), framework="pt") as f:
            meta = json.loads(f.metadata()["slider"])
            sd = {k: f.get_tensor(k) for k in f.keys()}
        self.add(name, **meta)
        for path, layer in self._layers.items():
            if name in layer.down:
                layer.down[name].weight.data.copy_(sd[f"{path}.down"])
                layer.up[name].weight.data.copy_(sd[f"{path}.up"])
        return meta
