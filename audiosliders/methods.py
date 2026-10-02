"""Ways to turn a slider position into a change in generation.

Every method takes one scale per sample and returns the pieces `StableAudio.generate`
needs. They share prompts, seeds, sampler, and step count, so differences in the output
come from the control method alone.

  lora      the trained slider (one extra low-rank path, no extra forward passes)
  guidance  add scale * eta * (v(c+) - v(c-)) at every step; this is the LoRA's training
            target applied directly, and costs two more forward passes per step
  embed     interpolate the text conditioning toward the positive or negative prompt
  dsp       post-process the unsteered clip (handled by the caller, see dsp.py)
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor

from .backbone import Cond, Predictor, StableAudio
from .lora import SliderBank


def lora(bank: SliderBank, name: str, scales: Tensor, start: float = 1.0) -> Callable[[Predictor], Predictor]:
    return lambda predict: bank.gated(predict, {name: scales}, start=start)


def guidance(
    model: StableAudio,
    prompts: Sequence[str],
    positive: str,
    negative: str,
    scales: Tensor,
    seconds: float,
    eta: float = 4.0,
    start: float = 1.0,
) -> Callable[[Predictor], Predictor]:
    pair = Cond.cat(
        [
            model.encode([f"{p}, {positive}" for p in prompts], seconds),
            model.encode([f"{p}, {negative}" for p in prompts], seconds),
        ]
    )
    weight = (scales * eta).view(-1, 1, 1)

    def wrap(predict: Predictor) -> Predictor:
        def steered(z: Tensor, t: Tensor) -> Tensor:
            v = predict(z, t)
            if float(t) > start:
                return v
            vp, vn = model.v(torch.cat([z, z]), t, pair).chunk(2)
            return v + weight * (vp - vn)

        return steered

    return wrap


def embed(
    model: StableAudio, prompts: Sequence[str], positive: str, negative: str, scales: Tensor, seconds: float
) -> Cond:
    """Conditioning moved from the base prompt toward one end of the slider.

    scale = +1 is exactly the prompt with the positive phrase appended, -1 the negative
    one, and values past 1 extrapolate along the same line.
    """
    base = model.encode(prompts, seconds)
    pos = model.encode([f"{p}, {positive}" for p in prompts], seconds)
    neg = model.encode([f"{p}, {negative}" for p in prompts], seconds)
    s = scales.view(-1, 1, 1)
    cross = base.cross + s.clamp(min=0) * (pos.cross - base.cross) + (-s).clamp(min=0) * (neg.cross - base.cross)
    return Cond(cross, base.glob)
