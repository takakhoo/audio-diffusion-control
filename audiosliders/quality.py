"""Reference-free quality scores per clip, from Meta's Audiobox Aesthetics predictor.

Four axes on a 1 to 10 scale: content enjoyment (ce), content usefulness (cu), production
complexity (pc), production quality (pq). A slider that keeps working music should hold
ce and pq roughly flat as it moves; a slider that wrecks the clip shows up as a drop.
"""

from __future__ import annotations

import numpy as np
import torch


class Aesthetics:
    def __init__(self):
        from audiobox_aesthetics.infer import initialize_predictor

        self.predictor = initialize_predictor()

    @torch.no_grad()
    def __call__(self, audio: np.ndarray | torch.Tensor, sample_rate: int, batch: int = 16) -> list[dict[str, float]]:
        """audio is (B, C, T). Returns one dict per clip with keys ce, cu, pc, pq."""
        audio = torch.as_tensor(audio, dtype=torch.float32)
        out = []
        for i in range(0, len(audio), batch):
            items = [dict(path=a, sample_rate=sample_rate) for a in audio[i : i + batch]]
            out += [{k.lower(): float(v) for k, v in r.items()} for r in self.predictor.forward(items)]
        return out
