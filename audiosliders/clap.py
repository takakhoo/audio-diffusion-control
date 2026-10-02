"""CLAP audio and text embeddings (LAION, music checkpoint) for evaluation and discovery."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

CLAP_ID = "laion/larger_clap_music_and_speech"
CLAP_SR = 48_000


class Clap:
    def __init__(self, device: str = "cuda", model_id: str = CLAP_ID):
        from transformers import ClapModel, ClapProcessor

        self.device = torch.device(device)
        self.model = ClapModel.from_pretrained(model_id).to(self.device).eval().requires_grad_(False)
        self.processor = ClapProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def text(self, texts: Sequence[str]) -> Tensor:
        tok = self.processor(text=list(texts), return_tensors="pt", padding=True).to(self.device)
        out = self.model.get_text_features(**tok)
        out = out if isinstance(out, Tensor) else out.pooler_output
        return F.normalize(out.float(), dim=-1)

    @torch.no_grad()
    def audio(self, wav: Tensor | np.ndarray, sample_rate: int, batch: int = 16) -> Tensor:
        """wav is (B, C, T) or (B, T). Returns unit-norm embeddings, (B, 512)."""
        import torchaudio.functional as AF

        wav = torch.as_tensor(wav, dtype=torch.float32)
        if wav.dim() == 3:
            wav = wav.mean(1)
        if sample_rate != CLAP_SR:
            wav = AF.resample(wav.to(self.device), sample_rate, CLAP_SR).cpu()
        outs = []
        for i in range(0, len(wav), batch):
            chunk = [w.numpy() for w in wav[i : i + batch]]
            feats = self.processor(audio=chunk, sampling_rate=CLAP_SR, return_tensors="pt").to(self.device)
            out = self.model.get_audio_features(**feats)
            out = out if isinstance(out, Tensor) else out.pooler_output
            outs.append(F.normalize(out.float(), dim=-1))
        return torch.cat(outs)
