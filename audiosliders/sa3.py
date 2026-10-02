"""Stable Audio 3 base checkpoints behind the same interface as StableAudio.

A rectified-flow model: z_t = (1 - t) x0 + t eps and the transformer predicts
v = eps - x0, with t = 1 pure noise and sampling running from t = 1 down to 0. Latents are
(256, frames) at 44100 / 4096 = 10.77 Hz and decode to 44.1 kHz stereo. As in the official
sampler, a clip gets 6 s of extra frames after the requested duration (172 frames for 10 s)
and the decoded audio is cut back to the duration.

Sampling follows the stable-audio-3 package for the base (not post-trained) models: 50 Euler
steps spaced uniformly in log-SNR from -6.2 to 2.0, and guidance 7.0 applied with the
conditional-minus-unconditional update projected orthogonal to the conditional estimate of x0.
The model code comes from that package (github.com/Stability-AI/stable-audio-3, installed with
--no-deps); the checkpoints are not in the diffusers layout. Transformer weights stay float32
and run under bfloat16 autocast, which keeps the residual stream and the time embedding in
float32.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .backbone import Cond, Predictor, StableAudio, seeded_noise

REPO = {"medium": "stabilityai/stable-audio-3-medium-base", "small": "stabilityai/stable-audio-3-small-music-base"}
HEADROOM = 6.0
LOGSNR_START, LOGSNR_END = -6.2, 2.0


class StableAudio3:
    def __init__(self, variant: str = "medium", device: str = "cuda"):
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from stable_audio_3.factory import create_diffusion_cond_from_config
        from stable_audio_3.loading_utils import copy_state_dict
        from stable_audio_3.models import transformer

        # Without flash-attn the decoder's windowed attention falls back to a torch.compile'd flex_attention,
        # which needs a writable compiler cache. None selects the package's exact chunked attention instead.
        transformer.flex_attention_compiled = None
        root = Path(snapshot_download(REPO[variant], allow_patterns=["model*", "t5gemma-b-b-ul2/*"]))
        config = json.loads((root / "model_config.json").read_text())
        for c in config["model"]["conditioning"]["configs"]:
            if "subfolder" in c["config"]:
                # The config names the gated post-trained repo; the base repo ships the same text encoder.
                c["config"]["model_path"] = str(root / c["config"].pop("subfolder"))
        wrapper = create_diffusion_cond_from_config(config)
        copy_state_dict(wrapper, load_file(str(root / "model.safetensors")))

        self.name = f"sa3-{variant}"
        self.steps, self.guidance = 50, 7.0
        self.device = torch.device(device)
        self.dit = wrapper.model.model.to(self.device).eval().requires_grad_(False)
        self.vae = wrapper.pretransform.to(self.device).eval().requires_grad_(False)
        self.vae.model.pretransform.enable_grad = True  # a reshape; lets gradients through decode
        self.conditioner = wrapper.conditioner.to(self.device).eval().requires_grad_(False)
        self.sample_rate = int(wrapper.sample_rate)
        self.hop = int(self.vae.downsampling_ratio)
        self.channels = int(wrapper.io_channels)
        self.amp = torch.bfloat16
        self._local = int(config["model"]["diffusion"]["config"]["local_add_cond_dim"])
        self._text: dict[str, Tensor] = {}
        self.fit_peak = StableAudio.fit_peak

    def frames(self, seconds: float) -> int:
        return math.ceil(int(seconds * self.sample_rate) / self.hop) + int(HEADROOM * self.sample_rate / self.hop)

    def latent_shape(self, seconds: float) -> tuple[int, int]:
        return (self.channels, self.frames(seconds))

    @staticmethod
    def diffuse(x0: Tensor, eps: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        t = t.view(-1, 1, 1)
        return (1 - t) * x0 + t * eps, eps - x0

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], seconds: float) -> Cond:
        """Cross-attention gets 256 text tokens and the duration token; the duration is also the global input."""
        missing = [p for p in dict.fromkeys(prompts) if p not in self._text]
        if missing:
            tokens, _ = self.conditioner.conditioners["prompt"](missing, self.device)
            self._text.update(zip(missing, tokens.float()))
        text = torch.stack([self._text[p] for p in prompts])
        total, _ = self.conditioner.conditioners["seconds_total"]([seconds] * len(prompts), self.device)
        return Cond(torch.cat([text, total.float()], dim=1), total[:, 0].float())

    def v(self, z: Tensor, t: Tensor, cond: Cond) -> Tensor:
        """Raw transformer call. z is (B, 256, L); t is a scalar or (B,) in (0, 1]."""
        t = torch.as_tensor(t, device=z.device, dtype=torch.float32).expand(z.shape[0])
        # The inpainting inputs (mask and masked latent), all zero for plain generation.
        local = z.new_zeros(z.shape[0], self._local, z.shape[-1])
        with torch.autocast(self.device.type, dtype=self.amp):
            out = self.dit(z, t, cross_attn_cond=cond.cross, global_embed=cond.glob, local_add_cond=local,
                           use_checkpointing=False)
        return out.float()

    def cfg(self, cond: Cond, guidance: float) -> Predictor:
        if guidance == 1.0:
            return lambda z, t: self.v(z, t, cond)
        both = Cond.cat([cond, cond.null()])

        def predict(z: Tensor, t: Tensor) -> Tensor:
            vc, vu = self.v(torch.cat([z, z]), t, both).chunk(2)
            unit = F.normalize(z - t * vc, dim=(-2, -1))
            step = vc - vu
            step = step - (step * unit).sum(dim=(-2, -1), keepdim=True) * unit
            return vc + (guidance - 1.0) * step

        return predict

    def schedule(self, steps: int) -> Tensor:
        u = torch.linspace(1.0, 0.0, steps + 1, device=self.device)
        t = torch.sigmoid(u * (LOGSNR_END - LOGSNR_START) - LOGSNR_END)
        t[0], t[-1] = 1.0, 0.0
        return t

    def sample(self, predict: Predictor, noise: Tensor, steps: int | None = None, stop: int | None = None,
               **_) -> tuple[Tensor, Tensor]:
        """Euler steps from noise. With stop=k, returns the state after k steps and its time."""
        ts = self.schedule(steps or self.steps)
        x = noise
        last = len(ts) - 1 if stop is None else stop
        for i in range(last):
            x = x + (ts[i + 1] - ts[i]) * predict(x, ts[i])
        return x, ts[last]

    @torch.no_grad()
    def encode_audio(self, audio: Tensor) -> Tensor:
        """Stereo waveforms (B, 2, T) at 44.1 kHz to latents shaped (B, 256, ceil(T / 4096))."""
        return self.vae.encode(audio.to(self.device, torch.float32)).float()

    def decode(self, z: Tensor, seconds: float | None = None) -> Tensor:
        """Latents to stereo waveforms, shape (B, 2, T). The decoder draws noise, so it is seeded here."""
        with torch.random.fork_rng(devices=[self.device]):
            torch.manual_seed(0)
            audio = torch.cat([self.vae.decode(z[i : i + 8].float()) for i in range(0, len(z), 8)])
        if seconds is not None:
            audio = audio[..., : int(seconds * self.sample_rate)]
        return audio

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], seeds: Sequence[int], seconds: float = 10.0,
                 steps: int | None = None, guidance: float | None = None,
                 wrap: Callable[[Predictor], Predictor] | None = None, latents: bool = False,
                 cond: Cond | None = None, **_) -> Tensor:
        cond = cond if cond is not None else self.encode(prompts, seconds)
        predict = self.cfg(cond, self.guidance if guidance is None else guidance)
        if wrap is not None:
            predict = wrap(predict)
        noise = seeded_noise(seeds, self.latent_shape(seconds), self.device)
        z, _ = self.sample(predict, noise, steps=steps or self.steps)
        return z if latents else self.fit_peak(self.decode(z, seconds))
