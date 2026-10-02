"""Stable Audio Open 1.0 behind a small interface: encode text, predict v, sample, decode.

Conventions used everywhere in this package:

  z_t = alpha_t * x0 + sigma_t * eps,   alpha_t = cos(pi t / 2),  sigma_t = sin(pi t / 2)
  v   = alpha_t * eps - sigma_t * x0    (what the transformer predicts)
  x0  = alpha_t * z_t - sigma_t * v
  eps = sigma_t * z_t + alpha_t * v

The samplers run in the k-diffusion form x = x0 + s * eps with s = tan(pi t / 2),
which is the same process rescaled by 1 / alpha_t.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

MODEL_ID = "stabilityai/stable-audio-open-1.0"
SIGMA_MIN, SIGMA_MAX = 0.3, 500.0

Predictor = Callable[[Tensor, Tensor], Tensor]


@dataclass
class Cond:
    """Conditioning for a batch: cross-attention tokens and the global duration embedding."""

    cross: Tensor  # (B, 130, 768): 128 T5 tokens, then seconds_start and seconds_total
    glob: Tensor  # (B, 1, 1536)

    def __len__(self) -> int:
        return self.cross.shape[0]

    def null(self) -> "Cond":
        return Cond(torch.zeros_like(self.cross), self.glob)

    def __getitem__(self, idx) -> "Cond":
        return Cond(self.cross[idx], self.glob[idx])

    @staticmethod
    def cat(conds: Sequence["Cond"]) -> "Cond":
        return Cond(torch.cat([c.cross for c in conds]), torch.cat([c.glob for c in conds]))


def t_to_sigma(t: Tensor) -> Tensor:
    return torch.tan(t * math.pi / 2)


def sigma_to_t(sigma: Tensor) -> Tensor:
    return torch.atan(sigma) * 2 / math.pi


def sigma_schedule(steps: int, device=None) -> Tensor:
    """Log-linear sigmas from SIGMA_MAX to SIGMA_MIN, then 0. Matches the released sampler config."""
    s = torch.exp(torch.linspace(math.log(SIGMA_MAX), math.log(SIGMA_MIN), steps, device=device))
    return torch.cat([s, s.new_zeros(1)])


def seeded_noise(seeds: Sequence[int], shape: tuple[int, ...], device) -> Tensor:
    """One generator per sample, so a clip depends on its own seed and not on its batch."""
    out = [torch.randn(shape, generator=torch.Generator().manual_seed(int(s))) for s in seeds]
    return torch.stack(out).to(device)


class StableAudio:
    def __init__(self, device: str = "cuda", model_id: str = MODEL_ID, half: bool = True):
        from diffusers import StableAudioPipeline
        from diffusers.models.embeddings import get_1d_rotary_pos_embed

        pipe = StableAudioPipeline.from_pretrained(model_id)
        self.device = torch.device(device)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.device).eval()
        self.projection = pipe.projection_model.to(self.device).eval()
        self.dit = pipe.transformer.to(self.device).eval()
        self.vae = pipe.vae.to(self.device).eval()
        for m in (self.text_encoder, self.projection, self.dit, self.vae):
            m.requires_grad_(False)
        self.sample_rate = int(self.vae.config.sampling_rate)
        self.hop = int(self.vae.hop_length)
        self.channels = int(self.dit.config.in_channels)
        self.amp = torch.bfloat16 if half else torch.float32
        self._rot_fn = get_1d_rotary_pos_embed
        self._rot_dim = self.dit.config.attention_head_dim // 2
        self._rot: dict[int, tuple[Tensor, Tensor]] = {}
        self._text: dict[str, Tensor] = {}

    def frames(self, seconds: float) -> int:
        return math.ceil(seconds * self.sample_rate / self.hop)

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str]) -> Tensor:
        missing = [p for p in dict.fromkeys(prompts) if p not in self._text]
        if missing:
            tok = self.tokenizer(
                missing,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.text_encoder(tok.input_ids, attention_mask=tok.attention_mask)[0]
            hidden = self.projection(text_hidden_states=hidden).text_hidden_states
            hidden = hidden * tok.attention_mask.unsqueeze(-1).to(hidden.dtype)
            for p, h in zip(missing, hidden):
                self._text[p] = h
        return torch.stack([self._text[p] for p in prompts])

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], seconds: float, start: float = 0.0) -> Cond:
        text = self.encode_text(prompts)
        n = len(prompts)
        proj = self.projection(
            start_seconds=torch.full((n,), float(start), device=self.device),
            end_seconds=torch.full((n,), float(seconds), device=self.device),
        )
        s, e = proj.seconds_start_hidden_states, proj.seconds_end_hidden_states
        return Cond(torch.cat([text, s, e], dim=1), torch.cat([s, e], dim=2))

    def _rotary(self, length: int) -> tuple[Tensor, Tensor]:
        if length not in self._rot:
            cos, sin = self._rot_fn(self._rot_dim, length, use_real=True, repeat_interleave_real=False)
            self._rot[length] = (cos.to(self.device), sin.to(self.device))
        return self._rot[length]

    def v(self, z: Tensor, t: Tensor, cond: Cond) -> Tensor:
        """Raw transformer call. z is (B, 64, L); t is a scalar or (B,) in (0, 1)."""
        t = torch.as_tensor(t, device=z.device, dtype=torch.float32).expand(z.shape[0])
        with torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp != torch.float32):
            out = self.dit(
                z,
                t,
                encoder_hidden_states=cond.cross,
                global_hidden_states=cond.glob,
                rotary_embedding=self._rotary(z.shape[-1] + 1),
                return_dict=False,
            )[0]
        return out.float()

    def cfg(self, cond: Cond, guidance: float) -> Predictor:
        """Classifier-free guidance against the all-zero cross-attention conditioning."""
        if guidance == 1.0:
            return lambda z, t: self.v(z, t, cond)
        both = Cond.cat([cond.null(), cond])

        def predict(z: Tensor, t: Tensor) -> Tensor:
            vu, vc = self.v(torch.cat([z, z]), t, both).chunk(2)
            return vu + guidance * (vc - vu)

        return predict

    def sample(
        self,
        predict: Predictor,
        noise: Tensor,
        steps: int = 50,
        stop: int | None = None,
        sde: bool = False,
        sde_seeds: Sequence[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """DPM-Solver++(2M) from pure noise.

        Returns (z, t): the latent in the z_t convention and its time. With stop=None
        this is the clean latent at t=0; with stop=k it is the state after k steps,
        which is how training draws partially denoised latents.
        """
        sig = sigma_schedule(steps, noise.device)
        x = noise * sig[0]
        old = None
        last = steps if stop is None else stop
        gens = None
        if sde:
            gens = [torch.Generator().manual_seed(int(s) + 7919) for s in (sde_seeds or range(len(noise)))]
        for i in range(last):
            s, s_next = sig[i], sig[i + 1]
            c_in = 1.0 / torch.sqrt(s * s + 1.0)
            v = predict(x * c_in, sigma_to_t(s))
            x0 = x * c_in * c_in - v * s * c_in
            if s_next == 0:
                x = x0
                break
            h = torch.log(s) - torch.log(s_next)
            if sde:
                # DPM-Solver++(2M) SDE with eta = 1, as in k-diffusion's sample_dpmpp_2m_sde.
                x_new = torch.exp(-h) * (s_next / s) * x + (-torch.expm1(-2 * h)) * x0
                if old is not None:
                    r = h_last / h
                    x_new = x_new + 0.5 * (-torch.expm1(-2 * h)) * (1 / r) * (x0 - old)
                eps = torch.stack([torch.randn(x.shape[1:], generator=g) for g in gens]).to(x)
                x = x_new + eps * s_next * torch.sqrt(-torch.expm1(-2 * h))
            else:
                d = x0
                if old is not None:
                    r = h_last / h
                    d = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * old
                x = (s_next / s) * x - torch.expm1(-h) * d
            old, h_last = x0, h
        s_end = sig[last]
        return x / torch.sqrt(s_end * s_end + 1.0), sigma_to_t(s_end)

    def decode(self, z: Tensor, seconds: float | None = None, chunk: int = 8) -> Tensor:
        """Latents to stereo waveforms in [-1, 1], shape (B, 2, T). Differentiable."""
        outs = [self.vae.decode(z[i : i + chunk].float()).sample for i in range(0, len(z), chunk)]
        audio = torch.cat(outs)
        if seconds is not None:
            audio = audio[..., : int(seconds * self.sample_rate)]
        return audio.clamp(-1, 1)

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[str],
        seeds: Sequence[int],
        seconds: float = 10.0,
        steps: int = 50,
        guidance: float = 7.0,
        wrap: Callable[[Predictor], Predictor] | None = None,
        sde: bool = False,
        latents: bool = False,
    ) -> Tensor:
        """Text to audio. `wrap` lets sliders and baselines modify the guided predictor."""
        cond = self.encode(prompts, seconds)
        predict = self.cfg(cond, guidance)
        if wrap is not None:
            predict = wrap(predict)
        noise = seeded_noise(seeds, (self.channels, self.frames(seconds)), self.device)
        z, _ = self.sample(predict, noise, steps=steps, sde=sde, sde_seeds=seeds)
        return z if latents else self.decode(z, seconds)
