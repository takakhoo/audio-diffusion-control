"""ACE-Step 1.5 XL behind the same interface as StableAudio.

A rectified-flow model: x_t = (1 - t) x0 + t eps and the transformer predicts
v = eps - x0, with t = 1 pure noise. Latents are (frames, 64) at 25 Hz and decode to
48 kHz stereo. `turbo` is the 8-step distilled model and runs without guidance; `base`
uses the pipeline's norm-limited classifier-free guidance.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
from torch import Tensor

from .backbone import Cond, Predictor, StableAudio, seeded_noise

REPO = "ACE-Step/acestep-v15-xl-{}-diffusers"


class AceStep:
    def __init__(self, variant: str = "turbo", device: str = "cuda", shift: float = 3.0, text_tokens: int = 128):
        from diffusers import AceStepPipeline

        self.name = f"ace-{variant}"
        self.steps, self.guidance = (8, 1.0) if variant == "turbo" else (30, 7.0)
        self.shift = shift
        self.device = torch.device(device)
        self.pipe = AceStepPipeline.from_pretrained(REPO.format(variant), torch_dtype=torch.bfloat16).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.dit = self.pipe.transformer.eval().requires_grad_(False)
        self.vae = self.pipe.vae.eval().requires_grad_(False)
        self.sample_rate = int(self.pipe.sample_rate)
        self.rate = float(self.pipe.latents_per_second)
        self.channels = int(self.dit.config.audio_acoustic_hidden_dim)
        self.amp = torch.bfloat16
        self._instruction = self.pipe._get_task_instruction(
            task_type="text2music", track_name=None, complete_track_classes=None
        )
        self._cache: dict[tuple[str, float], Tensor] = {}
        self.text_tokens = text_tokens
        self._filler = "music " * 400
        self.fit_peak = StableAudio.fit_peak

    def frames(self, seconds: float) -> int:
        return math.ceil(seconds * self.rate)

    def latent_shape(self, seconds: float) -> tuple[int, int]:
        return (self.frames(seconds), self.channels)

    @staticmethod
    def diffuse(x0: Tensor, eps: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        t = t.view(-1, 1, 1)
        return (1 - t) * x0 + t * eps, eps - x0

    @torch.no_grad()
    def _encode_one(self, prompt: str, seconds: float) -> Tensor:
        """Conditioning tokens for one prompt, padded to a fixed length.

        The pipeline pads a batch to its longest prompt and the transformer attends to the
        padding. Encoding every prompt next to one over-long filler reproduces that with a
        constant length, so a prompt's tokens do not depend on what else is in the batch.
        """
        key = (prompt, seconds)
        if key not in self._cache:
            pipe = self.pipe
            text, text_mask, lyric, lyric_mask = pipe.encode_prompt(
                prompt=[prompt, self._filler], lyrics=["[instrumental]"] * 2, device=self.device,
                vocal_language="en", audio_duration=seconds, instruction=self._instruction, bpm=None,
                keyscale=None, timesignature=None, max_text_length=self.text_tokens, max_lyric_length=2048,
            )
            timbre = pipe.condition_encoder.silence_latent[:, : math.ceil(30 * self.rate), :]
            timbre = timbre.to(self.device, torch.bfloat16).expand(2, -1, -1).contiguous()
            hidden, _ = pipe.condition_encoder(
                text_hidden_states=text, text_attention_mask=text_mask, lyric_hidden_states=lyric,
                lyric_attention_mask=lyric_mask, refer_audio_acoustic_hidden_states_packed=timbre,
                refer_audio_order_mask=torch.arange(2, device=self.device),
            )
            self._cache[key] = hidden[0]
        return self._cache[key]

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], seconds: float) -> Cond:
        cross = torch.stack([self._encode_one(p, seconds) for p in prompts])
        n, length = len(prompts), self.frames(seconds)
        src, _ = self.pipe.prepare_src_latents(device=self.device, dtype=torch.bfloat16, batch_size=n,
                                               latent_length=length)
        context = torch.cat([src, torch.ones_like(src)], dim=-1)
        empty = getattr(self.pipe.condition_encoder, "null_condition_emb", None)
        return Cond(cross, context, None if empty is None else empty.to(cross))

    def v(self, z: Tensor, t: Tensor, cond: Cond) -> Tensor:
        t = torch.as_tensor(t, device=z.device, dtype=torch.float32).expand(z.shape[0]).to(torch.bfloat16)
        with torch.autocast(self.device.type, dtype=self.amp):
            out = self.dit(hidden_states=z.to(torch.bfloat16), timestep=t, timestep_r=t,
                           encoder_hidden_states=cond.cross, context_latents=cond.glob, return_dict=False)[0]
        return out.float()

    def cfg(self, cond: Cond, guidance: float) -> Predictor:
        if guidance == 1.0:
            return lambda z, t: self.v(z, t, cond)
        from diffusers.pipelines.ace_step.pipeline_ace_step import MomentumBuffer, normalized_guidance

        both = Cond.cat([cond, cond.null()])
        momentum = MomentumBuffer(momentum=-0.75)

        def predict(z: Tensor, t: Tensor) -> Tensor:
            vc, vu = self.v(torch.cat([z, z]), t, both).chunk(2)
            return normalized_guidance(pred_cond=vc, pred_uncond=vu, guidance_scale=guidance - 1.0,
                                       momentum_buffer=momentum, eta=0.0, norm_threshold=2.5,
                                       use_original_formulation=True, norm_dim=(1,))

        return predict

    def schedule(self, steps: int) -> Tensor:
        t = torch.linspace(1.0, 0.0, steps + 1, device=self.device)
        return self.shift * t / (1 + (self.shift - 1) * t)

    def sample(self, predict: Predictor, noise: Tensor, steps: int | None = None, stop: int | None = None,
               **_) -> tuple[Tensor, Tensor]:
        """Euler steps from noise. With stop=k, returns the state after k steps and its time."""
        ts = self.schedule(steps or self.steps)
        x = noise
        last = len(ts) - 1 if stop is None else stop
        for i in range(last):
            x = x + (ts[i + 1] - ts[i]) * predict(x, ts[i])
        return x, ts[last]

    def decode(self, z: Tensor, seconds: float | None = None) -> Tensor:
        chunk = max(1, 2048 // z.shape[1])
        outs = [self.vae.decode(z[i : i + chunk].transpose(1, 2).to(self.vae.dtype)).sample.float()
                for i in range(0, len(z), chunk)]
        audio = torch.cat(outs)
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
