# Stable Audio 3 as a third backbone (pilot)

Stable Audio 3 is a 2026 flow-matching model: z_t = (1 - t) x0 + t eps, the network predicts eps - x0, 44.1 kHz stereo. The base checkpoints (`stabilityai/stable-audio-3-medium-base` and `-small-music-base`) are open; the 8-step post-trained ones are gated. [`audiosliders/sa3.py`](../../audiosliders/sa3.py) wraps the official `stable-audio-3` package behind the same interface as the other two backbones, with the official sampler settings (50 Euler steps, guidance 7 with adaptive projected guidance).

## The wrapper matches the official pipeline

First 8 evaluation prompts, 10-second clips, medium checkpoint.

| | Wrapper | Official pipeline |
|---|---:|---:|
| CLAP prompt similarity, 24 clips | 0.533 ± 0.020 | 0.528 ± 0.019 |
| Content enjoyment, 24 clips | 6.62 | 6.73 |
| Production quality, 24 clips | 7.78 | 7.83 |

Given the official pipeline's own initial noise, the wrapper's latents have cosine 0.979 to 1.000 with the official ones per clip.

## A slider trains and steers

`python -m audiosliders.train brightness --backbone sa3-medium --iters 400`, then a sweep on 8 held-out prompts:

| Position | Spectral centroid (Hz) | CLAP direction | CLAP similarity to unsteered | Content enjoyment |
|---:|---:|---:|---:|---:|
| -2 | 128 | -0.232 | 0.52 | 4.56 |
| -1 | 215 | -0.144 | 0.70 | 5.34 |
| 0 | 608 | -0.047 | 1.00 | 6.69 |
| +1 | 1,451 | 0.125 | 0.78 | 6.60 |
| +2 | 2,272 | 0.244 | 0.52 | 6.14 |

The centroid and the CLAP direction are monotone across all five positions. After 400 iterations the slider reproduces 39% of its guidance target. The dark side also collapses stereo width, the same leak the brightness slider shows on Stable Audio Open.

## Caveats

- Training is slow on the medium checkpoint: 8.5 s per iteration on a GPU shared with another job.
- The official package was run without flash-attn, on its fallback attention.
- Only the prompt-pair trainer and the sweep were exercised; the set trainer, activation steering, and the full evaluation have not been run on this backbone.
- Weights are under the Stability AI Community License. LoRA weights count as derivative works that may be shared for research and non-commercial use with the license and notice attached.

## Six sliders on the small checkpoint

`stabilityai/stable-audio-3-small-music-base`, six prompt-pair sliders at 600 iterations each (about 25 minutes on a shared GPU), 24 held-out prompts, 2 seeds, positions -1.5 to +1.5, no gating. Figures and the full table are in this directory ([`summary.md`](summary.md)).

| Slider | Descriptor | ρ | Ends ordered | Descriptor moved in the usable span (std) | Usable span | Piece kept at ±1 | Enjoyment at -1 / 0 / +1 |
|---|---|---:|---:|---:|---|---:|---|
| brightness | spectral centroid | 0.84 ± 0.08 | 98% | 0.87 | 0 to +1.5 | 0.63 | 4.83 / 6.57 / 6.58 |
| energy | spectral flux | 0.70 ± 0.08 | 98% | 1.53 | -1 to +1 | 0.63 | 6.23 / 6.57 / 6.10 |
| harmony | harmonic change rate | 0.68 ± 0.09 | 98% | 0.80 | -0.5 to +1.5 | 0.74 | 5.74 / 6.57 / 6.92 |
| ensemble | production complexity | 0.67 ± 0.10 | 96% | 0.70 | 0 to +1.5 | 0.69 | 5.38 / 6.57 / 6.96 |
| tempo | beat-tracked tempo | 0.43 ± 0.13 | 72% | 1.24 | -0.5 to +1.5 | 0.72 | 5.77 / 6.57 / 6.79 |
| mood | major-minus-minor fit | 0.34 ± 0.14 | 77% | 0.95 | -1.5 to +1.5 | 0.76 | 6.33 / 6.57 / 6.82 |

- **The same recipe transfers to a third backbone with no change.** Every descriptor follows its slider, and four of six have 96% or more of trajectories ordered. Harmony follows its descriptor better here (0.68) than on ACE-Step (0.41).
- **The cost is quality on the negative side.** Unsteered clips score 6.57, and the dark, sparse, and solo ends drop to 4.8 to 5.4 by position -1, so brightness and ensemble are usable only upward. This is the pattern Stable Audio Open showed before timestep gating ([`../gating`](../gating/)), and these sliders were run without it. The usable spans here are the ungated ones.
- **Less of the piece survives** (0.63 to 0.76 at ±1, against 0.80 to 0.90 on ACE-Step), again as on the other 50-step model before gating.
- The tags move the right way: energy lowers *reverberant, calm, dreamy, minor key*; mood raises *happy* and lowers *dark-toned, mysterious*.

The next step for this backbone is the gated evaluation that fixed Stable Audio Open, and more than 600 iterations.

## With timestep gating

The same six sliders with the slider switched on only after the first 10 or 20 of the 50 Euler steps (`--start 0.99` and `0.95` on this model's sigmoid schedule; [`gate099/`](gate099/summary.md), [`gate095/`](gate095/summary.md)).

| Slider | Steps left unsteered | ρ | Usable span | Descriptor moved in the span (std) | Piece kept | Enjoyment at the ends |
|---|---:|---:|---|---:|---:|---:|
| brightness | 0 | 0.84 | 0 to +1.5 | 0.87 | 0.66 | 5.16 |
| brightness | 20 | 0.48 | -1 to +1.5 | 0.38 | 0.81 | 6.23 |
| energy | 0 | 0.70 | -1 to +1 | 1.53 | 0.63 | 5.70 |
| energy | 20 | 0.72 | -1.5 to +1.5 | 1.13 | 0.69 | 6.39 |
| harmony | 0 | 0.68 | -0.5 to +1.5 | 0.80 | 0.75 | 6.00 |
| harmony | 20 | 0.67 | -1 to +1.5 | 0.69 | 0.80 | 6.33 |
| ensemble | 0 | 0.67 | 0 to +1.5 | 0.70 | 0.63 | 5.87 |
| ensemble | 20 | 0.57 | -0.5 to +1.5 | 0.54 | 0.85 | 6.20 |
| tempo | 0 | 0.43 | -0.5 to +1.5 | 1.24 | 0.78 | 5.82 |
| tempo | 20 | 0.50 | -1 to +1.5 | 1.33 | 0.79 | 6.39 |
| mood | 0 | 0.34 | -1.5 to +1.5 | 0.95 | 0.66 | 6.43 |
| mood | 20 | 0.32 | -1.5 to +1.5 | 0.65 | 0.79 | 6.65 |

The same trade as on Stable Audio Open: leaving the first 20 steps alone keeps more of the piece (0.69 to 0.85 against 0.63 to 0.78), lifts enjoyment at the ends by 0.2 to 1.1 points, and widens every usable span on the negative side, at the cost of range. Brightness pays the most, since its effect is set early; tempo and energy lose almost nothing. Leaving only 10 steps alone buys little.
