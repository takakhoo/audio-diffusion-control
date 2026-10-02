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
