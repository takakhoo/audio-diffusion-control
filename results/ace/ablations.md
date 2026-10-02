# Training choices and composition (ACE-Step, mood slider)

## What changes when the training recipe changes

The mood slider retrained with one setting changed, 24 held-out prompts, 2 seeds (3 for the default), positions -2 to +2. Descriptor: major-minus-minor key fit.

| Recipe | ρ with the descriptor | Ends ordered | ρ with the CLAP direction | Usable span | Piece kept at ±1 | Enjoyment at -2 / 0 / +2 | Brightness leak (std per unit) |
|---|---:|---:|---:|---|---:|---|---:|
| default: rank 4, 1,000 iterations, all attention and feed-forward layers, η = 2 | 0.39 ± 0.10 | 75% | 0.92 | -1 to +2 | 0.85 | 6.17 / 6.95 / 7.13 | +0.81 |
| rank 16 | 0.30 ± 0.14 | 67% | 0.92 | -1 to +2 | 0.84 | 5.93 / 6.96 / 7.15 | +0.73 |
| 3,000 iterations | 0.52 ± 0.12 | 83% | 0.92 | -1 to +2 | 0.83 | 5.90 / 6.96 / 7.27 | +0.84 |
| cross-attention layers only | 0.29 ± 0.14 | 54% | 0.93 | -1 to +2 | 0.84 | 6.23 / 6.96 / 7.19 | +0.80 |
| η = 4 | 0.32 ± 0.14 | 69% | 0.92 | -0.5 to +2 | 0.80 | 5.70 / 6.96 / 7.05 | +1.01 |

- The CLAP direction score is the same for every recipe (0.92 to 0.93). It cannot tell them apart. The waveform descriptor can.
- A higher rank buys nothing. A larger guidance multiplier costs range, similarity, and enjoyment, and leaks more brightness.
- Restricting the update to cross-attention keeps the embedding score and loses a quarter of the descriptor correlation; only 54% of trajectories end up ordered.
- Three times the training is the one change that helps the descriptor (0.52 against 0.39), with overlapping intervals.

## Two sliders at once

Both sliders on a 3 by 3 grid of positions (-1, 0, +1), 24 prompts. A descriptor's change is regressed on its own slider's position, the other slider's position, and their product, in standard deviations of unsteered clips ([`experiments/compose.py`](../../experiments/compose.py)).

| Pair | Descriptor | Slope on its own slider | Slope on the other slider | Interaction |
|---|---|---:|---:|---:|
| mood + brightness | major-minor fit (mood) | 0.39 | 0.21 | 0.09 |
| mood + brightness | spectral centroid (brightness) | 1.91 | 1.12 | 0.44 |
| harmony + groove | harmonic change rate (harmony) | 0.31 | 0.14 | 0.06 |
| ensemble + tension | production complexity (ensemble) | 0.27 | 0.42 | -0.09 |

The interaction terms are a fifth to a third of the main effects, so two sliders applied together roughly add. The off-diagonal slopes are the leaks already seen one slider at a time: mood brightens the clip, and tension raises production complexity more than the ensemble slider does.
