# Axes inside the model

Every other discovery result in this repository looks at the model's output through an embedding (CLAP or MuQ-MuLan). This one looks inside the generator. No text, no embedding model, and no training picks the axes.

**Setup.** ACE-Step 1.5 XL turbo, 48 prompts, 16 seeds each (768 clips). For every clip we store the mean output of each of the 32 cross-attention blocks (width 2,560), averaged over tokens and the 8 sampling steps. Each prompt's mean is removed and the rest is decomposed jointly over blocks with PCA. An axis is one vector per block, scaled to one standard deviation of the clips' natural spread. Steering adds a multiple of it to the block outputs, with the same hook as the activation-steering baseline.

## The spectrum is short

| Axis | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Share of within-prompt variance | 38.0% | 13.7% | 12.4% | 8.5% | 6.1% | 4.5% | 3.6% | 2.8% |
| Block holding most of the axis | 19 | 31 | 17 | 27 | 15 | 21 | 25 | 23 |
| Share of the axis in that block | 45% | 62% | 50% | 59% | 35% | 69% | 28% | 49% |

Eight axes carry 90% of how clips of the same prompt differ in these activations. Fitted on two disjoint halves of the prompts, the first eight match with mean cosine 0.82, so they belong to the model rather than to the prompt list. No axis has weight in the first eight blocks. The decomposition is in raw activation units, so blocks with larger outputs count for more.

## What steering along them does

24 held-out prompts, 2 seeds, positions from -12 to +12 natural standard deviations ([`axes.md`](axes.md)):

| Axis | Tags that rise | Tags that fall | Consistency | Largest measured change (ρ) | Piece kept | Enjoyment at ends (6.98 unsteered) |
|---|---|---|---:|---|---:|---:|
| 1 | complex, improvised, jazz | vocals, hip hop, lo-fi | 0.34 | loudness up (+0.77), flatness down (-0.64) | 0.90 | 6.79 |
| 2 | complex, electric guitar, acoustic | lo-fi, hip hop, pop | 0.38 | loudness up (+0.79), flatness down (-0.64) | 0.93 | 6.84 |
| 3 | rhythmic, techno, bells | dry, slow, major key | 0.17 | loudness up (+0.60) | 0.96 | 6.91 |
| 4 | electronic dance, techno, pop | trumpet, improvised, orchestral | 0.17 | brightness down (-0.35) | 0.97 | 6.95 |
| 5 | melancholic, lo-fi, epic | cello, hand percussion, staccato | 0.16 | brightness up (+0.59) | 0.96 | 6.91 |
| 6 | distorted, pop, dark | acoustic guitar, acoustic, uplifting | 0.12 | width up (+0.42) | 0.97 | 6.93 |
| 7 | dark-toned, reverberant, lo-fi | latin, happy, playful | 0.21 | brightness down (-0.60) | 0.97 | 6.94 |
| 8 | latin, funk, techno | quiet, dark-toned, piano | 0.10 | brightness up (+0.45) | 0.95 | 6.91 |

Consistency is the mean cosine between one trajectory's change in CLAP embedding and the mean change of all the others: 1 if the axis does the same thing to every prompt, 0 if it does something unrelated each time.

## Reading

- **The model's two strongest internal axes are about production.** Both raise level and lower spectral flatness (less noise-like), and both move tags from lo-fi toward complex and live-sounding. On a six-prompt pilot the first axis moved predicted enjoyment from 6.15 at -16 to 7.07 at +16. This agrees with what the embeddings of real recordings showed: the direction along which recordings of one genre differ most is how polished they sound.
- **The rest are weak and inconsistent as controls.** Twelve natural standard deviations leave 95% or more of the piece in place and move descriptors by well under one standard deviation of unsteered clips. Pushed to ±48 on six prompts, axis 4 raises onset rate (1.6 to 3.0 per second) and axes 7 and 8 shift brightness by about an octave, and the clip breaks at one end.
- **Musical axes did not come out of this decomposition.** Arousal, valence, and instrument-family contrasts appear when the same procedure is run on MuQ-MuLan embeddings of real music ([`../real/`](../real/README.md)). Where you look for axes decides what kind you find: the generator's raw activations give acoustic ones.

This is a narrower result than the image literature reports for PCA of U-Net bottleneck activations (Haas et al., 2023), where leading components are semantic. Three differences could explain it and none has been tested here: averaging over tokens removes anything that varies in time, cross-attention outputs are one of several places to look, and an 8-step distilled model may hold less variation per prompt.

## Reproduce

```bash
python experiments/discover_internal.py --backbone ace-turbo --split train --seeds 16 --out runs/discovery/internal_ace
python experiments/run_jobs.py experiments/jobs/15_internal.tsv --gpus 0
python experiments/report_axes.py --eval runs/eval/internal_ace --vocab runs/reference/vocab.npz --out results/discovery/internal
```
