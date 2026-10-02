# Second-generation set-trained sliders

Fifteen sliders on ACE-Step 1.5 XL turbo, trained with no text from two sets of the model's own clips. What changed from the first generation:

- **A corpus 5 times larger and 13 times more varied**: 15,552 clips from 648 prompts (24 seeds each), against 3,072 clips from 48 prompts.
- **Sets cut within each prompt at the top and bottom 20%**, after removing from the sorting measurement what a linear fit on loudness, brightness, and predicted enjoyment explains. A slider has no reason to learn those along with its target.
- **Axes from real music.** Seven of the sliders sort the clips along a direction found in 14,985 FMA recordings ([`../discovery/real`](../discovery/real/README.md)).

24 held-out prompts, 3 seeds, positions -1.5 to +1.5. The tables below use -1 to +1, for the reason given at the end.

## Sliders along axes of real music

| Slider | Axis: one end / other end | ρ with the axis | -1 and +1 ordered | Moved, in std of real music | Moved, in std of one prompt's seeds | Piece kept at ±1 | Enjoyment at -1 / 0 / +1 | Musicality at -1 / 0 / +1 |
|---|---|---:|---:|---:|---:|---:|---|---|
| arousal | quiet, minor key, dreamy / aggressive, dry, rhythmic | 0.82 | 99% | 1.19 | 2.14 | 0.84 | 7.09 / 6.95 / 7.16 | 2.72 / 2.77 / 2.68 |
| classical to funk | classical, epic, melodic / funk, drums, happy | 0.64 | 96% | 0.77 | 1.65 | 0.85 | 7.16 / 6.95 / 7.09 | 2.72 / 2.77 / 2.67 |
| piano | piano, melancholic, electric piano / organ, rock, folk | 0.63 | 96% | 0.75 | 1.42 | 0.87 | 7.10 / 6.95 / 7.06 | 2.66 / 2.77 / 2.72 |
| jazz to electronic | jazz, saxophone, trumpet / electronic dance, synthesizer | 0.61 | 94% | 0.61 | 1.23 | 0.87 | 7.08 / 6.95 / 7.09 | 2.77 / 2.77 / 2.63 |
| acoustic to electronic | acoustic, sad, blues / studio, playful, electronic dance | 0.55 | 93% | 0.64 | 1.48 | 0.86 | 7.08 / 6.95 / 7.14 | 2.68 / 2.77 / 2.69 |
| valence | dark, metal, dark-toned / playful, pop, simple | 0.50 | 93% | 0.62 | 1.23 | 0.87 | 7.10 / 6.95 / 7.05 | 2.78 / 2.77 / 2.67 |
| strings to synth | acoustic guitar, electric guitar, strings / reverberant, choir, synthesizer | 0.40 | 82% | 0.44 | 0.81 | 0.87 | 7.15 / 6.95 / 7.03 | 2.68 / 2.77 / 2.69 |

ρ is the rank correlation between slider position and the projection of the output on the axis, in the embedding the axis was found in: CLAP for classical-funk and acoustic-electronic, MuQ-MuLan for the other five. "Moved" is the change in that projection between -1 and +1. The full table with tags and descriptors is [`real_axes.md`](real_axes.md).

- All seven follow their axis, and on six of them at least 93% of trajectories have the two ends in the right order.
- The arousal slider moves a clip 1.19 standard deviations of real music along the arousal axis. The model's whole output over 648 prompts has a standard deviation of 0.83 on that axis ([`../coverage`](../coverage/README.md)), so one slider on one prompt and seed crosses more of the axis than changing the prompt typically does.
- The piece survives (CLAP similarity 0.84 to 0.87 at ±1), enjoyment is at or above the unsteered level at both ends, and SongEval musicality drops by 0.1 at most (2.63 to 2.78 against 2.77).

## Sliders along a measurement

| Slider | Sorted by | ρ with the measurement | -1 and +1 ordered | Moved (std of unsteered clips) | Selectivity | Piece kept at ±1 | Enjoyment at ±1 |
|---|---|---:|---:|---:|---:|---:|---:|
| energy | spectral flux | 0.83 | 99% | 1.06 | 3.8 | 0.85 | 7.09 |
| harmony | harmonic change rate | 0.72 | 99% | 1.25 | 6.6 | 0.88 | 7.09 |
| density | onset rate | 0.63 | 90% | 0.70 | 4.7 | 0.87 | 7.07 |
| ensemble | production complexity | 0.60 | 94% | 0.66 | 2.4 | 0.86 | 7.12 |
| quality | content enjoyment | 0.53 | 86% | 0.62 | 3.7 | 0.86 | 7.07 |
| tempo | beat-tracked tempo | 0.06 | 38% | 0.11 | 0.8 | 0.87 | 7.04 |
| mood | "happy" minus "sad" tag score | -0.03 | 46% | -0.10 | -0.2 | 0.85 | 7.15 |

Selectivity is the slope of the slider's own measurement divided by the mean absolute slope of the other descriptors in the leakage panel, all in standard deviations of unsteered clips. Full table: [`summary.md`](summary.md).

The same attributes trained from a prompt pair, measured over the same positions (-1 to +1):

| Attribute | Trained from | ρ with the measurement | -1 and +1 ordered | Moved (std) | Selectivity | Piece kept at ±1 | ρ with the MuQ text direction |
|---|---|---:|---:|---:|---:|---:|---:|
| harmony | prompt pair | 0.27 | 65% | 0.45 | 0.8 | 0.91 | 0.63 |
| harmony | two sets | 0.72 | 99% | 1.25 | 6.6 | 0.88 | -0.07 |
| density | prompt pair | 0.74 | 90% | 0.88 | 1.5 | 0.88 | 0.73 |
| density | two sets | 0.63 | 90% | 0.70 | 4.7 | 0.87 | 0.24 |
| ensemble | prompt pair | 0.50 | 82% | 0.64 | 1.5 | 0.89 | 0.65 |
| ensemble | two sets | 0.60 | 94% | 0.66 | 2.4 | 0.86 | 0.03 |

- **The set-trained sliders are the selective ones.** They move their own measurement 2.4 to 6.6 times as far as an average bystander, against 0.8 to 1.5 for the prompt-pair sliders. Balancing the two sets on loudness, brightness, and enjoyment is what does it.
- **Harmony is much better this way.** ρ 0.72 with 99% of trajectories ordered, against 0.27 and 65% for the prompt-pair slider over the same positions.
- **Each trainer moves what it was trained on.** The set-trained harmony slider moves harmonic change rate (0.72) and leaves the MuQ-MuLan score for "rich harmony" against "simple harmony" where it was (-0.07). The prompt-pair slider does the reverse: 0.63 in MuQ, 0.27 on the waveform. Neither kind of score can stand in for the other.
- **Two failures.** Tempo does not move. Mood sorted by a "happy" minus "sad" tag score moves the MuQ text direction a little (ρ 0.39) and the major-minor fit not at all; the first-generation mood slider, sorted by the major-minor fit itself, reached ρ 0.43.

## Why ±1

| Position | -1.5 | -1 | -0.5 | 0 | +0.5 | +1 | +1.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Enjoyment (Audiobox), harmony slider | 6.80 | 7.04 | 6.95 | 6.95 | 6.98 | 7.13 | 6.89 |
| Musicality (SongEval), harmony slider | 2.33 | 2.66 | 2.73 | 2.77 | 2.76 | 2.71 | 2.37 |
| Piece kept, harmony slider | 0.72 | 0.89 | 0.96 | 1.00 | 0.95 | 0.87 | 0.67 |

Past ±1 every set-trained slider loses the piece and SongEval musicality falls by 0.3 to 0.5 in both directions, while Audiobox enjoyment stays flat. The second predictor catches a failure the first one misses. These sliders are trained at exactly ±1 and do not extrapolate, unlike the prompt-pair sliders, which are usable to ±2.

## Reproduce

```bash
python experiments/run_jobs.py experiments/jobs/11_ace_large_corpus.tsv --gpus 0 1 2 3
python experiments/run_jobs.py experiments/jobs/12_v2_sliders.tsv --gpus 0 1 2
venv-muq/bin/python experiments/muq_score.py runs/eval/ace_v2/contrast_*
venv-muq/bin/python experiments/songeval_score.py --songeval ../SongEval runs/eval/ace_v2/contrast_*
python experiments/report.py --eval runs/eval/ace_v2 --out results/ace_v2 --real runs/reference/fma --vocab runs/reference/vocab.npz --max-scale 1
python experiments/report_real_axes.py --eval runs/eval/ace_v2 --coverage results/coverage --vocab runs/reference/vocab.npz --out results/ace_v2
```
