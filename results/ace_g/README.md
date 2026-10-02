# Graded set training: the text-free slider past ±1

The set trainer puts the top and bottom 20% of clips at +1 and -1 and never shows the slider any other position. Its sliders work inside ±1 and fall off a cliff beyond it ([`../ace_v2`](../ace_v2/README.md): similarity to the unsteered clip drops to 0.64 to 0.72 and SongEval musicality by 0.3 to 0.5 at ±1.5).

`--graded` changes one thing: every clip is trained at its own position, its standing inside its prompt scaled so that the two 20% tails average +1 and -1 and capped at ±2.5, with clips far from zero drawn more often. Six sliders were retrained this way on the same corpus and settings and evaluated from -2 to +2 (24 held-out prompts, 3 seeds). Tables and figures: [`summary.md`](summary.md), [`real_axes.md`](real_axes.md), [`quality_check.md`](quality_check.md).

## The cliff is gone

Similarity to the unsteered clip and SongEval musicality by position, plain against graded:

| Slider | Trainer | -2 | -1.5 | -1 | 0 | +1 | +1.5 | +2 |
|---|---|---|---|---|---|---|---|---|
| energy | plain | | 0.69 / 2.41 | 0.86 / 2.66 | 1.00 / 2.77 | 0.84 / 2.71 | 0.64 / 2.26 | |
| energy | graded | 0.83 / 2.69 | 0.85 / 2.72 | 0.86 / 2.71 | 1.00 / 2.77 | 0.85 / 2.70 | 0.82 / 2.76 | 0.78 / 2.82 |
| harmony | plain | | 0.72 / 2.33 | 0.89 / 2.66 | 1.00 / 2.77 | 0.87 / 2.71 | 0.67 / 2.37 | |
| harmony | graded | 0.86 / 2.73 | 0.88 / 2.71 | 0.89 / 2.69 | 1.00 / 2.77 | 0.85 / 2.72 | 0.85 / 2.70 | 0.84 / 2.74 |
| arousal | plain | | 0.71 / 2.43 | 0.87 / 2.72 | 1.00 / 2.77 | 0.81 / 2.68 | 0.67 / 2.43 | |
| arousal | graded | 0.85 / 2.79 | 0.86 / 2.76 | 0.87 / 2.68 | 1.00 / 2.77 | 0.81 / 2.69 | 0.81 / 2.68 | 0.79 / 2.69 |

Every graded slider is usable from -2 to +2 under both quality predictors; enjoyment is at or above the unsteered 6.95 at every position.

## What it moves

| Slider | Trainer | ρ, -1 to +1 | ρ, whole range | Ends ordered | Moved between -1 and +1 | Moved between the ends |
|---|---|---:|---:|---:|---:|---:|
| energy (spectral flux, std) | plain | 0.83 | 0.73 (±1.5) | 99% | 1.06 | 1.31 (±1.5) |
| energy | graded | 0.67 | 0.89 (±2) | 100% | 0.74 | 1.75 (±2) |
| harmony (harmonic change, std) | plain | 0.72 | 0.60 (±1.5) | 99% | 1.25 | 0.77 (±1.5) |
| harmony | graded | 0.67 | 0.60 (±2) | 92% | 1.11 | 1.02 (±2) |
| arousal (real-music axis, real std) | plain | 0.82 | 0.68 (±1.5) | 99% | 1.19 | 1.11 (±1.5) |
| arousal | graded | 0.78 | 0.80 (±2) | 96% | 1.00 | 1.46 (±2) |
| jazz to electronic (real std) | plain | 0.61 | 0.47 (±1.5) | 94% | 0.61 | 0.33 (±1.5) |
| jazz to electronic | graded | 0.63 | 0.56 (±2) | 93% | 0.61 | 0.65 (±2) |
| piano (real std) | plain | 0.63 | 0.57 (±1.5) | 96% | 0.75 | 0.94 (±1.5) |
| piano | graded | 0.58 | 0.60 (±2) | 89% | 0.62 | 0.84 (±2) |
| valence (real std) | plain | 0.50 | 0.45 (±1.5) | 93% | 0.62 | 0.45 (±1.5) |
| valence | graded | 0.49 | 0.56 (±2) | 82% | 0.49 | 0.61 (±2) |

- **Over the whole range the graded sliders are the more monotone** (0.56 to 0.89 against 0.45 to 0.73), because the plain ones reverse past ±1.
- **Between -1 and +1 they are a little weaker** (energy moves 0.74 against 1.06 standard deviations and its rank correlation there is 0.67 against 0.83), and they take a larger first step: similarity at ±0.5 is 0.85 to 0.90 against 0.92 to 0.96.
- **The reach doubles.** Energy moves 1.75 standard deviations between -2 and +2 and arousal 1.46 standard deviations of real music, where the plain versions had stopped being usable.
- Harmony's negative side flattens past -1 for both trainers: harmonic change rate has a floor in this model's output that the sets cannot push below.

Usable from -2 to +2 with no loss on either quality predictor, these are the versions to use when range matters; the plain ones are slightly stronger inside ±1.
