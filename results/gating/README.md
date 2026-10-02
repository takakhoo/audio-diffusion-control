# Timestep gating (Stable Audio Open)

The slider is switched on only for sampler times t <= `start`; the earliest, noisiest steps run unsteered. 24 held-out prompts, one seed each, positions -1.5 to 1.5; the table columns compare positions -1 and +1 with position 0. `skipped_steps` is how many of the 50 sampling steps run before the slider turns on.

Read [`summary.csv`](summary.csv). The pattern is the same for all four sliders: as `start` falls, more of the piece survives (CLAP, chroma, and rhythm similarity all rise) and the enjoyment score at ±1 climbs back toward its unsteered value, at the cost of a smaller change in the target descriptor. `start = 0.97` keeps most of the range for brightness (1.41 of 2.37 standard deviations) while cutting the enjoyment loss from 1.0 to 0.3 points, and is the setting used in the main evaluation.

Two sliders fail the descriptor test at every setting. "Density" and "percussion" move the output along their CLAP text direction, but onset rate and percussive energy share do not follow (rank correlation at or below 0.37, range near zero).

Reproduce with `experiments/jobs/02_gating.tsv`.
