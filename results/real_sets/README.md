# Training directly between sets of real recordings: a negative result

The set trainer works on the model's own clips. We tried it on real ones: 24,975 FMA recordings encoded into ACE-Step's latent space, sorted by beat-tracked tempo or by a "happy" minus "sad" tag score, with the top and bottom 20% as the two sets (vocal-heavy tracks removed, sets balanced within genre). 1,500 iterations, 12 held-out prompts, one seed. The plain run was made when three quarters of the corpus had been encoded (about 18,700 recordings).

| Slider | Trainer | What the target measurement did across the range | Enjoyment at -1 / 0 / +1 | CLAP similarity to the unsteered clip at ±1 |
|---|---|---|---|---|
| tempo | plain | no trend (116 / 134 / 119 BPM at -0.5 / 0 / +0.5); beyond ±1 both directions turn to noise | 6.2 / 7.0 / 5.5 | 0.46 and 0.36 |
| tempo | with symmetry penalty | no trend (111 / 128 / 113 BPM at -1 / 0 / +1) | 7.1 / 7.0 / 7.0 | 0.90 and 0.93 |
| mood | with symmetry penalty | major/minor fit 0.09 / 0.10 / 0.15 at -1 / 0 / +1; weak | 6.9 / 7.0 / 7.1 | 0.95 and 0.79 |

Two things are going on.

- **The plain trainer damages the clip equally in both directions.** An effect that is the same at +s and -s cannot be the attribute; it is the size of a noisy weight update showing through. Adding a penalty that makes the slider's predictions at +1 and -1 average to the frozen model's removes it: enjoyment and similarity stay at their unsteered values across ±1.
- **With the damage gone, almost no steering is left.** Real recordings sit off the model's own output distribution and their captions are weak, so the denoising loss on them is dominated by that mismatch. The mismatch is common to both sets and cancels, as designed, and the difference between the sets that remains is too small to learn at this budget.

What works instead is to keep real music for what it is good at, finding and naming the axes ([`results/discovery/real/`](../discovery/real/)), and to train the slider on the model's own clips sorted along those axes.

On the model's own clips the symmetry penalty trades effect for preservation: for a discovered axis it halved the movement along the axis at ±1 (0.26 against 0.46) and raised similarity to the unsteered clip (0.91 against 0.84). Neither version stays usable beyond ±1.
