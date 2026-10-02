# Audio Sliders

[![tests](https://github.com/takakhoo/audio-diffusion-control/actions/workflows/tests.yml/badge.svg)](https://github.com/takakhoo/audio-diffusion-control/actions/workflows/tests.yml)

**Sliders for generated music that are measured, and that keep it sounding like music.** A slider is a small LoRA on a frozen text-to-music model. Drag it and the same piece, same prompt and same seed, moves along one axis: sad to happy, solo to full ensemble, stiff to groovy, plain harmony to rich harmony, dark to bright. The model is untouched and the slider adds no extra sampling passes.

![Four sliders sweeping across their range on ACE-Step, with the spectrogram, enjoyment score, and similarity to the original at each position](results/demo/sliders.gif)

*Four real sliders on four held-out prompts. Each panel is one prompt and one seed; only the slider moves. The knob sweeps from the middle to +2, back to -2, and home, while the spectrogram redraws and the readout shows the Audiobox enjoyment score and how close the clip stays to the unsteered one.*

**[Listen and drag the sliders yourself: live demo](https://takakhoo.github.io/audio-diffusion-control/)** (18 sliders, 5 prompts, 920 loudness-matched clips, with the measured numbers at every position)

**[Take the ten-minute listening test](https://takakhoo.github.io/audio-diffusion-control/listen.html)**: 23 questions on the same clips, nothing uploaded, and it shows how your ears line up with the measurements at the end.

> **Status (2 Oct 2026).** Everything below is measured and reproducible from this repository. Still running: the text sliders retrained on 648 prompts, graded set training, Stable Audio 3, and a replication of the real-music axes on 81,600 more recordings.

## Headline

Eight sliders on ACE-Step 1.5 XL turbo, each trained from one prompt pair in about 20 minutes. 24 held-out prompts, 3 seeds, 9 slider positions, 648 clips per slider.

| Slider | What rises toward + (CLAP tags) | What falls | Measured descriptor follows? (ρ, ends ordered) | Second model agrees? (MuQ ρ, ordered) | Usable span | Piece kept | Enjoyment at the ends |
|---|---|---|---|---|---|---:|---:|
| [**mood**](https://takakhoo.github.io/audio-diffusion-control/?model=ace&method=lora&slider=mood&prompt=0&x=2) (sad to happy) | happy, latin, reggae | distorted, dark-toned, lo-fi | major/minor fit: 0.39, 75% | 0.88, 100% | -1 to +2 | 0.79 | 6.65 |
| [**ensemble**](https://takakhoo.github.io/audio-diffusion-control/?model=ace&method=lora&slider=ensemble&prompt=0&x=2) (solo to full) | orchestral film score, epic, choir | straight, funk, hip hop | production complexity: 0.60, 90% | 0.77, 94% | -1.5 to +2 | 0.80 | 6.56 |
| [**groove**](https://takakhoo.github.io/audio-diffusion-control/?model=ace&method=lora&slider=groove&prompt=0&x=2) (stiff to groovy) | latin, funk, reggae | choir, strings, cello | no descriptor assigned | 0.56, 96% | -1.5 to +2 | 0.79 | 6.88 |
| [**harmony**](https://takakhoo.github.io/audio-diffusion-control/?model=ace&method=lora&slider=harmony&prompt=0&x=2) (plain to rich) | minor key, melancholic, major key | country, vocals, violin | harmonic change rate: 0.41, 76% | 0.73, 96% | -1.5 to +2 | 0.83 | 6.64 |
| [**melody**](https://takakhoo.github.io/audio-diffusion-control/?model=ace&method=lora&slider=melody&prompt=0&x=2) (texture to tune) | romantic, happy, blues | lo-fi, mysterious, calm | key clarity: 0.48, 78% | 0.89, 100% | -1 to +2 | 0.80 | 6.47 |
| [**tension**](https://takakhoo.github.io/audio-diffusion-control/?model=ace&method=lora&slider=tension&prompt=0&x=1) (relaxed to tense) | metal, rock, distorted | melancholic, uplifting, sad | no descriptor assigned | 0.86, 100% | -1.5 to +1 | 0.83 | 5.64 |
| [**brightness**](https://takakhoo.github.io/audio-diffusion-control/?model=ace&method=lora&slider=brightness&prompt=0&x=2) (dark to bright) | happy, repetitive, romantic | dark-toned, lo-fi, quiet | spectral centroid: 0.97, 100% | 0.80, 100% | -1 to +2 | 0.74 | 6.04 |
| [**density**](https://takakhoo.github.io/audio-diffusion-control/?model=ace&method=lora&slider=density&prompt=0&x=2) (sparse to dense) | latin, repetitive, romantic | dark-toned, quiet, lo-fi | onset rate: 0.78, 92% | 0.86, 100% | -0.5 to +2 | 0.84 | 6.09 |

**Axes nobody named.** The second set of sliders runs along directions found in 14,985 real recordings: [arousal](https://takakhoo.github.io/audio-diffusion-control/?model=ace&slider=arousal&method=lora&prompt=0&x=-2), [valence](https://takakhoo.github.io/audio-diffusion-control/?model=ace&slider=valence&method=contrast&prompt=0&x=1), [electronic to jazz](https://takakhoo.github.io/audio-diffusion-control/?model=ace&slider=jazz_electronic&method=lora&prompt=1&x=2), [synth to strings](https://takakhoo.github.io/audio-diffusion-control/?model=ace&slider=strings_synth&method=lora&prompt=1&x=2), [piano](https://takakhoo.github.io/audio-diffusion-control/?model=ace&slider=piano_axis&method=lora&prompt=2&x=1). They follow those axes on held-out prompts with rank correlation up to 0.91, and the arousal slider moves a clip 2.4 standard deviations of real music along its axis before quality drops ([section 9](#9-axes-of-real-music)).

Each slider name opens the live demo on that slider. The tag columns and the direction score come from CLAP; the "second model" column repeats the direction test with MuQ-MuLan, a music-text model that shares nothing with CLAP, and it agrees on all eight. For scale: unsteered clips score 6.95 on Audiobox Aesthetics content enjoyment, and 2,000 real recordings from FMA average 6.1. "Usable span" is how far the slider goes before mean enjoyment falls more than 0.5 below the unsteered clips. "Piece kept" is CLAP similarity to the unsteered clip at the ends of that span. Full tables: [`results/ace/`](results/ace/).

**Target venue: ISMIR 2027** (London, September 2027; six pages, double-blind). The draft in [`paper/`](paper/) is already on the official ISMIR template and within its limits. Why this venue, its rules, and what the paper still needs to be competitive there are in [`docs/VENUE.md`](docs/VENUE.md).

## Contents

- [How it works](#how-it-works)
- [The science, step by step](#the-science-step-by-step)
- [Try it](#try-it)
- [What happened to v0](#what-happened-to-v0)
- [Layout](#layout)

## How it works

![Where the slider sits inside the transformer](results/figures/architecture.png)

1. **One mechanism.** Every linear layer in the transformer's attention and feed-forward blocks gets a rank-4 update whose strength is a number read at each forward pass. Zero is the original model; negative values work as well as positive ones; several sliders add ([`lora.py`](audiosliders/lora.py)).
2. **Trained from a prompt pair.** The slider at ±1 learns to reproduce the frozen model's prediction shifted by the difference between its predictions for "prompt, happy" and "prompt, sad". This is the Concept Sliders objective, here for a v-prediction diffusion model and a rectified-flow model ([`train.py`](audiosliders/train.py)).
3. **Or trained from two sets of clips, with no text.** Generate a corpus with the model, measure something on every clip, and train the slider with the plain denoising loss at +1 on the top 20 to 30% and at -1 on the bottom 20 to 30%. One update serves both ends with opposite sign, so what the sets share cancels. The measurement can be a signal descriptor, a quality score, or the projection on a discovered direction ([`contrast.py`](audiosliders/contrast.py)).
![How a slider is used and the two ways to train one](results/figures/pipeline.png)

4. **Two backbones, one interface.** ACE-Step 1.5 XL turbo (48 kHz, 8 steps, 0.4 s per 10 s clip) and Stable Audio Open 1.0 (44.1 kHz, 50 steps, 0.65 s per clip) ([`ace.py`](audiosliders/ace.py), [`backbone.py`](audiosliders/backbone.py)).

## The science, step by step

### 1. A slider has to pass three tests

Published audio sliders are scored by text-audio similarity in a learned embedding. That says the output drifted toward a word. It does not say the audio changed in the intended way, that it is still the same piece, or that it is still music. So every slider here is tested on:

- **A measured property of the waveform**, chosen before training: spectral centroid, onset rate, harmonic change rate, key clarity, major/minor fit, pulse clarity, level dynamics, and more ([`descriptors.py`](audiosliders/descriptors.py)). No learned model is involved.
- **Whether it still sounds like music**: Audiobox Aesthetics on every clip, anchored by 2,000 real recordings ([`quality.py`](audiosliders/quality.py)).
- **Whether it is still the same piece**: CLAP, chroma, and onset-envelope similarity to the same seed at position zero.

A fourth readout says what changed in words: 80 instrument, genre, mood, and character tags scored in CLAP space ([`tags.py`](audiosliders/tags.py)).

### 2. The measured descriptor follows the slider

![Descriptor against slider position, ACE-Step](results/ace/response.png)

Each curve is the mean change from the unsteered clip over 72 trajectories, in units of the descriptor's spread across unsteered clips. Brightness moves the spectral centroid by more than four standard deviations with every one of its 72 trajectories ordered correctly. Harmony moves the harmonic change rate by about half a standard deviation in each direction, and density, ensemble, melody, and mood all rise with their slider.

### 3. Pushing too far stops sounding like music, and that is measurable

![Enjoyment scores of unsteered model output and real recordings](results/figures/quality_reference.png)

First the yardstick. Audiobox Aesthetics scores 2,000 real recordings from FMA at 6.12 on average. Unsteered ACE-Step output scores 6.92 with a much tighter spread, and unsteered Stable Audio Open output scores 6.16, about the same as the real recordings. Then the same score at every slider position:

![Content enjoyment against slider position, ACE-Step](results/ace/quality.png)

The dashed line is the mean of real recordings. Seven of the eight sliders stay at or above it across their usable span. Tension collapses past +1, which is why its usable span ends there. The negative ends (sadder, sparser, plainer) cost a little enjoyment; the positive ends cost almost none.

**A second predictor agrees.** Audiobox is one learned model, so every clip was also scored with SongEval, a head on MuQ features trained on human ratings of generated songs (musicality, 1 to 5). It shares no data or weights with Audiobox. Unsteered ACE-Step clips score 2.77 and the 2,000 real recordings 2.78. The usable span it implies is the same for six of the eight sliders and half a step wider for the other two, and over individual clips the two predictors correlate at 0.52 to 0.67. Where they disagree is on the first-generation set-trained sliders, which Audiobox rates as flat out to ±1.5 and SongEval marks down past ±1. All methods side by side: [`results/ace/quality_check.md`](results/ace/quality_check.md).

### 4. Leaving the first steps alone keeps the piece

On Stable Audio Open the slider can be switched on only after the earliest, noisiest sampling steps, the ones that decide the layout of the piece. For the brightness slider at positions ±1 on 24 held-out prompts:

| Slider switched on after | Centroid moved (std) | CLAP similarity to original | Chroma similarity | Enjoyment (6.16 unsteered) |
|---|---:|---:|---:|---:|
| step 0 of 50 | 2.37 | 0.67 | 0.68 | 5.15 |
| step 14 | 1.98 | 0.74 | 0.74 | 5.45 |
| **step 21** | **1.41** | **0.85** | **0.84** | **5.87** |
| step 29 | 0.42 | 0.97 | 0.97 | 6.08 |

![Range, similarity, and enjoyment against the step at which the slider turns on](results/figures/gating.png)

Starting at step 21 keeps most of the effect and cuts the loss in enjoyment from 1.0 to 0.3 points. All four tested sliders show the same pattern ([`results/gating/`](results/gating/)).

### 5. The embedding can say yes while the waveform says no

All 20 prompt-pair sliders were trained on Stable Audio Open and evaluated with the gate on (24 held-out prompts, 3 seeds, 7 positions). Every one of them moves the output along its CLAP text direction. Scored against the waveform, they split:

| Follows its descriptor | ρ | Ends ordered | | Does not | ρ | Ends ordered |
|---|---:|---:|---|---|---:|---:|
| width (side/mid energy) | 0.95 ± 0.01 | 100% | | mood (major/minor fit) | -0.04 ± 0.13 | 43% |
| bass (energy below 150 Hz) | 0.92 ± 0.05 | 97% | | harmony (harmonic change) | 0.19 ± 0.14 | 64% |
| brightness (centroid) | 0.84 ± 0.04 | 96% | | density (onset rate) | 0.30 ± 0.16, no net change | 75% |
| energy (spectral flux) | 0.71 ± 0.08 | 94% | | percussion (percussive share) | 0.34 ± 0.18, no net change | 71% |
| ensemble (production complexity) | 0.70 ± 0.09 | 93% | | | | |
| tempo (beat tracker) | 0.57 ± 0.09 | 78% | | | | |
| distortion (spectral flatness) | 0.50 ± 0.12 | 81% | | | | |

Timbre, space, and tempo pass: the tempo slider takes the mean beat-tracked tempo from 90 to 167 BPM across its range. Tonality, note density, and percussion do not: those sliders drift toward their words in the embedding while the key fit, the onset count, and the percussive share stay where they were. Scored by embedding alone, as published audio sliders are, all of these would be reported as working.

The measuring tool matters too. With the common librosa tempo estimate, which makes octave errors, the tempo slider scored ρ = -0.01 and looked like a failure. A modern beat tracker (Beat This) shows it works. A descriptor test is only as good as its descriptor. Full table: [`results/main/summary.md`](results/main/summary.md).

The backbone matters. On ACE-Step the same density prompt pair does move the onset rate (ρ = 0.78), and mood follows the major/minor fit there.

**Why not just use an EQ?** For brightness, you should. A spectral tilt applied to the unsteered clip moves the centroid further than the slider (4.1 against 1.7 standard deviations inside the usable span), is perfectly monotone, and keeps more of the piece (CLAP similarity 0.89 against 0.82). The sliders earn their place on attributes no effect can produce: mood, harmony, ensemble size, melody.

### 6. What else moves

![Slope of every descriptor for every ACE-Step slider](results/ace/leakage_lora.png)

Each row is a slider and each cell is how far a descriptor moves per unit of slider, in standard deviations of unsteered clips; bold marks the slider's own descriptor. Sliders are not clean. Mood and melody both brighten the clip (centroid +0.8 per unit), and density, groove, harmony, and ensemble all raise loudness (+0.6 to +0.9). A slider named for one thing moves several, and the table says which.

A measured leak can be cancelled by adding a second slider against it. Mood and melody were rerun with the brightness slider applied at -0.53 and -0.55 times their position (`--mix` in `experiments/evaluate.py`), coefficients read off the slopes above:

| Slider | Brightness leak (std per unit) | Piece kept at ±1 | Enjoyment at -2 | ρ with its CLAP direction |
|---|---:|---:|---:|---:|
| mood | +0.81 | 0.85 | 6.17 | 0.92 |
| mood, brightness cancelled | -0.02 | 0.90 | 6.57 | 0.69 |
| melody | +0.83 | 0.87 | 6.12 | 0.93 |
| melody, brightness cancelled | +0.09 | 0.92 | 6.43 | 0.72 |

The leak goes away and more of the piece survives. The CLAP score for the concept drops as well, which says that part of what CLAP hears as "happier" or "more melodic" was the brightness.

### 7. A slider trained with no text

The second trainer never sees a prompt pair. Generate a corpus with the model (15,552 clips from 648 prompts), measure something on every clip, and train one update with opposite signs on the top and bottom 20% within each prompt. Before the cut, whatever a linear fit on loudness, brightness, and enjoyment explains is removed from the measurement, so the two sets differ in the target and little else ([`results/ace_v2/`](results/ace_v2/README.md)).

![Descriptor response of the set-trained sliders](results/ace_v2/response.png)

Same attributes, both trainers, positions -1 to +1:

| Attribute | Trained from | ρ with the waveform descriptor | Ends ordered | Selectivity | ρ with the MuQ text direction |
|---|---|---:|---:|---:|---:|
| harmony | prompt pair | 0.27 | 65% | 0.8 | 0.63 |
| harmony | two sets | 0.72 | 99% | 6.6 | -0.07 |
| density | prompt pair | 0.74 | 90% | 1.5 | 0.73 |
| density | two sets | 0.63 | 90% | 4.7 | 0.24 |
| ensemble | prompt pair | 0.50 | 82% | 1.5 | 0.65 |
| ensemble | two sets | 0.60 | 94% | 2.4 | 0.03 |
| energy | two sets | 0.83 | 99% | 3.8 | 0.16 |

Selectivity is how far a slider moves its own descriptor relative to the average bystander descriptor. The set-trained sliders are the selective ones, and for harmony they are far better at moving the thing itself. They also show the limits of each kind of score: the set-trained harmony slider moves harmonic change rate and leaves the text-embedding score flat, and the prompt-pair slider does the reverse. Each trainer moves what it was trained on.

Two of these fail (tempo, and mood sorted by a tag score), and all of them stop at ±1: past that the piece is lost and SongEval musicality drops, while Audiobox enjoyment stays flat. The prompt-pair sliders reach ±2.

### 8. Axes nobody named

For each of five broad concepts, 1,024 clips were generated on ACE-Step and their CLAP embeddings decomposed with PCA. The leading components carry 16 to 25% of the variance within a concept, and the tags they point toward and away from read as musical contrasts:

![Variance carried by the first six components of each concept](results/figures/discovery.png)

| Concept | Component | Toward | Away |
|---|---|---|---|
| guitar music | 1 (17.8%) | latin, repetitive, happy, funk | mysterious, distorted, tense, improvised |
| guitar music | 4 (6.8%) | brass, blues, playful, romantic | dreamy, harp, ambient, mysterious |
| piano music | 1 (25.3%) | lo-fi, distorted, hip hop, electric piano | acoustic guitar, reggae, happy, repetitive |
| piano music | 2 (14.0%) | blues, loud, playful, organ | dreamy, ambient, reverberant, harp |
| electronic dance music | 3 (7.1%) | calm, minor key, lo-fi, piano | aggressive, trumpet, loud, latin |
| orchestral music | 4 (7.3%) | piano, lo-fi, dreamy, cello | organ, choir, bells, loud |

All 30 are in [`results/discovery/ace/pca.md`](results/discovery/ace/pca.md) with their correlations to every descriptor.

**The axes become working sliders.** For three concepts we trained the four leading components into sliders with the set trainer (top against bottom 30% of clips along the component) and tested each on 32 fresh seeds of its concept. All 12 follow the axis they were trained on: the rank correlation between slider position and the output's projection on the axis is between 0.43 and 0.81. The words agree too. The leading piano axis was labelled "lo-fi, distorted, hip hop" against "acoustic guitar, reggae, happy" from the corpus alone, and in the slider's output the tags that rise are "distorted, dark-toned, lo-fi" and those that fall are "latin, reggae, happy". Full table: [`results/discovery/ace/sliders.md`](results/discovery/ace/sliders.md). These sliders stay usable to about ±1.5; past that they degrade the clip, unlike the prompt-pair sliders, which reach ±2.

### 9. Axes of real music

The axes above come from the model's own clips. The same decomposition was run on real recordings: 24,975 thirty-second tracks from FMA, with the most vocal 40% removed and each genre's mean subtracted, in two embedding spaces and with three methods ([`results/discovery/real/`](results/discovery/real/README.md)).

| Embedding | Method | Reproduced on a second half of the corpus (matched cosine) |
|---|---|---:|
| MuQ-MuLan | principal components | 0.98 |
| CLAP | principal components | 0.94 |
| MuQ-MuLan | independent components | 0.90 |
| CLAP | independent components | 0.83 |
| CLAP | sparse autoencoder, 1,024 features | 9 features found by both of two seeds |

Independent components of MuQ-MuLan give the axes that read most like music. Two of them are the dimensions listener studies keep finding:

| Axis | One end | Other end | Strongest measured correlate |
|---|---|---|---|
| arousal | quiet, minor key, dreamy, melancholic | aggressive, dry, rhythmic, energetic | spectral flatness -0.40 |
| valence | dark, metal, dark-toned, distorted | playful, pop, simple, happy | content enjoyment -0.35 |
| jazz to electronic | jazz, saxophone, trumpet, improvised | electronic dance, synthesizer, electronic | content enjoyment +0.41 |
| plucked and bowed to synthetic | acoustic guitar, electric guitar, strings | reverberant, choir, synthesizer | production complexity -0.29 |
| piano | piano, melancholic, electric piano, slow | organ, rock, folk, metal | production quality +0.33 |

Training the set slider directly between sets of real recordings did not work: the mismatch between real recordings and the model's own output dominates the loss, and once it cancels almost no steering is left ([`results/real_sets/`](results/real_sets/README.md)). What works is to keep the axis from real music and take the two training sets from the model's own clips sorted along it.

**Sliders along those axes, trained two ways** ([`results/ace_v2/`](results/ace_v2/README.md)). One route sorts the model's own clips along the axis and trains between the two ends, with no text. The other reads the tags at the two ends of the axis and uses them as a prompt pair. Either way the output is scored by its projection on the axis from real recordings, in standard deviations of those recordings. 24 held-out prompts, positions -1 to +1:

| Axis | Trained from | ρ with the axis | Ends ordered | Moved, in std of real music | Piece kept | Enjoyment at -1 / 0 / +1 | Musicality at -1 / 0 / +1 |
|---|---|---:|---:|---:|---:|---|---|
| arousal | prompt pair | 0.91 | 100% | 1.85 | 0.80 | 7.11 / 6.95 / 6.57 | 2.87 / 2.77 / 2.55 |
| arousal | two sets | 0.82 | 99% | 1.19 | 0.84 | 7.09 / 6.95 / 7.16 | 2.72 / 2.77 / 2.68 |
| jazz to electronic | prompt pair | 0.86 | 97% | 1.86 | 0.82 | 6.77 / 6.95 / 7.35 | 2.68 / 2.77 / 2.75 |
| jazz to electronic | two sets | 0.61 | 94% | 0.61 | 0.87 | 7.08 / 6.95 / 7.09 | 2.77 / 2.77 / 2.63 |
| strings to synth | prompt pair | 0.80 | 97% | 1.78 | 0.73 | 5.97 / 6.95 / 7.43 | 2.42 / 2.77 / 2.87 |
| strings to synth | two sets | 0.40 | 82% | 0.44 | 0.87 | 7.15 / 6.95 / 7.03 | 2.68 / 2.77 / 2.69 |
| piano | prompt pair | 0.78 | 97% | 1.56 | 0.80 | 6.80 / 6.95 / 6.76 | 2.77 / 2.77 / 2.71 |
| piano | two sets | 0.63 | 96% | 0.75 | 0.87 | 7.10 / 6.95 / 7.06 | 2.66 / 2.77 / 2.72 |
| valence | prompt pair | 0.42 | 75% | 0.86 | 0.85 | 6.96 / 6.95 / 6.55 | 2.82 / 2.77 / 2.52 |
| valence | two sets | 0.50 | 93% | 0.62 | 0.87 | 7.10 / 6.95 / 7.05 | 2.78 / 2.77 / 2.67 |
| classical to funk | two sets | 0.64 | 96% | 0.77 | 0.85 | 7.16 / 6.95 / 7.09 | 2.72 / 2.77 / 2.67 |
| acoustic to electronic | two sets | 0.55 | 93% | 0.64 | 0.86 | 7.08 / 6.95 / 7.14 | 2.68 / 2.77 / 2.69 |

![Four sliders along axes found in real music, sweeping across their range](results/demo/axes.gif)

*Four axes from real recordings as sliders, one prompt and seed per panel, swept from the middle to +2, back to -2, and home.*

- Every one of the twelve follows its axis. The prompt pair is the stronger route on four of five axes and stays usable beyond ±1: over its usable span of -2 to +1 the arousal slider moves a clip 2.4 standard deviations of real music.
- The set-trained sliders move less, keep more of the piece, and hold both quality scores level at both ends.
- Nobody chose these prompt pairs. "Quiet, dreamy, melancholic, minor key" against "aggressive, energetic, rhythmic, dry" is what an independent component of real-music embeddings looks like when its two ends are put into words.

### 10. How much of real music the model covers

Projecting 15,552 generated clips (648 prompts) on the real-music axes shows where the model's output sits and how far it spreads ([`results/coverage/`](results/coverage/README.md)).

![Spread of generated clips along each real-music axis](results/figures/coverage.png)

Along every axis the model is narrower than real music. Across all 648 prompts it spans 28 to 83% of the real spread, and with one prompt and many seeds 38 to 69% of what recordings of one genre span. Its average clip sits 1.35 standard deviations toward the quiet end of the arousal axis and 1.36 toward the playful end of the valence axis. Stable Audio Open shows the same pattern on the CLAP axes. The directions along which generated clips differ most are also mostly different ones: the top eight principal directions of the generated corpus contain 31% of the top eight of the real corpus.

This is the case for sliders along real-music axes, and the sliders of the previous section deliver on it. Where the same 24 prompts and seeds land on each axis, with 0 the average real recording and the unit one standard deviation of real music:

| Axis (positive end) | Slider at -2 | at -1 | unsteered | at +1 | at +2 |
|---|---:|---:|---:|---:|---:|
| arousal (quiet, dreamy) | -0.26 | 0.31 | 1.21 | 2.16 | 2.63 |
| jazz to electronic (jazz) | -0.51 | -0.12 | 0.60 | 1.74 | 2.29 |
| strings to synth (guitars, strings) | -1.21 | -0.77 | 0.23 | 1.01 | 1.19 |
| piano (piano) | -0.90 | -0.51 | 0.23 | 1.05 | 1.38 |
| valence (dark, distorted) | -1.86 | -1.67 | -1.26 | -0.81 | 0.30 |

Unsteered, the model's clips sit 1.21 standard deviations on the quiet side of the arousal axis. One slider takes them across the mean of real music to -0.26. For comparison, across 648 different prompts the model's clips have a standard deviation of 0.83 on this axis.

### 11. Axes inside the model

The same question can be put to the generator directly, with no embedding model in between. We recorded the mean cross-attention output of each of ACE-Step's 32 blocks for 768 clips, removed each prompt's mean, and took principal components ([`results/discovery/internal/`](results/discovery/internal/README.md)). Eight axes carry 90% of the variance and reproduce across disjoint prompt sets (0.82). Steering along them shows what they are:

- The two strongest raise level and lower spectral flatness, and move tags from lo-fi toward complex and live-sounding. They are production axes.
- The other six leave 95% or more of the piece in place at twelve natural standard deviations and act differently on different prompts (consistency 0.10 to 0.21).

Arousal, valence, and instrument contrasts did not come out of the model's raw activations. They came out of a music embedding of real recordings. Where the axes are looked for decides what kind is found.

### 12. How well a slider learns its target

![Share of the guidance target not yet reproduced, by training iteration](results/figures/training_curves.png)

The prompt-pair trainer asks the slider to reproduce a shift in the frozen model's prediction. After 1,000 iterations the Stable Audio sliders reproduce about 62% of that shift on average and the ACE-Step sliders about 31%. The ACE-Step sliders work anyway, as section 2 shows, and they are the ones with the most room left.

Both obvious follow-ups were run on the mood slider ([`results/ace/ablations.md`](results/ace/ablations.md)). Three times the training raises the descriptor correlation from 0.39 to 0.52. Rank 16, a doubled guidance multiplier, and restricting the update to cross-attention all leave it at 0.29 to 0.32. The CLAP direction score reads 0.92 or 0.93 for every one of the five recipes, so the embedding score cannot tell a better slider from a worse one here and the waveform descriptor can. The same file has the two-slider grids: effects roughly add, with interaction terms a fifth to a third of the main effects.

### 13. Against other methods

Four other ways of moving the same attribute were run through the same protocol on ACE-Step, on the same prompts and seeds. Rank correlation with the waveform descriptor, then with the MuQ-MuLan direction:

| Slider | Trained slider | Prompt-pair guidance | Activation steering | Prompt interpolation |
|---|---:|---:|---:|---:|
| brightness | 0.97 / 0.80 | 0.89 / 0.81 | 0.93 / 0.81 | 0.49 / 0.35 |
| density | 0.78 / 0.86 | 0.74 / 0.81 | 0.71 / 0.86 | 0.10 / 0.38 |
| ensemble | 0.60 / 0.77 | 0.55 / 0.75 | 0.58 / 0.67 | 0.25 / 0.25 |
| melody | 0.48 / 0.89 | 0.51 / 0.83 | 0.53 / 0.83 | 0.31 / 0.38 |
| harmony | 0.41 / 0.73 | 0.08 / 0.68 | 0.24 / 0.72 | -0.01 / 0.23 |
| mood | 0.39 / 0.88 | 0.45 / 0.87 | 0.41 / 0.88 | 0.25 / 0.63 |
| groove (MuQ only) | 0.56 | 0.71 | 0.58 | 0.12 |
| tension (MuQ only) | 0.86 | 0.88 | 0.91 | 0.69 |
| Forward passes per sampling step | 1 | 3 | 1 | 1 |
| Training | about 22 min per slider on one GPU | none | none | none |

![Descriptor response of four methods on six sliders](results/ace/compare_response.png)

- **The trained slider reproduces prompt-pair guidance at a third of the sampling cost.** Guidance is the slider's own training target applied directly at every step (the FreeSliders recipe), and it needs two extra forward passes per step. The slider scores higher on brightness and harmony (0.41 against 0.08) and within the confidence interval on the other four descriptors.
- **Prompt interpolation does not work on this model.** Moving the text conditioning toward the positive or negative prompt gives the lowest correlation on every attribute, and enjoyment at the ends falls to between 4.2 and 5.9 (unsteered: 6.96).
- **Activation steering is a real competitor**, covered next.

Full table with confidence intervals: [`results/ace/summary.md`](results/ace/summary.md).

**Activation steering** (the training-free approach TADA argues for: add the mean difference in cross-attention output between the positive and the negative prompt, times the slider position). It needs no training at all.

| Slider | Method | ρ with descriptor | Descriptor moved in usable span (std) | Piece kept | Enjoyment at ends |
|---|---|---:|---:|---:|---:|
| brightness | trained slider | 0.97 | 4.41 | 0.74 | 6.04 |
| | activation steering | 0.93 | 1.72 | 0.83 | 6.11 |
| density | trained slider | 0.78 | 1.25 | 0.84 | 6.09 |
| | activation steering | 0.71 | 1.63 | 0.87 | 6.46 |
| ensemble | trained slider | 0.60 | 0.92 | 0.80 | 6.56 |
| | activation steering | 0.58 | 0.67 | 0.85 | 6.61 |
| melody | trained slider | 0.48 | 0.97 | 0.80 | 6.47 |
| | activation steering | 0.53 | 0.92 | 0.83 | 6.64 |
| harmony | trained slider | 0.41 | 0.84 | 0.83 | 6.64 |
| | activation steering | 0.24 | 0.60 | 0.83 | 6.72 |
| mood | trained slider | 0.39 | 0.87 | 0.79 | 6.65 |
| | activation steering | 0.41 | 1.28 | 0.83 | 6.37 |

Activation steering is a real competitor. Its rank correlation is within 0.05 of the trained slider's on four of six attributes, and it keeps more of the piece on five and the same on the sixth. The trained slider moves its descriptor further on brightness, ensemble, and harmony; activation steering moves it further on density and mood. Neither dominates, which agrees with TADA's finding on a different ACE-Step version and argues against treating LoRA sliders as the default.

**Community sliders.** Nineteen Concept Sliders for ACE-Step 1.5 XL are [published on Hugging Face](https://huggingface.co/Xanthius/Ace-Step-1.5-XL-Concept-Sliders) with no evaluation. `SliderBank.load_peft` loads them unchanged, and seven were measured here for the first time (24 prompts, 2 seeds, positions -6 to +6):

| Community slider | Descriptor | ρ | Moved in usable span (std) | Piece kept |
|---|---|---:|---:|---:|
| energetic-calm | spectral flux | 0.91 | 2.26 | 0.59 |
| bass | energy below 150 Hz | 0.83 | 0.86 | 0.86 |
| reverb | energy decay time | 0.81 | 1.36 | 0.80 |
| tempo | beat-tracked tempo | 0.61 | 0.60 | 0.70 |
| happiness | major/minor fit | 0.40 | 0.71 | 0.87 |
| drum | percussive share | 0.32 | -0.35 | 0.82 |

Most of them do what their names say; the drum slider moves the percussive share the wrong way. They need much larger positions than ours (their usable spans run to ±6) because each unit does less.

## Try it

```bash
pip install -e ".[dev]"
python -m pytest -q          # 57 CPU tests, no model download
```

With a GPU:

```bash
pip install -e ".[model,experiments,demo]" audiobox_aesthetics
python -m audiosliders.train mood --backbone ace-turbo --eta 2 --out runs/sliders/ace   # about 20 minutes
python experiments/sweep.py runs/sliders/ace/mood.safetensors --backbone ace-turbo      # descriptor table per position
python -m audiosliders.server --sliders runs/sliders/ace --backbone ace-turbo           # live page: type a prompt, drag sliders
```

A slider with no text, along an axis found in real music:

```bash
python experiments/make_corpus.py --backbone ace-turbo --split large --seeds 24 --save-audio --out runs/corpus/ace_large
venv-muq/bin/python experiments/muq_embed.py --corpus runs/corpus/ace_large --music runs/corpus/ace_large/audio --offset 0
python experiments/discover.py --corpus runs/corpus/real_ace --method ica --emb muq --max-vocal 0.6 \
    --vocab runs/reference/vocab_muq.npz --out runs/discovery/real_muq_ica
python -m audiosliders.contrast arousal --backbone ace-turbo --corpus runs/corpus/ace_large --emb muq \
    --by direction:runs/discovery/real_muq_ica/directions.npy:3 --fraction 0.2 --balance ce --symmetry 0 --out runs/sliders/ace_v2
```

The server page adds a live panel under the recorded demo: type any prompt, set any combination of sliders, and it returns the clip in two to four seconds with its enjoyment score and descriptors. On a remote GPU box, forward the port (`ssh -L 7860:127.0.0.1:7860 host`) and open `localhost:7860`.

ACE-Step 1.5 is MIT-licensed and ungated. Stable Audio Open 1.0 needs a Hugging Face account that has accepted its license.

## What happened to v0

The first version of this repository never ran a model. Its sampling script wrote white noise, its embedding script wrote hash-seeded random vectors, and its training script wrote a JSON plan. It is preserved unchanged in [`legacy/`](legacy/) with a note on what each piece did.

## Layout

- [`audiosliders/`](audiosliders/): backbones, slider LoRA, both trainers, measurement code, demo server
- [`configs/`](configs/): 28 slider definitions, the 48/24 train/eval prompt split, 648 prompts for the large corpus, discovery concepts
- [`experiments/`](experiments/): scripts and job lists behind each number
- [`results/`](results/): tables and figures
- [`paper/`](paper/): draft on the ISMIR template (`tectonic -X compile paper/audiosliders.tex`)
- [`docs/`](docs/): demo page and the [research map](docs/RESEARCH.md) of prior work
- [`tests/`](tests/): CPU tests

## Credits

By Taka Khoo. Built on ACE-Step 1.5 and Stable Audio Open 1.0. The prompt-pair objective is from Concept Sliders and the discovery idea from SliderSpace (Gandikota et al.). Audiobox Aesthetics is from Meta, CLAP from LAION, and the real-music reference is the FMA dataset.
