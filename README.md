# Audio Sliders

Continuous controls for text-to-music diffusion. A slider is a small LoRA on a frozen music model. Moving it changes one property of the generated piece (brightness, note density, mood, harmony, ensemble size, and so on) while the prompt and seed stay fixed. Every slider is scored three ways: does a measured property of the audio follow it, how much of the original piece survives, and does the result still sound like music.

> **Status (2 Oct 2026): rebuild in progress.** Training and evaluation are running now and this page will be replaced with the full results. Everything stated below is already measured and reproducible from this repository. Numbers marked *pilot* come from 8 held-out prompts and will be superseded.

## What is established so far

- **The sliders move what they claim to move.** The brightness slider, trained from the prompt pair "bright, crisp" / "dark, muffled", shifts the mean spectral centroid of held-out prompts from 280 Hz at -2 to 4,080 Hz at +2 (1,335 Hz unsteered), and all 8 pilot trajectories are ordered correctly end to end (*pilot*).
- **A slider can be trained with no text at all.** Sorting the model's own clips by measured centroid and training one LoRA with opposite signs on the two ends gives a brightness slider that moves the centroid from 408 Hz at -2 to 1,488 Hz at +1 (899 Hz unsteered) with stereo width, loudness, and bass nearly unchanged. The text slider drags all three along (*pilot*).
- **Pushing a slider too far stops sounding like music, and that is measurable.** On one jazz-trio clip, Audiobox Aesthetics "content enjoyment" stays at 7.3 for positions up to ±0.5, falls to about 6 at ±1, and to 3 to 5 at ±2. For scale, 2,000 real recordings from FMA average 6.1.
- **Two backbones run behind one interface.** Stable Audio Open 1.0 (44.1 kHz, 50 steps, 0.65 s per 10 s clip) and ACE-Step 1.5 XL turbo (48 kHz, 8 steps, 0.4 s per clip, production-quality score 8.2 against 6.8 for FMA).
- **The first version of this repository never ran a model.** It wrote white noise, random vectors, and a JSON plan. It is preserved in [`legacy/`](legacy/).

## How a slider is trained

Two trainers share one LoRA mechanism ([`audiosliders/lora.py`](audiosliders/lora.py)): a low-rank update on the transformer's attention and feed-forward layers whose strength is a number read at every forward pass, positive or negative.

1. **From a prompt pair** ([`train.py`](audiosliders/train.py)). The Concept Sliders objective: the LoRA at scale ±1 is trained to reproduce the frozen model's prediction plus or minus a multiple of the difference between its predictions for "prompt, bright" and "prompt, dark".
2. **From two sets of clips** ([`contrast.py`](audiosliders/contrast.py)). The LoRA at +1 is trained with the plain denoising loss on a "high" set and at -1 on a "low" set. Shared content cancels because one update serves both ends with opposite sign. The sets are the model's own clips sorted by a measured descriptor, by an aesthetics score, or along a principal direction of their CLAP embeddings.

## How a slider is measured

- **Descriptors** ([`descriptors.py`](audiosliders/descriptors.py)): spectral centroid, onset rate, pulse clarity, energy decay time, percussive share, bass and stereo-side ratios, key clarity, major/minor fit, harmonic change rate, level dynamics. Computed from the waveform with no learned model.
- **Musical quality** ([`quality.py`](audiosliders/quality.py)): Audiobox Aesthetics per clip, compared with real recordings.
- **Musical meaning** ([`tags.py`](audiosliders/tags.py)): 80 instrument, genre, mood, and character tags scored in CLAP space, used to say in words what a slider or a discovered direction changes.
- **Preservation**: CLAP similarity, chroma similarity, and onset-envelope correlation against the same seed with the slider at zero.
- **Baselines** ([`methods.py`](audiosliders/methods.py), [`dsp.py`](audiosliders/dsp.py)): the guidance target applied directly (three times the compute), prompt interpolation, and plain signal processing where an effect exists.

Prior work, what it offers, and the gaps this project targets are in [`docs/RESEARCH.md`](docs/RESEARCH.md).

## Try it

```bash
pip install -e ".[dev]"
python -m pytest -q          # 40 CPU tests, no model download
```

With a GPU and access to the model weights:

```bash
pip install -e ".[model,experiments,demo]" audiobox_aesthetics
python -m audiosliders.train brightness --out runs/sliders            # about 20 minutes
python experiments/sweep.py runs/sliders/brightness.safetensors       # descriptor table per slider position
python -m audiosliders.server --sliders runs/sliders --port 7860      # live demo page
```

The static demo page lives in [`docs/index.html`](docs/index.html) and will be published with audio once the evaluation finishes.

## Layout

- [`audiosliders/`](audiosliders/): backbones, slider LoRA, both trainers, measurement code, demo server
- [`configs/`](configs/): 20 slider definitions, the 48/24 train/eval prompt split, discovery concepts
- [`experiments/`](experiments/): the scripts and job lists behind each number
- [`tests/`](tests/): CPU tests
- [`docs/`](docs/): research map and demo page
- [`legacy/`](legacy/): the original scaffold

## Credits

By Taka Khoo. Built on Stable Audio Open (Stability AI) and ACE-Step 1.5, with the training objective from Concept Sliders and the discovery idea from SliderSpace (Gandikota et al.).
