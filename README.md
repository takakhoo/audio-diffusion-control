# Audio Sliders

Continuous controls for text-to-music diffusion. Each slider is a small LoRA on Stable Audio Open 1.0 that moves one property of the generated music (brightness, note density, reverberation, percussion, and so on) while the prompt and seed stay fixed. Every slider is scored against a descriptor measured from the output waveform.

> **Status (2 Oct 2026): rebuild in progress.** The first sliders are training now. This page will be replaced with results, audio, and a demo as they land. The earlier scaffold, which never ran a diffusion model, is preserved in [`legacy/`](legacy/).

## Layout

- [`audiosliders/`](audiosliders/): backbone wrapper, slider LoRA, trainer, CLAP, descriptors
- [`configs/`](configs/): slider definitions and the train/eval prompt split
- [`experiments/`](experiments/): scripts behind each reported number
- [`tests/`](tests/): CPU tests, no model download needed

```bash
pip install -e ".[dev]"
python -m pytest -q
```
