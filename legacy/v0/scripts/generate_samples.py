#!/usr/bin/env python3
"""Generate placeholder audio samples for a given text prompt.

This script mirrors the SliderSpace sampling stage: it loads a YAML config,
generates multiple clips for a fixed prompt, and stores both the resulting
audio and per-sample metadata so that downstream embedding/PCA scripts can
consume a consistent dataset.

The actual text-to-audio diffusion model is not wired up yet. Instead, we
generate deterministic noise waveforms as stand-ins and clearly mark the
sections that must be replaced with Stable Audio Open (or another diffusion
model) once the checkpoints and inference code are available on this system.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
import wave


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class AudioGenerator:
    """Placeholder audio generator until Stable Audio Open is integrated."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        audio_cfg = cfg.get("audio_model", {})
        self.sample_rate = int(audio_cfg.get("sample_rate", 44100))
        self.duration = float(audio_cfg.get("max_duration_sec", 10.0))

    def generate(self, prompt: str, seed: int) -> np.ndarray:
        """Return a dummy waveform for the given prompt and seed.

        TODO: Replace this method with real inference that loads Stable Audio
        Open (or another model) and returns multi-channel audio.
        """
        rng = np.random.default_rng(seed)
        num_samples = int(self.sample_rate * self.duration)
        waveform = rng.standard_normal(num_samples).astype(np.float32)
        waveform *= 0.02  # keep the amplitude low to avoid clipping
        return waveform


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write a mono waveform to disk using 16-bit PCM encoding."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def sanitize_concept(name: str) -> str:
    """Convert a free-form prompt into a filesystem-friendly concept string."""
    safe = "".join(c.lower() if c.isalnum() else "_" for c in name)
    safe = "_".join(filter(None, safe.split("_")))
    return safe[:80] or "concept"


def save_metadata(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate audio samples for SliderSpace-style discovery."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt to render. Defaults to experiment.default_prompt.",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default=None,
        help="Optional concept name used for organizing outputs.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to generate (defaults to config experiment.default_num_samples).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the root directory for raw audio outputs.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Additive seed offset to make repeated runs reproducible.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    cfg = load_config(args.config)

    prompt = args.prompt or cfg.get("experiment", {}).get(
        "default_prompt", "solo jazz guitar, warm tone, swing feel"
    )
    concept = args.concept or sanitize_concept(prompt)
    num_samples = args.num_samples or cfg.get("experiment", {}).get(
        "default_num_samples", 32
    )

    raw_root = Path(
        args.output_root
        or cfg.get("paths", {}).get("raw_audio_dir", "data/raw")
    )
    concept_dir = raw_root / concept
    concept_dir.mkdir(parents=True, exist_ok=True)

    generator = AudioGenerator(cfg)
    logging.info(
        "Generating %d samples for prompt '%s' into %s",
        num_samples,
        prompt,
        concept_dir,
    )

    for idx in range(num_samples):
        seed = args.seed_offset + idx
        audio = generator.generate(prompt, seed)
        stem = f"seed_{seed:05d}"
        wav_path = concept_dir / f"{stem}.wav"
        meta_path = concept_dir / f"{stem}.json"

        write_wav(wav_path, audio, generator.sample_rate)
        metadata = {
            "prompt": prompt,
            "concept": concept,
            "seed": seed,
            "sample_rate": generator.sample_rate,
            "duration_sec": len(audio) / generator.sample_rate,
            "model_name": "placeholder-white-noise-NOT-DIFFUSION",
            "requested_model_not_loaded": cfg.get("audio_model", {}).get("name", "TBD"),
            "is_placeholder": True,
            "guidance_scale": cfg.get("audio_model", {}).get("guidance_scale"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "notes": "Placeholder audio from stochastic stub. Replace with diffusion output.",
        }
        save_metadata(meta_path, metadata)
        logging.debug("Wrote %s", wav_path)

    logging.info("Done. Wrote %d placeholder clips for concept '%s'.", num_samples, concept)


if __name__ == "__main__":
    main()
