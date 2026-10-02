#!/usr/bin/env python3
"""Compute (placeholder) CLAP embeddings for generated audio clips."""

from __future__ import annotations

import argparse
import json
import logging
import hashlib
import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Load a PCM WAV file and return mono audio in [-1, 1]."""
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.getnframes()
        audio_bytes = wav_file.readframes(frames)
        nchannels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()

    if sample_width != 2:
        raise ValueError(f"{path} has unsupported sample width {sample_width}")

    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    if nchannels > 1:
        audio = audio.reshape(-1, nchannels).mean(axis=1)
    audio /= 32768.0
    return audio, sample_rate


class ClapEmbedder:
    """Minimal placeholder for CLAP embeddings.

    TODO: Replace this class with actual CLAP model loading and inference.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        emb_cfg = cfg.get("embedding_model", {})
        self.embedding_dim = int(emb_cfg.get("embedding_dim", 512))
        self.name = "placeholder-random-sha256-NOT-CLAP"

    def encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a deterministic pseudo-embedding based on the waveform."""
        digest = hashlib.sha256(audio.tobytes() + sample_rate.to_bytes(4, "little")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.embedding_dim).astype(np.float32)


def collect_audio_files(concept_dir: Path) -> List[Path]:
    wavs = sorted(concept_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV files found in {concept_dir}")
    return wavs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CLAP embeddings for audio samples.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_config.yaml"),
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        help="Concept name whose raw audio clips will be embedded.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Override the raw audio root directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output path for the embeddings npz file.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    cfg = load_config(args.config)

    raw_root = Path(args.raw_root or cfg.get("paths", {}).get("raw_audio_dir", "data/raw"))
    embeddings_root = Path(cfg.get("paths", {}).get("embeddings_dir", "data/embeddings"))
    embeddings_root.mkdir(parents=True, exist_ok=True)

    concept_dir = raw_root / args.concept
    wav_paths = collect_audio_files(concept_dir)

    embedder = ClapEmbedder(cfg)
    embeddings = []
    index_entries: List[Dict[str, Any]] = []

    logging.info("Computing embeddings for %d files from %s", len(wav_paths), concept_dir)
    for wav_path in wav_paths:
        audio, sample_rate = load_wav(wav_path)
        embedding = embedder.encode(audio, sample_rate)
        embeddings.append(embedding)

        meta_path = wav_path.with_suffix(".json")
        metadata = {}
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

        index_entries.append(
            {
                "file": str(wav_path),
                "sample_rate": sample_rate,
                "metadata": metadata,
            }
        )

    embedding_matrix = np.stack(embeddings, axis=0)
    output_path = args.output_path or embeddings_root / f"{args.concept}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        embeddings=embedding_matrix,
        filenames=[str(p) for p in wav_paths],
        embedding_model=embedder.name,
        embedding_dim=embedder.embedding_dim,
        is_placeholder=True,
    )

    index_path = output_path.with_suffix(".index.json")
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index_entries, handle, indent=2)

    logging.info(
        "Saved embeddings to %s (shape=%s) and metadata index to %s",
        output_path,
        embedding_matrix.shape,
        index_path,
    )


if __name__ == "__main__":
    main()
