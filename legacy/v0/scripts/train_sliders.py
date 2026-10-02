#!/usr/bin/env python3
"""Stub script for training SliderSpace-inspired LoRA adapters on audio models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LoRA sliders aligned with PCA principal directions (stub)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_config.yaml"),
        help="YAML configuration describing model + paths.",
    )
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        help="Concept name whose PCA directions we want to align with sliders.",
    )
    parser.add_argument(
        "--pca-path",
        type=Path,
        default=None,
        help="Path to PCA .npz file. Defaults to data/pca/{concept}_pca.npz.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="LoRA rank to use for adapters once training is implemented.",
    )
    parser.add_argument(
        "--num-sliders",
        type=int,
        default=8,
        help="How many PCA directions to convert into sliders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sliders"),
        help="Directory where trained sliders/checkpoints will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    cfg = load_config(args.config)

    pca_root = Path(cfg.get("paths", {}).get("pca_dir", "data/pca"))
    pca_path = args.pca_path or pca_root / f"{args.concept}_pca.npz"
    if not pca_path.exists():
        raise FileNotFoundError(f"PCA file not found: {pca_path}")

    output_dir = args.output_dir / args.concept
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "concept": args.concept,
        "pca_path": str(pca_path),
        "rank": args.rank,
        "num_sliders": args.num_sliders,
        "audio_model_name": cfg.get("audio_model", {}).get("name", "TBD"),
        "embedding_model_name": cfg.get("embedding_model", {}).get("name", "TBD"),
        "status": "TODO: implement training loop that optimizes LoRA adapters "
        "so CLAP embedding deltas align with PCA directions.",
    }

    plan_path = output_dir / "training_plan.json"
    with plan_path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)

    logging.info(
        "Slider training stub executed. Stored plan at %s. Implement actual training next.",
        plan_path,
    )


if __name__ == "__main__":
    main()
