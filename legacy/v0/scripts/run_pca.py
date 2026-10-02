#!/usr/bin/env python3
"""Run PCA on stored embedding matrices."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from sklearn.decomposition import PCA


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCA on CLAP embeddings.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        help="Concept name to process.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=16,
        help="Number of principal components to retain.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=None,
        help="Optional explicit path to embeddings npz.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit PCA output file (.npz).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    cfg = load_config(args.config)

    embeddings_root = Path(cfg.get("paths", {}).get("embeddings_dir", "data/embeddings"))
    pca_root = Path(cfg.get("paths", {}).get("pca_dir", "data/pca"))
    pca_root.mkdir(parents=True, exist_ok=True)

    embeddings_path = args.embeddings_path or embeddings_root / f"{args.concept}.npz"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    logging.info("Loading embeddings from %s", embeddings_path)
    with np.load(embeddings_path, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        filenames = data["filenames"]
        embedding_model = data["embedding_model"].item() if data["embedding_model"].shape == () else data["embedding_model"]

    logging.info("Running PCA with %d components on matrix of shape %s", args.n_components, embeddings.shape)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all() or not 1 <= args.n_components <= min(embeddings.shape):
        raise ValueError("Need finite 2D embeddings and 1 <= n_components <= min(samples, features)")
    pca = PCA(n_components=args.n_components, svd_solver="full", whiten=False)
    projections = pca.fit_transform(embeddings)

    output_path = args.output_path or pca_root / f"{args.concept}_pca.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        singular_values=pca.singular_values_,
        projections=projections,
        filenames=filenames,
        embedding_model=embedding_model,
    )

    summary = {
        "concept": args.concept,
        "n_components": args.n_components,
        "embedding_shape": embeddings.shape,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "source_embeddings": str(embeddings_path),
        "output_npz": str(output_path),
    }
    summary_path = output_path.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logging.info("Saved PCA results to %s and summary to %s", output_path, summary_path)


if __name__ == "__main__":
    main()
