"""Unsupervised directions in the corpus: PCA of CLAP embeddings, read against descriptors.

Each prompt's mean embedding is removed first, so the components describe how clips of
the same prompt differ from one another. For every component the script reports its share
of that within-prompt variance and its rank correlation with each measured descriptor,
which is what tells us what a discovered direction means before any slider is trained.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from audiosliders.contrast import load_corpus, principal_directions

KEYS = ["centroid_hz", "rolloff_hz", "bass_ratio", "flatness", "flux", "rms_db", "onset_rate", "pulse_bpm",
        "percussive_ratio", "decay_s", "side_ratio", "majorness", "crest_db", "clap_prompt"]

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--corpus", default="runs/corpus/train")
ap.add_argument("--out", default="results/discovery")
ap.add_argument("--n", type=int, default=16)
args = ap.parse_args()

corpus = load_corpus(args.corpus)
rows = corpus["rows"]
groups = np.array([r["prompt_index"] for r in rows])
directions, share = principal_directions(corpus["clap"], groups, args.n)


def within(values: np.ndarray) -> np.ndarray:
    out = values.astype(float).copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = (out[m] - np.nanmean(out[m])) / (np.nanstd(out[m]) + 1e-9)
    return out


desc = {k: within(np.array([r.get(k, np.nan) for r in rows], dtype=float)) for k in KEYS}
proj = corpus["clap"] @ directions.T
table = []
for i in range(args.n):
    p = within(proj[:, i])
    corr = {}
    for k in KEYS:
        ok = np.isfinite(desc[k])
        corr[k] = float(spearmanr(p[ok], desc[k][ok]).statistic)
    best = sorted(corr, key=lambda k: -abs(corr[k]))[:3]
    table.append(dict(component=i, variance_share=float(share[i]), correlations=corr, strongest=best))
    print(f"PC{i:02d} {100 * share[i]:5.1f}%  " + "  ".join(f"{k} {corr[k]:+.2f}" for k in best))

# The reverse view: how much of each descriptor do the first n components explain together?
explained = {}
for k in KEYS:
    ok = np.isfinite(desc[k])
    x = np.stack([within(proj[:, i])[ok] for i in range(args.n)], 1)
    coef, *_ = np.linalg.lstsq(x, desc[k][ok], rcond=None)
    explained[k] = float(1 - ((desc[k][ok] - x @ coef) ** 2).sum() / (desc[k][ok] ** 2).sum())
print("R^2 of each descriptor from the components:", {k: round(v, 2) for k, v in explained.items()})

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "pca.json").write_text(json.dumps(dict(n_clips=len(rows), n_prompts=int(len(np.unique(groups))),
                                             components=table, descriptor_r2=explained), indent=1))
np.save(out / "directions.npy", directions)
