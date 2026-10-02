"""Unsupervised axes: PCA over CLAP embeddings of many clips, read in musical terms.

    python experiments/discover.py --corpus runs/corpus/concepts --per-prompt --out results/discovery/concepts

With --per-prompt every prompt is decomposed separately (one concept, many seeds), which is
the SliderSpace setting. Without it, each prompt's mean is removed and all prompts are
pooled. For every component the script reports its share of variance, the tags it points
toward and away from, and its rank correlation with each measured descriptor and with the
aesthetics scores. Nothing here trains a slider; it says what a direction means first.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from audiosliders.contrast import load_corpus, principal_directions
from audiosliders.tags import label_direction

KEYS = ["centroid_hz", "bass_ratio", "flatness", "flux", "rms_db", "onset_rate", "pulse_bpm", "pulse_clarity",
        "percussive_ratio", "decay_s", "side_ratio", "majorness", "key_clarity", "harmonic_change", "dynamics_db",
        "clap_prompt", "ce", "pq", "pc"]


def zscore(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = values.astype(float).copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = (out[m] - np.nanmean(out[m])) / (np.nanstd(out[m]) + 1e-9)
    return out


def analyse(rows, clap, groups, n, vocab):
    directions, share = principal_directions(clap, groups, n)
    proj = clap @ directions.T
    desc = {k: zscore(np.array([r.get(k, np.nan) for r in rows], dtype=float), groups) for k in KEYS
            if any(k in r for r in rows)}
    table = []
    for i in range(len(directions)):
        p = zscore(proj[:, i], groups)
        corr = {}
        for k, v in desc.items():
            ok = np.isfinite(v)
            corr[k] = float(spearmanr(p[ok], v[ok]).statistic) if ok.sum() > 10 else float("nan")
        entry = dict(component=i, variance_share=float(share[i]), correlations=corr,
                     strongest=sorted((k for k in corr if np.isfinite(corr[k])), key=lambda k: -abs(corr[k]))[:3])
        if vocab is not None:
            entry["toward"], entry["away"] = label_direction(directions[i], vocab, k=4)
        table.append(entry)
    return directions, table


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default="runs/reference/vocab.npz")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--per-prompt", action="store_true")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    rows, clap = corpus["rows"], corpus["clap"]
    groups = np.array([r["prompt_index"] for r in rows])
    vocab = np.load(args.vocab)["text"] if Path(args.vocab).exists() else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report, lines = {}, []
    sets = [(rows[int(np.argmax(groups == g))]["prompt"], groups == g) for g in np.unique(groups)] if args.per_prompt \
        else [("all prompts, per-prompt mean removed", np.ones(len(rows), dtype=bool))]
    for name, mask in sets:
        sub = [r for r, m in zip(rows, mask) if m]
        directions, table = analyse(sub, clap[mask], groups[mask], args.n, vocab)
        report[name] = dict(n_clips=int(mask.sum()), components=table)
        np.save(out / f"directions_{name.replace(' ', '_').replace(',', '')[:40]}.npy", directions)
        lines += [f"### {name} ({int(mask.sum())} clips)", "",
                  "| PC | Variance | Toward | Away | Strongest measured correlates |", "|---:|---:|---|---|---|"]
        for e in table:
            corr = ", ".join(f"{k} {e['correlations'][k]:+.2f}" for k in e["strongest"])
            lines.append(f"| {e['component']} | {100 * e['variance_share']:.1f}% | {', '.join(e.get('toward', []))} | "
                         f"{', '.join(e.get('away', []))} | {corr} |")
        lines.append("")
    (out / "pca.json").write_text(json.dumps(report, indent=1))
    (out / "pca.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
