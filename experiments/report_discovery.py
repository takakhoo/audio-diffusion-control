"""Summarise sliders trained on discovered (PCA) directions.

    python experiments/report_discovery.py --eval runs/eval/ace_pca --pca runs/discovery/ace_concepts/pca.json \
        --vocab runs/reference/vocab.npz --out results/discovery

For each slider: the component's share of variance and tag labels from the corpus, then
what the trained slider does on fresh seeds of the same concept. `follows` is the rank
correlation between slider position and the projection of the output's CLAP embedding on
the direction the slider was trained along.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

from audiosliders import metrics as M
from audiosliders.tags import tag_shift

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--eval", required=True)
ap.add_argument("--pca", required=True)
ap.add_argument("--vocab", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

concepts = yaml.safe_load(Path("configs/prompts.yaml").read_text())["concepts"]
pca = json.loads(Path(args.pca).read_text())
text = np.load(args.vocab)["text"]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

table, curves = [], {}
for run in sorted(Path(args.eval).iterdir()):
    m = re.fullmatch(r"c(\d+)_pc(\d+)", run.name)
    if not m or not (run / "rows.jsonl").exists():
        continue
    c, i = int(m.group(1)), int(m.group(2))
    rows = M.load_rows(run)
    emb = np.load(run / "clap.npy")
    scales = sorted({r["scale"] for r in rows})
    lo, hi = M.usable_span(rows)
    inside = M.within(rows, lo, hi)
    hi_mask = np.array([r["scale"] == hi for r in rows])
    lo_mask = np.array([r["scale"] == lo for r in rows])
    up, down = tag_shift(emb[hi_mask], emb[lo_mask], text, k=3) if hi > lo else ([], [])
    comp = pca[concepts[c]]["components"][i]
    resp = M.response(rows, "clap_dir")
    curves[run.name] = dict(scales=resp["scales"].tolist(), level=resp["level"].tolist(),
                            ce=M.response(rows, "ce")["level"].tolist())
    keep = [r["clap_keep"] for r in inside if r["scale"] in (lo, hi) and r["scale"] != 0]
    table.append(dict(
        concept=concepts[c], component=i, variance_share=comp["variance_share"],
        corpus_toward=comp.get("toward", []), corpus_away=comp.get("away", []),
        follows=M.monotonicity(rows, "clap_dir")["rho"], ordered=M.monotonicity(rows, "clap_dir")["consistent"],
        usable_lo=lo, usable_hi=hi,
        moved=float(M.response(inside, "clap_dir")["level"][-1] - M.response(inside, "clap_dir")["level"][0]) if hi > lo else 0.0,
        natural_spread=float(np.std([r["clap_dir"] for r in rows if r["scale"] == 0])),
        kept=float(np.mean(keep)) if keep else float("nan"),
        ce_zero=float(np.mean([r["ce"] for r in rows if r["scale"] == 0])),
        ce_ends=float(np.mean([r["ce"] for r in rows if r["scale"] in (scales[0], scales[-1])])),
        rises=[t for t, _ in up], falls=[t for t, _ in down],
    ))

lines = ["| Concept | PC | Variance | Corpus labels (toward / away) | Follows ρ | Usable span | Moved (in std of unsteered) | "
         "Piece kept | Tags that rise / fall in the output |", "|---|---:|---:|---|---:|---|---:|---:|---|"]
for r in table:
    spread = r["moved"] / r["natural_spread"] if r["natural_spread"] else float("nan")
    lines.append(
        f"| {r['concept']} | {r['component']} | {100 * r['variance_share']:.1f}% | "
        f"{', '.join(r['corpus_toward'][:3])} / {', '.join(r['corpus_away'][:3])} | {r['follows']:.2f} | "
        f"{r['usable_lo']:+.1f} to {r['usable_hi']:+.1f} | {spread:.1f} | {r['kept']:.2f} | "
        f"{', '.join(r['rises'])} / {', '.join(r['falls'])} |")
(out / "sliders.md").write_text("\n".join(lines) + "\n")
(out / "sliders.json").write_text(json.dumps(dict(sliders=table, curves=curves), indent=1))
print("\n".join(lines))
