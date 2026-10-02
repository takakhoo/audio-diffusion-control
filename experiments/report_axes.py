"""Say what a set of unnamed axes does, from evaluation runs alone.

    python experiments/report_axes.py --eval runs/eval/internal_ace --vocab runs/reference/vocab.npz \
        --out results/discovery/internal

Works for any steering method that has no prompt pair or descriptor to be judged against.
For every run: the tags that rise and fall between the two ends, how consistently the
axis moves different prompts in the same direction, the descriptors it moves most, how
much of the piece survives, and predicted enjoyment.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from audiosliders import metrics as M
from audiosliders.tags import tag_shift

KEYS = ["beat_bpm", "centroid_oct", "bass_ratio", "flatness", "flux", "rms_db", "onset_rate", "pulse_clarity",
        "percussive_ratio", "decay_s", "side_ratio", "majorness", "key_clarity", "harmonic_change", "dynamics_db", "pc"]


def consistency(shifts: np.ndarray) -> float:
    """Mean cosine between each trajectory's embedding shift and the mean shift of the others."""
    total = shifts.sum(0)
    cos = []
    for s in shifts:
        rest = total - s
        cos.append(float(s @ rest / (np.linalg.norm(s) * np.linalg.norm(rest) + 1e-12)))
    return float(np.mean(cos))


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--eval", required=True)
ap.add_argument("--vocab", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--match", default="", help="only run directories whose name contains this")
args = ap.parse_args()

text = np.load(args.vocab)["text"]
table = []
for run in sorted(Path(args.eval).iterdir()):
    if args.match not in run.name or not (run / "rows.jsonl").exists():
        continue
    rows = M.load_rows(run)
    emb = np.load(run / "clap.npy")
    scales = sorted({r["scale"] for r in rows})
    lo, hi = M.usable_span(rows)
    if not lo < 0 < hi:
        lo, hi = max(s for s in scales if s < 0), min(s for s in scales if s > 0)
    at = lambda x: np.array([r["scale"] == x for r in rows])
    order = np.lexsort(([r["seed"] for r in rows], [r["prompt_index"] for r in rows]))
    pick = lambda x: emb[[i for i in order if rows[i]["scale"] == x]]
    up, down = tag_shift(emb[at(hi)], emb[at(lo)], text, k=4)
    present = [k for k in KEYS if all(k in r for r in rows)]
    leak = M.leakage(rows, present)
    top = sorted(leak, key=lambda k: -abs(leak[k]))[:3]
    table.append(dict(
        name=run.name, usable_lo=lo, usable_hi=hi, consistency=consistency(pick(hi) - pick(lo)),
        rises=[t for t, _ in up], falls=[t for t, _ in down],
        descriptors={k: dict(slope=leak[k], rho=M.monotonicity(rows, k)["rho"]) for k in top},
        kept=float(np.mean([r["clap_keep"] for r in rows if r["scale"] in (lo, hi)])),
        ce_zero=float(np.mean([r["ce"] for r in rows if r["scale"] == 0])),
        ce_ends=float(np.mean([r["ce"] for r in rows if r["scale"] in (scales[0], scales[-1])])),
    ))

lines = ["| Axis | Tags that rise | Tags that fall | Consistency | Descriptors moved most (std per unit, ρ) | Usable span | "
         "Piece kept | Enjoyment at 0 / at ends |", "|---|---|---|---:|---|---|---:|---:|"]
for r in table:
    desc = ", ".join(f"{k} {v['slope']:+.2f} ({v['rho']:+.2f})" for k, v in r["descriptors"].items())
    lines.append(f"| {r['name']} | {', '.join(r['rises'])} | {', '.join(r['falls'])} | {r['consistency']:.2f} | {desc} | "
                 f"{r['usable_lo']:+.1f} to {r['usable_hi']:+.1f} | {r['kept']:.2f} | {r['ce_zero']:.2f} / {r['ce_ends']:.2f} |")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "axes.md").write_text("\n".join(lines) + "\n")
(out / "axes.json").write_text(json.dumps(table, indent=1))
print("\n".join(lines))
