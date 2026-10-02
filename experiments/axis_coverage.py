"""Which axes of real music does a model's output span?

    python experiments/axis_coverage.py --real runs/corpus/real_ace --directions runs/discovery/real_muq_ica \
        --generated ace=runs/corpus/ace_large --emb muq --max-vocal 0.6 --out results/coverage/muq_ica

Takes axes discovered in real recordings (experiments/discover.py) and projects a model's
generated corpus on them. For every axis it reports the spread of the model's output as a
fraction of the spread in real music, over all prompts and for a fixed prompt, and where
the average generated clip sits. An axis with low spread is one the model does not explore
on its own, which is where a slider has the most to add and the least to work with.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from audiosliders.contrast import axis_coverage, load_corpus, subspace_overlap

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--real", required=True)
ap.add_argument("--directions", required=True, help="directory written by experiments/discover.py")
ap.add_argument("--generated", action="append", required=True, metavar="NAME=CORPUS")
ap.add_argument("--emb", default="clap", choices=["clap", "muq"])
ap.add_argument("--max-vocal", type=float, default=None)
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--out", required=True)
args = ap.parse_args()

real = load_corpus(args.real, latents=False)
emb, rows = real[args.emb], real["rows"]
if args.max_vocal is not None:
    vocal = np.array([r.get("vocal_score", -np.inf) for r in rows])
    keep = vocal <= np.quantile(vocal, args.max_vocal)
    emb, rows = emb[keep], [r for r, k in zip(rows, keep) if k]
groups = np.array([r["prompt_index"] for r in rows])
directions = np.load(Path(args.directions) / "directions.npy")[: args.n]
labels = next(v for k, v in json.loads((Path(args.directions) / "pca.json").read_text()).items()
              if not k.startswith("_"))["components"]

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
report = dict(real_clips=len(rows), axes=[dict(axis=i, toward=c.get("toward", []), away=c.get("away", []))
                                           for i, c in enumerate(labels[: len(directions)])], models={})
for item in args.generated:
    name, _, path = item.partition("=")
    gen = load_corpus(path, latents=False)
    g_groups = np.array([r["prompt_index"] for r in gen["rows"]])
    cover = axis_coverage(emb, groups, gen[args.emb], g_groups, directions)
    report["models"][name] = dict(
        clips=len(gen["rows"]), prompts=int(len(np.unique(g_groups))),
        overlap={k: subspace_overlap(emb, gen[args.emb], k) for k in (4, 8, 16)},
        **{k: v.tolist() for k, v in cover.items()})

names = list(report["models"])
head = "| Axis | Toward | Away | " + " | ".join(f"{n}: all prompts | {n}: one prompt | {n}: offset" for n in names) + " |"
lines = [head, "|---:|---|---|" + "---:|" * (3 * len(names))]
for i, a in enumerate(report["axes"]):
    cells = " | ".join(f"{report['models'][n]['total'][i]:.2f} | {report['models'][n]['within'][i]:.2f} | "
                       f"{report['models'][n]['offset'][i]:+.2f}" for n in names)
    lines.append(f"| {i} | {', '.join(a['toward'][:3])} | {', '.join(a['away'][:3])} | {cells} |")
lines += ["", "Share of the leading real-music subspace that the model's own leading subspace contains:", "",
          "| Model | Clips | Prompts | Top 4 | Top 8 | Top 16 |", "|---|---:|---:|---:|---:|---:|"]
for n in names:
    m = report["models"][n]
    lines.append(f"| {n} | {m['clips']:,} | {m['prompts']} | {m['overlap'][4]:.2f} | {m['overlap'][8]:.2f} | {m['overlap'][16]:.2f} |")
(out / "coverage.md").write_text("\n".join(lines) + "\n")
(out / "coverage.json").write_text(json.dumps(report, indent=1))
print("\n".join(lines))
