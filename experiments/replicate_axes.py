"""Do axes found in one sample of real music come back in another?

    python experiments/replicate_axes.py --a runs/corpus/real_ace --b runs/corpus/real_large --emb muq \
        --vocab runs/reference/vocab_muq.npz --out results/discovery/replication

Fits principal and independent axes separately on two corpora that share no recording,
matches them one to one, and reports the cosine of every matched pair with the tags at the
ends of each. Splitting one corpus in halves (axis_stability) asks whether the axes are
stable under resampling; this asks whether they survive a different and larger sample.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from audiosliders.contrast import independent_directions, load_corpus, principal_directions
from audiosliders.tags import label_direction

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--emb", default="muq", choices=["clap", "muq"])
ap.add_argument("--vocab", required=True)
ap.add_argument("--max-vocal", type=float, default=0.6)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--out", required=True)
args = ap.parse_args()


def load(path):
    c = load_corpus(path, latents=False)
    vocal = np.array([r.get("vocal_score", -np.inf) for r in c["rows"]])
    keep = vocal <= np.quantile(vocal, args.max_vocal)
    return c[args.emb][keep], np.array([r["prompt_index"] for r in c["rows"]])[keep]


(ea, ga), (eb, gb) = load(args.a), load(args.b)
vocab = np.load(args.vocab)["text"]
report = dict(clips=dict(a=len(ea), b=len(eb)), methods={})
lines = [f"Corpus A: {len(ea):,} recordings. Corpus B: {len(eb):,} recordings, none shared with A.", ""]
for method, fit in (("independent components", independent_directions), ("principal components", principal_directions)):
    da, db = fit(ea, ga, args.n)[0], fit(eb, gb, args.n)[0]
    da, db = (d / np.linalg.norm(d, axis=1, keepdims=True) for d in (da, db))
    cos = da @ db.T
    rows, cols = linear_sum_assignment(-np.abs(cos))
    matched = []
    for i, j in zip(rows, cols):
        sign = np.sign(cos[i, j])
        toward, away = label_direction(da[i], vocab, k=3)
        toward_b, away_b = label_direction(sign * db[j], vocab, k=3)
        matched.append(dict(axis=int(i), partner=int(j), cosine=float(abs(cos[i, j])), toward=toward, away=away,
                            toward_b=toward_b, away_b=away_b))
    report["methods"][method] = dict(mean_cosine=float(np.mean([m["cosine"] for m in matched])), axes=matched)
    lines += [f"### {method}: mean matched cosine {report['methods'][method]['mean_cosine']:.2f}", "",
              "| Axis in A | Tags at its ends in A | Matched cosine | Tags at the ends of its partner in B |", "|---:|---|---:|---|"]
    for m in matched:
        lines.append(f"| {m['axis']} | {', '.join(m['toward'])} / {', '.join(m['away'])} | {m['cosine']:.2f} | "
                     f"{', '.join(m['toward_b'])} / {', '.join(m['away_b'])} |")
    lines.append("")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / f"{args.emb}.md").write_text("\n".join(lines))
(out / f"{args.emb}.json").write_text(json.dumps(report, indent=1))
print("\n".join(lines))
