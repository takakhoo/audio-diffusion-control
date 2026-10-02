"""Summarise answer files downloaded from docs/listen.html.

    python experiments/listening_results.py answers/*.json --out results/listening

A session is dropped if it rated the hidden identical pair below 4 on "same piece". For the
direction questions the script reports, per slider and method, how often listeners picked
the clip the slider was pushed toward, with an exact binomial interval (chance is 50%). For
the edited-clip questions it reports the mean "same piece" and "finished music" ratings.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest


def interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson interval for k successes in n trials."""
    a = (1 - level) / 2
    return (float(beta.ppf(a, k, n - k + 1)) if k else 0.0, float(beta.ppf(1 - a, k + 1, n - k)) if k < n else 1.0)


def summarise(sessions: list[dict]) -> dict:
    kept = [s for s in sessions if all(a["same"] >= 4 for a in s["answers"] if a.get("check"))]
    direction, same = defaultdict(list), defaultdict(list)
    for s in kept:
        for a in s["answers"]:
            if a["task"] == "direction":
                direction[(a["slider"], a["method"])].append(a["choice"] == a["expected"])
            elif not a.get("check"):
                same[(a["slider"], a["method"], a["xB"])].append((a["same"], a["quality"]))
    rows = []
    for (slider, method), hits in sorted(direction.items()):
        k, n = int(sum(hits)), len(hits)
        lo, hi = interval(k, n)
        rows.append(dict(slider=slider, method=method, n=n, accuracy=k / n, low=lo, high=hi,
                         p=float(binomtest(k, n, 0.5, alternative="greater").pvalue)))
    edits = [dict(slider=sl, method=m, position=x, n=len(v), same=float(np.mean([a for a, _ in v])),
                  quality=float(np.mean([b for _, b in v]))) for (sl, m, x), v in sorted(same.items())]
    hits = [h for v in direction.values() for h in v]
    return dict(sessions=len(sessions), kept=len(kept), direction=rows, edits=edits,
                overall=dict(n=len(hits), accuracy=float(np.mean(hits)) if hits else float("nan")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = summarise([json.loads(Path(f).read_text()) for f in args.files])
    lines = [f"{report['kept']} of {report['sessions']} sessions kept. Direction heard as expected in "
             f"{100 * report['overall']['accuracy']:.0f}% of {report['overall']['n']} answers.", "",
             "| Slider | Made by | Answers | Picked the expected clip | 95% interval | p against chance |", "|---|---|---:|---:|---|---:|"]
    for r in report["direction"]:
        lines.append(f"| {r['slider']} | {r['method']} | {r['n']} | {100 * r['accuracy']:.0f}% | "
                     f"{100 * r['low']:.0f} to {100 * r['high']:.0f}% | {r['p']:.3f} |")
    lines += ["", "| Slider | Made by | Position | Answers | Same piece (1 to 5) | Finished music (1 to 5) |", "|---|---|---:|---:|---:|---:|"]
    for r in report["edits"]:
        lines.append(f"| {r['slider']} | {r['method']} | {r['position']:+g} | {r['n']} | {r['same']:.2f} | {r['quality']:.2f} |")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "listening.md").write_text("\n".join(lines) + "\n")
    (out / "listening.json").write_text(json.dumps(report, indent=1))
    print("\n".join(lines))
