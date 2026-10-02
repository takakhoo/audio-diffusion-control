"""Score sliders trained along axes of real music, in units of real music.

    python experiments/report_real_axes.py --eval runs/eval/ace_v2 --coverage results/coverage \
        --vocab runs/reference/vocab.npz --out results/ace_v2

For every slider with an `axis` entry in configs/sliders.yaml, trained either way: how reliably the
output's projection on that direction follows the slider, and how far it moves, expressed
in standard deviations of real recordings along the axis and in standard deviations of
what seeds of one prompt reach on their own.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from audiosliders import metrics as M
from audiosliders.tags import tag_shift

COVERAGE = dict(real_muq_ica="muq_ica", real_muq_pca="muq_pca", real_ica="clap_ica", real_pca="clap_pca")
METHODS = dict(contrast="two sets of the model's clips", lora="prompt pair from the axis's tags")
KEYS = ["beat_bpm", "centroid_oct", "bass_ratio", "flatness", "flux", "rms_db", "onset_rate", "pulse_clarity",
        "percussive_ratio", "decay_s", "side_ratio", "majorness", "key_clarity", "harmonic_change", "dynamics_db", "pc"]

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--eval", required=True)
ap.add_argument("--coverage", required=True)
ap.add_argument("--vocab", required=True)
ap.add_argument("--model", default="ace")
ap.add_argument("--out", required=True)
args = ap.parse_args()

text = np.load(args.vocab)["text"]
spec = yaml.safe_load(Path("configs/sliders.yaml").read_text())
table = []
for run in sorted(Path(args.eval).iterdir()):
    method, _, name = run.name.partition("_")
    declared = spec.get(name, {}).get("axis")
    if not (run / "rows.jsonl").exists() or not declared or method not in METHODS:
        continue
    source, index = Path(declared["source"]), int(declared["index"])
    cov = json.loads((Path(args.coverage) / COVERAGE[source.parent.name] / "coverage.json").read_text())
    axis, model = cov["axes"][index], cov["models"][args.model]
    rows, emb = M.load_rows(run), np.load(run / "clap.npy")
    if declared["emb"] == "muq":
        key = "muq_axis" if "muq_axis" in rows[0] else "muq_dir"
        if key not in rows[0]:
            continue
        proj = [r[key] for r in rows]
    else:
        proj = emb @ np.load(source)[index]
    for r, v in zip(rows, proj):
        r["axis"] = float(v)
    mono, resp = M.monotonicity(rows, "axis"), M.response(rows, "axis")
    scales = list(resp["scales"])
    span = lambda x: float(resp["mean"][scales.index(x)] - resp["mean"][scales.index(-x)])
    at = lambda x: np.array([r["scale"] == x for r in rows])
    up, down = tag_shift(emb[at(1.0)], emb[at(-1.0)], text, k=4)
    leak = M.leakage(rows, [k for k in KEYS if all(k in r for r in rows)])
    top = sorted(leak, key=lambda k: -abs(leak[k]))[:3]
    ce = {x: float(np.mean([r["ce"] for r in rows if r["scale"] == x])) for x in (scales[0], 0.0, scales[-1])}
    table.append(dict(
        slider=name, trained_from=METHODS[method], embedding=declared["emb"], axis=index, toward=axis["toward"][:3], away=axis["away"][:3],
        rho=mono["rho"], rho_ci=mono["rho_ci"], ordered=mono["consistent"],
        top=scales[-1], moved_real_std=[span(x) / model["real_std"][index] for x in (1.0, scales[-1])],
        moved_seed_std=[span(x) / model["generated_within_std"][index] for x in (1.0, scales[-1])],
        coverage_all_prompts=model["total"][index], coverage_one_prompt=model["within"][index],
        kept=float(np.mean([r["clap_keep"] for r in rows if abs(r["scale"]) == 1.0])), ce=list(ce.values()),
        rises=[t for t, _ in up], falls=[t for t, _ in down],
        descriptors={k: leak[k] for k in top}, n=len({(r["prompt_index"], r["seed"]) for r in rows}),
    ))

lines = ["| Slider | Trained from | Axis from real music (toward / away) | ρ with the axis | Ends ordered | "
         "Moved at ±1 (std of real music) | Moved at the ends | Same, in std of one prompt's seeds | Piece kept at ±1 | "
         "Enjoyment at low end / 0 / high end | Tags that rise / fall | Descriptors moved most (std per unit) |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
for r in sorted(table, key=lambda r: (r["slider"], r["trained_from"])):
    lines.append(
        f"| {r['slider']} | {r['trained_from']} | {', '.join(r['toward'])} / {', '.join(r['away'])} | "
        f"{r['rho']:.2f} ± {r['rho_ci']:.2f} | {100 * r['ordered']:.0f}% | {r['moved_real_std'][0]:.2f} | "
        f"{r['moved_real_std'][1]:.2f} (±{r['top']:g}) | {r['moved_seed_std'][1]:.2f} | {r['kept']:.2f} | "
        f"{' / '.join(f'{v:.2f}' for v in r['ce'])} | {', '.join(r['rises'])} / {', '.join(r['falls'])} | "
        f"{', '.join(f'{k} {v:+.2f}' for k, v in r['descriptors'].items())} |")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "real_axes.md").write_text("\n".join(lines) + "\n")
(out / "real_axes.json").write_text(json.dumps(table, indent=1))
print("\n".join(lines))
