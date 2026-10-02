"""Build the static demo in docs/ from evaluation runs that were saved with --save-audio.

    python experiments/export_demo.py --source ace=runs/eval/ace --source sao=runs/eval/main \
        --discovered ace=runs/eval/ace_pca:results/discovery/ace/sliders.json --prompts 0 3 7 12

Copies one seed per prompt as AAC and writes docs/demo/manifest.json, which the page
reads. Run on the machine that holds the evaluation output; needs ffmpeg.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import yaml

from audiosliders import metrics

MEASURES = dict(
    centroid_oct=("spectral centroid", "oct above A440"), centroid_hz=("spectral centroid", "Hz"),
    onset_rate=("onsets per second", "/s"), decay_s=("energy decay time", "s"), flux=("spectral flux", ""),
    pulse_bpm=("estimated tempo", "BPM"), beat_bpm=("tempo (beat tracker)", "BPM"), percussive_ratio=("percussive energy share", ""),
    flatness=("spectral flatness", ""), rolloff_oct=("95% rolloff", "oct above A440"), rolloff_hz=("95% rolloff", "Hz"),
    bass_ratio=("energy below 150 Hz", "dB"), side_ratio=("side / mid energy", "dB"),
    majorness=("major minus minor key fit", ""), harmonic_change=("harmonic change rate", ""),
    key_clarity=("key clarity", ""), dynamics_db=("level variation", "dB"), pc=("production complexity", "/10"),
    pulse_clarity=("pulse clarity", ""), ce=("content enjoyment", "/10"), pq=("production quality", "/10"),
)
METHODS = dict(
    lora=("Text slider", "A LoRA trained from a prompt pair, scaled by the slider position. Same cost as plain generation."),
    contrast=("Set-trained slider", "A LoRA trained with no text, between two sets of the model's own clips: the top and bottom 20% along a measurement or along an axis found in real music. Usable from -1 to +1."),
    discovered=("Discovered axis", "A LoRA trained along a principal direction of the model's own output for this concept. Nobody named it in advance; the labels come from the tags it moves."),
    internal=("The model's own axis", "A principal direction of the transformer's own activations across seeds of the same prompt, added back at sampling time. No text, no embedding model, and no training chose it; the labels come from the tags it moves."),
    guidance=("Prompt-pair guidance", "The text slider's training target applied directly at every step. Two extra forward passes per step."),
    embed=("Prompt interpolation", "The text conditioning is moved toward the positive or negative prompt. No training."),
    dsp=("Signal processing", "The unsteered clip run through a conventional effect. Shown where one exists."),
)
MODELS = dict(
    ace=("ACE-Step 1.5 XL turbo", "48 kHz, 8 sampling steps."),
    sao=("Stable Audio Open 1.0", "44.1 kHz, 50 sampling steps."),
)
CONTRAST_MEASURE = dict(quality="ce", production="pq", ensemble="pc", harmony="harmonic_change", groove="pulse_clarity")

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--source", action="append", default=[], metavar="MODEL=DIR[:METHOD,METHOD]",
                help="evaluation directory for a model, optionally limited to some methods; repeatable per model")
ap.add_argument("--max-scale", action="append", default=[], metavar="METHOD=VALUE",
                help="publish only positions up to this magnitude for a method")
ap.add_argument("--discovered", action="append", default=[], metavar="MODEL=EVALDIR:SLIDERS_JSON")
ap.add_argument("--axes", action="append", default=[], metavar="MODEL=EVALDIR:AXES_JSON",
                help="unnamed axes described by experiments/report_axes.py")
ap.add_argument("--out", default="docs")
ap.add_argument("--prompts", type=int, nargs="+", default=None, help="eval prompt indices to publish")
ap.add_argument("--discovered-seeds", type=int, default=3)
ap.add_argument("--sliders", nargs="+", default=None)
ap.add_argument("--methods", nargs="+", default=["lora", "contrast", "guidance", "embed", "dsp"])
ap.add_argument("--summary", default=None, help="JSON written by experiments/report.py")
ap.add_argument("--bitrate", default="96k")
args = ap.parse_args()

spec = yaml.safe_load(Path("configs/sliders.yaml").read_text())
limit = {m: float(v) for m, _, v in (item.partition("=") for item in args.max_scale)}
out = Path(args.out)
manifest = dict(models={}, methods={}, clips={})


def publish(model, method, name, run, rows, pid, seed, prompt_slot):
    """Convert one trajectory to AAC and return its clip list."""
    mine = sorted((r for r in rows if r["prompt_index"] == pid and r["seed"] == seed
                   and abs(r["scale"]) <= limit.get(method, float("inf"))), key=lambda r: r["scale"])
    clips = []
    for r in mine:
        src = run / f"p{pid:02d}_s{seed:05d}_x{r['scale']:+.2f}.flac"
        rel = Path("audio") / model / method / name / f"p{prompt_slot:02d}_x{r['scale']:+.2f}.m4a"
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            # Every clip is brought to the same loudness so a slider is not judged by how loud it gets.
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                            "-ar", "44100", "-c:a", "aac", "-b:a", args.bitrate, "-movflags", "+faststart", str(dst)],
                           check=True)
        clips.append(dict(x=r["scale"], f=str(rel), keep=r.get("clap_keep"), chroma=r.get("chroma_sim"),
                          dir=r.get("clap_dir"), ce=r.get("ce"), pq=r.get("pq"), row=r))
    return clips


def entry_for(model, name, label, low, high, measure):
    sliders = manifest["models"][model]["sliders"]
    return sliders.setdefault(name, dict(
        label=label, low=low, high=high, measure=measure, methods=[],
        measure_label=MEASURES[measure][0] if measure else None, unit=MEASURES[measure][1] if measure else None))


for source in args.source:
    model, _, root = source.partition("=")
    root, _, only = root.partition(":")
    root = Path(root)
    manifest["models"].setdefault(model, dict(label=MODELS[model][0], note=MODELS[model][1], sliders={}, prompts=[]))
    prompts = manifest["models"][model]["prompts"]
    names = args.sliders or list(dict.fromkeys(list(spec) + ["production"]))
    for name in names:
        for method in (only.split(",") if only else args.methods):
            run = root / f"{method}_{name}"
            if not (run / "rows.jsonl").exists():
                continue
            rows = metrics.load_rows(run)
            s = spec.get(name, {})
            measure = s.get("display") or s.get("measure") or CONTRAST_MEASURE.get(name)
            ends = s.get("ends") or ["rougher production", "cleaner production"]
            entry = entry_for(model, name, name.replace("_axis", "").replace("_", " / ").capitalize(), ends[0], ends[1], measure)
            if method not in entry["methods"]:
                entry["methods"].append(method)
            manifest["methods"][method] = dict(label=METHODS[method][0], note=METHODS[method][1])
            for pid in args.prompts or sorted({r["prompt_index"] for r in rows}):
                mine = [r for r in rows if r["prompt_index"] == pid]
                if not mine:
                    continue
                if mine[0]["prompt"] not in prompts:
                    prompts.append(mine[0]["prompt"])
                slot = prompts.index(mine[0]["prompt"])
                clips = publish(model, method, name, run, rows, pid, min(r["seed"] for r in mine), slot)
                for c in clips:
                    c["v"] = c.pop("row").get(measure) if measure else None
                manifest["clips"][f"{model}/{method}/{name}/{slot}"] = clips

for item in args.discovered:
    model, _, rest = item.partition("=")
    root, _, table = rest.partition(":")
    info = {(s["concept"], s["component"]): s for s in json.loads(Path(table).read_text())["sliders"]}
    concepts = yaml.safe_load(Path("configs/prompts.yaml").read_text())["concepts"]
    prompts = manifest["models"][model]["prompts"]
    manifest["methods"]["discovered"] = dict(label=METHODS["discovered"][0], note=METHODS["discovered"][1])
    for run in sorted(Path(root).iterdir()):
        m = re.fullmatch(r"c(\d+)_pc(\d+)", run.name)
        if not m or not (run / "rows.jsonl").exists():
            continue
        concept, i = concepts[int(m.group(1))], int(m.group(2))
        s = info.get((concept, i))
        if s is None:
            continue
        rows = metrics.load_rows(run)
        low = ", ".join(s["falls"][:2]) or ", ".join(s["corpus_away"][:2])
        high = ", ".join(s["rises"][:2]) or ", ".join(s["corpus_toward"][:2])
        name = f"{concept.replace(' ', '-')}-axis-{i + 1}"
        entry = entry_for(model, name, f"{concept}: axis {i + 1}", low, high, None)
        entry["methods"].append("discovered")
        for k, seed in enumerate(sorted({r["seed"] for r in rows})[: args.discovered_seeds]):
            label = f"{concept} (seed {k + 1})"
            if label not in prompts:
                prompts.append(label)
            slot = prompts.index(label)
            clips = publish(model, "discovered", name, run, rows, 0, seed, slot)
            for c in clips:
                c.pop("row")
                c["v"] = None
            manifest["clips"][f"{model}/discovered/{name}/{slot}"] = clips

for item in args.axes:
    model, _, rest = item.partition("=")
    root, _, table = rest.partition(":")
    prompts = manifest["models"][model]["prompts"]
    manifest["methods"]["internal"] = dict(label=METHODS["internal"][0], note=METHODS["internal"][1])
    for k, a in enumerate(json.loads(Path(table).read_text())):
        run = Path(root) / a["name"]
        rows = metrics.load_rows(run)
        name = f"own-axis-{k + 1}"
        entry = entry_for(model, name, f"Own axis {k + 1}", ", ".join(a["falls"][:2]), ", ".join(a["rises"][:2]), "ce")
        entry["methods"].append("internal")
        for pid in args.prompts or sorted({r["prompt_index"] for r in rows}):
            mine = [r for r in rows if r["prompt_index"] == pid]
            if not mine:
                continue
            if mine[0]["prompt"] not in prompts:
                prompts.append(mine[0]["prompt"])
            slot = prompts.index(mine[0]["prompt"])
            clips = publish(model, "internal", name, run, rows, pid, min(r["seed"] for r in mine), slot)
            for c in clips:
                c["v"] = c.pop("row").get("ce")
            manifest["clips"][f"{model}/internal/{name}/{slot}"] = clips

if args.summary:
    manifest.update(json.loads(Path(args.summary).read_text()))
(out / "demo").mkdir(parents=True, exist_ok=True)
(out / "demo" / "manifest.json").write_text(json.dumps(manifest))
print(f"{len(manifest['clips'])} slider tracks, {sum(len(c) for c in manifest['clips'].values())} clips")
