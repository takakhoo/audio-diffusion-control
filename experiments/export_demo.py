"""Build the static demo in docs/ from evaluation runs that were saved with --save-audio.

Copies one seed per prompt as MP3 and writes docs/demo/manifest.json, which the page
reads. Run on the machine that holds the evaluation output; needs ffmpeg.
"""

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from audiosliders import metrics

MEASURES = dict(
    centroid_oct=("spectral centroid", "oct above A440"), centroid_hz=("spectral centroid", "Hz"), onset_rate=("onsets per second", "/s"),
    decay_s=("energy decay time", "s"), flux=("spectral flux", ""), pulse_bpm=("estimated tempo", "BPM"),
    percussive_ratio=("percussive energy share", ""), flatness=("spectral flatness", ""),
    rolloff_oct=("95% rolloff", "oct above A440"), rolloff_hz=("95% rolloff", "Hz"), bass_ratio=("energy below 150 Hz", "dB"),
    side_ratio=("side / mid energy", "dB"), majorness=("major minus minor key fit", ""),
)
METHODS = dict(
    lora=("LoRA slider", "The trained slider: one low-rank update scaled by the slider position. Same cost as plain generation."),
    contrast=("Descriptor slider", "A slider trained without text, from the model's own clips sorted by the measured descriptor."),
    guidance=("Prompt-pair guidance", "The slider's training target applied directly at every step. Two extra forward passes per step."),
    embed=("Prompt interpolation", "The text conditioning is moved toward the positive or negative prompt. No training."),
    dsp=("Signal processing", "The unsteered clip run through a conventional effect. Shown where one exists."),
)

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--eval", required=True, help="directory holding <method>_<slider>/ runs")
ap.add_argument("--out", default="docs")
ap.add_argument("--prompts", type=int, nargs="+", default=None, help="prompt indices to publish")
ap.add_argument("--sliders", nargs="+", default=None)
ap.add_argument("--methods", nargs="+", default=list(METHODS))
ap.add_argument("--summary", default=None, help="JSON written by experiments/report.py")
ap.add_argument("--bitrate", default="128k")
args = ap.parse_args()

spec = yaml.safe_load(Path("configs/sliders.yaml").read_text())
root, out = Path(args.eval), Path(args.out)
manifest = dict(model="Stable Audio Open 1.0", methods={}, sliders={}, prompts=[], clips={}, scales=[])
prompt_ids: list[int] = []
for name in args.sliders or list(spec):
    for method in args.methods:
        run = root / f"{method}_{name}"
        if not (run / "rows.jsonl").exists():
            continue
        rows = metrics.load_rows(run)
        ids = args.prompts or sorted({r["prompt_index"] for r in rows})
        s = spec[name]
        measure = s.get("display") or s.get("measure")
        entry = manifest["sliders"].setdefault(name, dict(
            label=name.capitalize(), low=s["ends"][0], high=s["ends"][1], measure=measure, methods=[],
            measure_label=MEASURES[measure][0] if measure else None, unit=MEASURES[measure][1] if measure else None))
        entry["methods"].append(method)
        manifest["methods"][method] = dict(label=METHODS[method][0], note=METHODS[method][1])
        for pid in ids:
            mine = [r for r in rows if r["prompt_index"] == pid]
            if not mine:
                continue
            seed = min(r["seed"] for r in mine)
            mine = sorted((r for r in mine if r["seed"] == seed), key=lambda r: r["scale"])
            if pid not in prompt_ids:
                prompt_ids.append(pid)
                manifest["prompts"].append(mine[0]["prompt"])
            clips = []
            for r in mine:
                src = run / f"p{pid:02d}_s{seed:05d}_x{r['scale']:+.2f}.flac"
                rel = Path("audio") / method / name / f"p{pid:02d}_x{r['scale']:+.2f}.mp3"
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-b:a", args.bitrate, str(dst)],
                                   check=True)
                clips.append(dict(x=r["scale"], f=str(rel), v=r.get(measure) if measure else None,
                                  keep=r.get("clap_keep"), chroma=r.get("chroma_sim"), dir=r.get("clap_dir")))
            manifest["clips"][f"{method}/{name}/{prompt_ids.index(pid)}"] = clips
            manifest["scales"] = [c["x"] for c in clips]
if args.summary:
    manifest.update(json.loads(Path(args.summary).read_text()))
(out / "demo").mkdir(parents=True, exist_ok=True)
(out / "demo" / "manifest.json").write_text(json.dumps(manifest))
print(f"{len(manifest['clips'])} slider tracks, {sum(len(c) for c in manifest['clips'].values())} clips")
