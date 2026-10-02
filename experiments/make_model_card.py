"""Write the Hugging Face model card for the published sliders from the committed results.

    python experiments/make_model_card.py --out runs/release/README.md

Every number in the card is read from results/, so the card cannot drift from the tables.
"""

import argparse
import json
from pathlib import Path

import yaml

from audiosliders.hub import REPO

SITE = "https://takakhoo.github.io/audio-diffusion-control"
CODE = "https://github.com/takakhoo/audio-diffusion-control"
RAW = "https://raw.githubusercontent.com/takakhoo/audio-diffusion-control/main"
TEXT = ["mood", "ensemble", "groove", "harmony", "melody", "tension", "brightness", "density", "energy", "tempo", "electronic", "vintage"]
MEASURED = ["energy", "harmony", "density", "ensemble", "quality"]
SAMPLES = [("text", "mood", "lora", [-2, 0, 2]), ("real-axes", "arousal", "lora", [-2, 0, 1]),
           ("real-axes", "jazz_electronic", "lora", [-2, 0, 2]), ("real-axes-sets", "arousal", "contrast", [-1, 0, 1]),
           ("measured-sets", "harmony", "contrast", [-1, 0, 1])]

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--out", required=True)
args = ap.parse_args()

spec = yaml.safe_load(Path("configs/sliders.yaml").read_text())
ace = {r["slider"]: r for r in json.loads(Path("results/ace/summary.json").read_text())["summary"] if r["method"] == "LoRA slider"}
v2 = {r["slider"]: r for r in json.loads(Path("results/ace_v2/summary.json").read_text())["summary"]
      if r["method"] == "Descriptor slider"}
axes = json.loads(Path("results/ace_v2/real_axes.json").read_text())
pair = {r["slider"]: r for r in axes if r["trained_from"].startswith("prompt pair")}
sets = {r["slider"]: r for r in axes if r["trained_from"].startswith("two sets")}
num = lambda v, d=2: "" if v is None else f"{v:.{d}f}"
ends = lambda n: f"{spec[n]['ends'][0]} → {spec[n]['ends'][1]}"
span = lambda lo, hi: f"{lo:+g} to {hi:+g}"

count = {25: "Twenty-five", 29: "Twenty-nine", 35: "Thirty-five"}
lines = [f"""---
license: mit
base_model: ACE-Step/acestep-v15-xl-turbo-diffusers
pipeline_tag: text-to-audio
tags:
- music
- text-to-music
- lora
- concept-sliders
- ace-step
- audio
---

# Audio Sliders for ACE-Step 1.5 XL turbo

{count.get(len(TEXT) + len(pair) + len(sets) + len(MEASURED) + 6, len(TEXT) + len(pair) + len(sets) + len(MEASURED) + 6)} sliders for generated music. Each one is a rank-4 LoRA on the frozen ACE-Step 1.5 XL turbo transformer whose
strength is a number you set at sampling time. The prompt and the seed stay fixed and the same piece moves along one
axis: sad to happy, solo to full ensemble, energetic to quiet and dreamy, electronic to jazz.

Every slider here was measured on 24 held-out prompts: does a property of the waveform follow the slider, does a second
embedding model agree, how much of the piece survives, and do two quality predictors still rate the result as music.
The numbers below come from that evaluation. Sliders that failed it were left out.
""",
         f"**[Live demo]({SITE}/)** · **[Code, results, and paper draft]({CODE})** · **[Ten-minute listening test]({SITE}/listen.html)**",
         "", f"![Four sliders sweeping across their range]({RAW}/results/demo/sliders.gif)", "",
         "## Listen", "",
         "One prompt (\"mellow jazz piano trio, brushed drums\") and one seed per slider; only the slider position changes. "
         "Clips are loudness-matched.", ""]
for group, name, method, xs in SAMPLES:
    lines += [f"**{spec[name].get('label', name.capitalize())}** (`{group}`): {ends(name)}", ""]
    for x in xs:
        url = f"https://huggingface.co/{REPO}/resolve/main/samples/{group}_{name}_{'m' if x < 0 else 'p'}{abs(x)}.m4a"
        lines.append(f"- {'unsteered' if x == 0 else f'position {x:+d}'}: <audio controls src=\"{url}\"></audio>")
    lines.append("")

lines += ["", "## What is in the repository", "",
          f"### `ace-step-1.5-xl-turbo/text`: {len(TEXT)} named attributes, trained from a prompt pair", "",
          "| Slider | Low → high | Waveform descriptor follows (ρ) | MuQ-MuLan agrees (ρ) | Usable positions | Piece kept | Enjoyment at the ends (6.95 unsteered) |",
          "|---|---|---:|---:|---|---:|---:|"]
for n in TEXT:
    r = ace[n]
    lines.append(f"| `{n}` | {ends(n)} | {num(r.get('rho')) or 'none assigned'} | {num(r.get('muq_rho'))} | "
                 f"{span(r['usable_lo'], r['usable_hi'])} | {num(r.get('usable_clap_keep'))} | {num(r.get('ce_at_ends'))} |")

lines += ["", "### `ace-step-1.5-xl-turbo/real-axes`: axes found in real music, trained from the axis's own tags", "",
          "The axes are independent components of MuQ-MuLan embeddings of 14,985 real recordings. Nobody chose them. The prompt "
          "pair for each slider is the set of tags at the two ends of its axis, and the output is scored by its projection on "
          "the axis, in standard deviations of real music.", "",
          "| Slider | Low → high | Follows the axis (ρ) | Ends ordered | Moved between -1 and +1 (std of real music) | Usable positions | Piece kept at ±1 |",
          "|---|---|---:|---:|---:|---|---:|"]
for n, r in pair.items():
    lines.append(f"| `{n}` | {ends(n)} | {num(r['rho'])} | {100 * r['ordered']:.0f}% | {num(r['moved_real_std'][0])} | "
                 f"{span(*r['usable'])} | {num(r['kept'])} |")

lines += ["", "### `ace-step-1.5-xl-turbo/real-axes-sets`: the same kind of axis, trained with no text", "",
          "Trained between the top and bottom 20% of the model's own clips along the axis (15,552 clips, 648 prompts). They "
          "move less than the prompt-pair versions, keep more of the piece, and hold quality level at both ends. "
          "**Use them between -1 and +1.**", "",
          "| Slider | Low → high | Follows the axis (ρ) | Ends ordered | Moved between -1 and +1 (std of real music) | Piece kept at ±1 |",
          "|---|---|---:|---:|---:|---:|"]
for n, r in sets.items():
    lines.append(f"| `{n}` | {ends(n)} | {num(r['rho'])} | {100 * r['ordered']:.0f}% | {num(r['moved_real_std'][0])} | {num(r['kept'])} |")

lines += ["", "### `ace-step-1.5-xl-turbo/measured-sets`: a measurement turned into a slider, with no text", "",
          "Clips sorted by a measurement, after removing what loudness, brightness, and predicted enjoyment explain. These are "
          "the selective sliders: selectivity is how far a slider moves its own measurement relative to the average bystander. "
          "**Use them between -1 and +1.**", "",
          "| Slider | Sorted by | Follows the measurement (ρ) | Ends ordered | Selectivity | Piece kept at ±1 |", "|---|---|---:|---:|---:|---:|"]
for n in MEASURED:
    r = v2[n]
    lines.append(f"| `{n}` | {r['measure'].replace('_', ' ')} | {num(r['rho'])} | {100 * r['consistent']:.0f}% | "
                 f"{num(r.get('selectivity'), 1)} | {num(r.get('usable_clap_keep'))} |")

graded = {r["slider"]: r for r in json.loads(Path("results/ace_g/summary.json").read_text())["summary"]}
graded_axes = {r["slider"]: r for r in json.loads(Path("results/ace_g/real_axes.json").read_text())}
lines += ["", "### `ace-step-1.5-xl-turbo/graded`: trained with no text at graded positions, usable from -2 to +2", "",
          "The set trainer above shows the slider only positions -1 and +1, and its sliders fall apart past ±1. These six "
          "were trained with every clip at its own position along the measurement or axis. They move slightly less inside "
          "±1 and keep working out to ±2 with no loss on either quality predictor.", "",
          "| Slider | Low → high | Follows it, -2 to +2 (ρ) | Ends ordered | Piece kept at ±2 | Enjoyment at the ends |",
          "|---|---|---:|---:|---:|---:|"]
for n in ("energy", "harmony", "arousal", "valence", "jazz_electronic", "piano_axis"):
    r = graded[n]
    rho = graded_axes[n]["rho_full"] if n in graded_axes else r.get("rho")
    ordered = graded_axes[n]["ordered"] if n in graded_axes else r.get("consistent")
    lines.append(f"| `{n}` | {ends(n)} | {num(rho)} | {100 * ordered:.0f}% | {num(r.get('usable_clap_keep'))} | {num(r.get('ce_at_ends'))} |")
lines += ["", "`axes/` holds the direction vectors of the real-music axes, so new clips can be scored against them.", "",
          "## Use", "", "```bash", f"pip install \"audiosliders[model,demo] @ git+{CODE}\" audiobox_aesthetics", "```", "",
          "A page where you type a prompt and drag the sliders:", "", "```bash",
          "python -m audiosliders.server --sliders hf:ace-step-1.5-xl-turbo/text --backbone ace-turbo", "```", "",
          "From Python:", "", "```python", "import soundfile as sf",
          "from audiosliders.backbone import load_backbone", "from audiosliders.hub import fetch",
          "from audiosliders.lora import SliderBank", "",
          "model = load_backbone(\"ace-turbo\")                    # downloads ACE-Step 1.5 XL turbo",
          "bank = SliderBank(model.dit)",
          "folder = fetch(\"ace-step-1.5-xl-turbo/real-axes\")",
          "bank.load(\"arousal\", folder / \"arousal.safetensors\")", "",
          "for position in (-2.0, 0.0, 1.0):",
          "    audio = model.generate([\"mellow jazz piano trio, brushed drums\"], [0], seconds=10,",
          "                           wrap=lambda p: bank.gated(p, {\"arousal\": position}))",
          "    sf.write(f\"arousal_{position:+.0f}.wav\", audio[0].T.cpu().numpy(), model.sample_rate)", "```", "",
          "Position 0 is the unchanged model. Several sliders can be loaded and set together; their updates add.", "",
          "## Limits", "",
          "- No listening study has been run yet. Quality rests on two learned predictors (Audiobox Aesthetics and SongEval) and "
          "meaning on signal descriptors and two embedding models.",
          "- Sliders leak. Mood and melody also brighten the clip, and several raise loudness. The leakage table is in the repository.",
          "- The set-trained sliders stop working past ±1. A set-trained tempo slider and a tag-sorted mood slider did not work and are not included.",
          "- Trained and tested on ten-second instrumental clips. Six of the text sliders were also checked at thirty seconds.",
          "- The files use this project's own LoRA layout and are loaded with `audiosliders.lora.SliderBank`.", "",
          "## License and credit", "",
          "MIT, the same as ACE-Step 1.5. The training objective for the prompt-pair sliders is Concept Sliders (Gandikota et al., "
          "ECCV 2024), and the idea of discovering axes instead of naming them comes from SliderSpace (Gandikota et al., ICCV 2025).", ""]
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines))
print(f"wrote {out}")
