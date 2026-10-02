"""Add beat-tracked tempo to evaluation runs that were made before the tracker was wired in.

    python experiments/beat_score.py runs/eval/main/lora_tempo runs/eval/ace/lora_*

Writes beat.jsonl next to rows.jsonl, in the same order; audiosliders.metrics merges it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from beat_this.inference import Audio2Beats

tracker = Audio2Beats(checkpoint_path="final0", device="cuda", dbn=False)
for run in map(Path, sys.argv[1:]):
    rows = [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines() if line]
    files = [run / f"p{r['prompt_index']:02d}_s{r['seed']:05d}_x{r['scale']:+.2f}.flac" for r in rows]
    if not all(f.exists() for f in files):
        print("no audio", run)
        continue
    out = []
    for f in files:
        audio, sr = sf.read(f)
        found, _ = tracker(audio.mean(1).astype(np.float32), sr)
        out.append(dict(file=f.name, beat_bpm=float(60 / np.median(np.diff(found))) if len(found) > 3 else float("nan")))
    (run / "beat.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n")
    print("scored", run, len(out))
