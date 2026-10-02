"""Live demo server: type a prompt, set any combination of sliders, hear the result.

    python -m audiosliders.server --sliders runs/sliders/ace --backbone ace-turbo --port 7860

Serves the static page in docs/ and two endpoints the page uses when it finds them:
GET /api/sliders and POST /api/generate.
"""

from __future__ import annotations

import argparse
import io
import threading
import time
from collections import OrderedDict
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .backbone import load_backbone
from .descriptors import describe
from .lora import SliderBank


class Request(BaseModel):
    prompt: str = Field(min_length=1, max_length=400)
    seed: int = 0
    seconds: float = Field(default=10.0, ge=2.0, le=30.0)
    steps: int | None = Field(default=None, ge=4, le=100)
    guidance: float | None = Field(default=None, ge=1.0, le=12.0)
    sliders: dict[str, float] = {}
    start: float = Field(default=1.0, ge=0.0, le=1.0)


def build(slider_dir: Path, static_dir: Path, backbone: str = "ace-turbo", device: str = "cuda") -> FastAPI:
    model = load_backbone(backbone, device)
    bank = SliderBank(model.dit)
    meta = {f.stem: bank.load(f.stem, f) for f in sorted(slider_dir.glob("*.safetensors"))}
    lock = threading.Lock()
    clips: OrderedDict[str, bytes] = OrderedDict()
    app = FastAPI(title="Audio Sliders")

    @app.get("/api/sliders")
    def sliders():
        return [dict(name=k, **{f: v.get(f) for f in ("positive", "negative", "by")}) for k, v in meta.items()]

    @app.post("/api/generate")
    def generate(req: Request):
        unknown = set(req.sliders) - set(meta)
        if unknown:
            raise HTTPException(400, f"unknown sliders: {sorted(unknown)}")
        scales = {k: max(-4.0, min(4.0, v)) for k, v in req.sliders.items() if v}
        t0 = time.time()
        with lock, torch.no_grad():
            wrap = (lambda p: bank.gated(p, scales, start=req.start)) if scales else None
            audio = model.generate([req.prompt], [req.seed], seconds=req.seconds, steps=req.steps,
                                   guidance=req.guidance, wrap=wrap)[0].cpu().numpy()
        buf = io.BytesIO()
        sf.write(buf, audio.T, model.sample_rate, format="WAV", subtype="PCM_16")
        key = f"{len(clips)}-{int(t0 * 1000)}"
        clips[key] = buf.getvalue()
        while len(clips) > 64:
            clips.popitem(last=False)
        keep = ("centroid_hz", "onset_rate", "percussive_ratio", "decay_s", "bass_ratio", "side_ratio", "rms_db")
        desc = describe(audio, model.sample_rate)
        return dict(audio=f"api/audio/{key}.wav", seconds=round(time.time() - t0, 2),
                    descriptors={k: desc[k] for k in keep})

    @app.get("/api/audio/{key}.wav")
    def audio(key: str):
        if key not in clips:
            raise HTTPException(404, "clip expired")
        return Response(clips[key], media_type="audio/wav")

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sliders", required=True, help="directory of .safetensors sliders")
    ap.add_argument("--static", default="docs")
    ap.add_argument("--backbone", default="ace-turbo")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    uvicorn.run(build(Path(args.sliders), Path(args.static), args.backbone), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
