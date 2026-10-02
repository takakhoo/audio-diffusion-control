"""Trained sliders on the Hugging Face Hub.

    from audiosliders.hub import fetch
    folder = fetch("ace-step-1.5-xl-turbo/text")        # downloads once, then reads the local cache

Anywhere a slider directory is expected, `hf:<group>` works as well:

    python -m audiosliders.server --sliders hf:ace-step-1.5-xl-turbo/text
"""

from __future__ import annotations

from pathlib import Path

REPO = "takakhoo/audio-sliders"
GROUPS = {
    "ace-step-1.5-xl-turbo/text": "twelve named attributes, each trained from a prompt pair",
    "ace-step-1.5-xl-turbo/real-axes": "five axes found in real music, trained from a prompt pair made of the axis's tags",
    "ace-step-1.5-xl-turbo/real-axes-sets": "seven axes found in real music, trained with no text; use between -1 and +1",
    "ace-step-1.5-xl-turbo/measured-sets": "five measurements, trained with no text; use between -1 and +1",
    "ace-step-1.5-xl-turbo/graded": "six sliders trained with no text at graded positions; usable from -2 to +2",
}


def fetch(group: str, repo: str = REPO) -> Path:
    """Download one group of sliders and return the directory that holds its .safetensors files."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo, allow_patterns=[f"{group}/*.safetensors"])
    return Path(root) / group


def resolve(path: str | Path) -> Path:
    """A local directory, or `hf:<group>` for a group on the Hub."""
    text = str(path)
    return fetch(text[3:]) if text.startswith("hf:") else Path(text)
