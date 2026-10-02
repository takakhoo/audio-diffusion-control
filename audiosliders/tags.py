"""A fixed vocabulary of musical tags, scored against clips in CLAP space.

This turns an embedding, or a direction between embeddings, into words: which
instruments, genres, moods, and playing characteristics a clip leans toward, and which
of them a slider or a discovered direction moves.
"""

from __future__ import annotations

import numpy as np

VOCAB: dict[str, list[str]] = {
    "instrument": [
        "piano", "acoustic guitar", "electric guitar", "bass guitar", "drums", "hand percussion", "strings",
        "violin", "cello", "brass", "trumpet", "saxophone", "flute", "synthesizer", "organ", "electric piano",
        "choir", "vocals", "harp", "bells",
    ],
    "genre": [
        "jazz", "classical", "rock", "electronic dance", "hip hop", "ambient", "folk", "funk", "blues", "reggae",
        "metal", "pop", "latin", "country", "soul", "techno", "orchestral film score", "lo-fi",
    ],
    "mood": [
        "happy", "sad", "calm", "energetic", "dark", "uplifting", "tense", "romantic", "aggressive", "dreamy",
        "melancholic", "playful", "epic", "mysterious",
    ],
    "character": [
        "fast", "slow", "sparse", "dense", "simple", "complex", "acoustic", "electronic", "live", "studio",
        "clean", "distorted", "dry", "reverberant", "rhythmic", "melodic", "improvised", "repetitive",
        "major key", "minor key", "swung", "straight", "legato", "staccato", "loud", "quiet", "bright", "dark-toned",
    ],
}
TEMPLATES = {"instrument": "music featuring {}", "genre": "{} music", "mood": "{} music", "character": "{} music"}


def vocabulary() -> list[tuple[str, str, str]]:
    """(group, tag, text to embed) for every tag."""
    return [(g, t, TEMPLATES[g].format(t)) for g, tags in VOCAB.items() for t in tags]


def scores(audio_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity of every clip to every tag, shape (clips, tags)."""
    return audio_emb @ text_emb.T


def label_direction(direction: np.ndarray, text_emb: np.ndarray, k: int = 4) -> tuple[list[str], list[str]]:
    """The tags most aligned with a direction, and those most opposed.

    Each tag is compared after removing the mean of its own group, so the answer says which
    instrument (or genre, or mood) the direction favours among the alternatives.
    """
    vocab = vocabulary()
    centered = text_emb.astype(np.float64).copy()
    groups = np.array([g for g, _, _ in vocab])
    for g in np.unique(groups):
        centered[groups == g] -= centered[groups == g].mean(0)
    centered /= np.linalg.norm(centered, axis=1, keepdims=True)
    align = centered @ (direction / np.linalg.norm(direction))
    order = np.argsort(align)
    names = [t for _, t, _ in vocab]
    return [names[i] for i in order[::-1][:k]], [names[i] for i in order[:k]]


def tag_shift(emb_high: np.ndarray, emb_low: np.ndarray, text_emb: np.ndarray, k: int = 4) -> tuple[list, list]:
    """Tags whose similarity rises most, and falls most, from one set of clips to another."""
    delta = scores(emb_high, text_emb).mean(0) - scores(emb_low, text_emb).mean(0)
    order = np.argsort(delta)
    names = [t for _, t, _ in vocabulary()]
    return ([(names[i], float(delta[i])) for i in order[::-1][:k]], [(names[i], float(delta[i])) for i in order[:k]])
