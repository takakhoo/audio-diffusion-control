"""Executable audio-control diagnostic; no diffusion, CLAP, or LoRA claims.

Render a plucked harmonic tone with brightness and note-rate controls, measure
the audio, and inspect PCA directions in standardized descriptor space.
"""
import argparse
import json
from pathlib import Path
import wave
import numpy as np
from scipy.signal import find_peaks

SR = 16000


def render(brightness, rate, seconds=3.0, seed=7):
    if not 0 <= brightness <= 1 or not 1 <= rate <= 10 or seconds <= 0:
        raise ValueError("brightness in [0,1], rate in [1,10], positive duration required")
    rng = np.random.default_rng(seed)
    t = np.arange(round(SR * seconds)) / SR
    output = np.zeros_like(t)
    phases = rng.uniform(0, 2 * np.pi, 10)
    weights = np.exp(-np.arange(10) * (3.0 - 2.7 * brightness))
    weights /= np.sqrt(np.sum(weights ** 2))
    for onset in np.arange(0, seconds, 1 / rate):
        local = t - onset
        active = (local >= 0) & (local < 0.1)
        tt = local[active]
        envelope = np.sin(np.pi * tt / 0.1) ** 2
        tone = sum(weight * np.sin(2 * np.pi * 220 * (k + 1) * tt + phases[k])
                   for k, weight in enumerate(weights))
        output[active] += 0.2 * envelope * tone
    return output


def descriptors(audio, sr=SR):
    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 1 or len(audio) < sr // 10 or not np.isfinite(audio).all():
        raise ValueError("Provide finite mono audio at least 0.1 seconds long")
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    frequencies = np.fft.rfftfreq(len(audio), 1 / sr)
    centroid = float(np.dot(frequencies, spectrum) / max(spectrum.sum(), 1e-20))
    frame = sr // 100
    framed = audio[:len(audio) // frame * frame].reshape(-1, frame)
    energy = np.sqrt(np.mean(framed ** 2, axis=1))
    peaks, _ = find_peaks(np.r_[0, energy, 0], prominence=max(energy.max() * 0.2, 1e-8), distance=5)
    return np.array([centroid, len(peaks) / (len(audio) / sr), np.sqrt(np.mean(audio ** 2))])


def fit_pca(features, rank=2):
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[0] < 2 or not np.isfinite(features).all():
        raise ValueError("Need at least two finite descriptor rows")
    if not 1 <= rank <= min(features.shape):
        raise ValueError("Invalid PCA rank")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1)
    normalized = (features - mean) / scale
    _, singular, vectors = np.linalg.svd(normalized, full_matrices=False)
    # Canonicalize signs: changing an SVD implementation must not flip slider labels.
    for vector in vectors:
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
    ratios = singular ** 2 / max(np.sum(singular ** 2), 1e-20)
    return {"mean": mean, "scale": scale, "components": vectors[:rank], "variance_ratio": ratios[:rank]}


def project(features, pca):
    return (features - pca["mean"]) / pca["scale"] @ pca["components"].T


def write_wav(path, audio):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def run(output="results/descriptor-demo"):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    controls = np.column_stack([rng.uniform(0, 1, 80), rng.uniform(2, 8, 80)])
    features = np.array([descriptors(render(b, r)) for b, r in controls])
    pca = fit_pca(features)
    held_controls = np.column_stack([rng.uniform(0, 1, 20), rng.uniform(2, 8, 20)])
    held = np.array([descriptors(render(b, r, seed=19)) for b, r in held_controls])
    coordinates = project(held, pca)
    restored = (coordinates @ pca["components"]) * pca["scale"] + pca["mean"]
    sweeps = {}
    for name, settings in [
        ("brightness", [(v, 4) for v in np.linspace(0, 1, 5)]),
        ("density", [(0.5, v) for v in [2, 3, 4, 6, 8]]),
    ]:
        measured = []
        for index, (brightness, rate) in enumerate(settings):
            audio = render(brightness, rate)
            descriptor = descriptors(audio)
            write_wav(out / f"{name}-{index}.wav", audio)
            measured.append(dict(brightness=brightness, note_rate=rate, descriptors=descriptor.tolist(),
                                 pca_coordinates=project(descriptor, pca).tolist()))
        sweeps[name] = measured
    result = dict(backend="additive-dsp-diagnostic", not_implemented=["diffusion inference", "CLAP", "LoRA training"],
                  descriptor_names=["power_spectral_centroid_hz", "detected_onsets_per_second", "rms"],
                  train_samples=80, held_out_samples=20, train_seed=7, held_audio_seed=19,
                  pca={key: value.tolist() for key, value in pca.items()}, sweeps=sweeps,
                  held_out_standardized_rmse=float(np.sqrt(np.mean(((restored - held) / pca["scale"]) ** 2))),
                  caveat="PCA variance directions are not automatically semantic or independent controls. This measures a known synthesizer, not a text-to-audio model.")
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez(out / "pca.npz", **pca, training_features=features, held_out_features=held)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    axes[0].plot([v["brightness"] for v in sweeps["brightness"]], [v["descriptors"][0] for v in sweeps["brightness"]], "o-")
    axes[0].set(title="Brightness measured from audio", xlabel="Brightness control", ylabel="Power centroid (Hz)")
    axes[1].plot([v["note_rate"] for v in sweeps["density"]], [v["descriptors"][1] for v in sweeps["density"]], "o-")
    axes[1].set(title="Density measured from audio", xlabel="Requested notes / second", ylabel="Detected onsets / second")
    points = axes[2].scatter(*project(features, pca).T, c=controls[:, 0], cmap="viridis", s=15)
    axes[2].set(title="Standardized descriptor PCA", xlabel="PC1", ylabel="PC2")
    fig.colorbar(points, ax=axes[2], label="Brightness setting")
    fig.savefig(out / "controls.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/descriptor-demo")
    run(parser.parse_args().output)
