# Legacy

`v0/` is the repository as it stood before the October 2026 rebuild, kept unchanged for reference.

It was a scaffold. `scripts/generate_samples.py` wrote white noise in place of diffusion output, `scripts/compute_embeddings.py` wrote hash-seeded random vectors in place of CLAP embeddings, and `scripts/train_sliders.py` wrote a JSON plan without training anything. The one part that measured real audio was `descriptor_demo.py`, a synthetic-tone diagnostic for spectral centroid and onset rate.

Nothing under `v0/` is imported by the current code. The descriptor idea survives in [`audiosliders/descriptors.py`](../audiosliders/descriptors.py), now applied to actual model output.

The SliderSpace paper PDF that used to live in `Papers/` was removed from the tree. Read it at https://arxiv.org/abs/2502.01639.
