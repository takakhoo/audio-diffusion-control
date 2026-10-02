# Results

What each file is and which script wrote it. Raw audio and per-clip rows live on the training machine; everything here is small enough to commit.

| Path | What it is | Written by |
|---|---|---|
| [`ace/summary.md`](ace/summary.md), `summary.csv`, `summary.json` | One row per slider on ACE-Step 1.5 XL turbo: monotonicity, usable span, descriptor and CLAP movement inside it, piece kept, enjoyment at zero and at the ends, kernel distance to real recordings | `experiments/report.py` |
| [`ace/tags.md`](ace/tags.md) | The tags that rise and fall most between the two ends of each slider | `experiments/report.py` |
| `ace/response.png`, `quality.png`, `tradeoff.png`, `leakage_lora.png`, `leakage.csv` | Descriptor against position; enjoyment against position; piece kept against descriptor moved; slope of every descriptor for every slider | `experiments/report.py` |
| [`gating/`](gating/) | Stable Audio Open: what switching the slider on later buys and costs, for four sliders at five start times | `experiments/jobs/02_gating.tsv`, summarised inline |
| [`discovery/ace/pca.md`](discovery/ace/pca.md), `pca.json` | First six principal components of CLAP embeddings for each of five concepts, with tag labels and descriptor correlations | `experiments/discover.py` |
| [`figures/`](figures/) | Architecture diagram, overview, training curves, gating, discovery spectrum, quality yardstick | `experiments/make_figures.py`, `experiments/make_pipeline_figure.py` |
| [`demo/sliders.gif`](demo/sliders.gif) | Animated sweep of four sliders | `experiments/make_gif.py` |

## Reading the summary table

- **Monotonicity ρ**: mean Spearman correlation between slider position and the slider's descriptor, over trajectories (one prompt and seed at every position).
- **Ends ordered**: share of trajectories whose two extreme positions are ordered the intended way.
- **Usable from / to**: positions, walking out from zero, over which mean content enjoyment stays within 0.5 of the unsteered clips.
- **Descriptor moved (std)**: change in the descriptor across the usable span, in standard deviations of the unsteered clips.
- **CLAP moved**: change across the usable span in the projection on the slider's text direction.
- **Piece kept**: CLAP similarity to the unsteered clip at the ends of the usable span.
- **Quality at 0 / at ends**: mean content enjoyment unsteered and at the two extreme positions rendered.

## Evaluation protocol

24 prompts that share nothing with the 48 training prompts ([`configs/prompts.yaml`](../configs/prompts.yaml)), three seeds each, ten-second clips. Every trajectory starts from the same initial noise at every position. Stable Audio Open uses 50 DPM-Solver++(2M) steps with guidance 7; ACE-Step turbo uses 8 Euler steps with no guidance.
