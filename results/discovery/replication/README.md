# Do the axes come back in a different sample of real music?

Fitting axes on two random halves of one corpus says they are stable under resampling ([`../real`](../real/README.md): matched cosine 0.83 to 0.98). This is a harder test. Corpus B is the 79,904 FMA-large recordings that are not in the first corpus (47,942 after removing the 40% most vocal), a different slice of the archive with a different genre mix: more experimental, lo-fi, metal, and classical, and with genre labels that are partly guessed from CLAP because FMA has none for many of these tracks. Axes were fitted on B from scratch and matched one to one with the axes from corpus A (14,985 recordings).

| Embedding | Method | Matched cosine, two halves of A | Matched cosine, A against B |
|---|---|---:|---:|
| MuQ-MuLan | principal components | 0.98 | 0.66 |
| MuQ-MuLan | independent components | 0.90 | 0.61 |
| CLAP | principal components | 0.94 | 0.64 |
| CLAP | independent components | 0.83 | 0.56 |

Per-axis tables: [`muq.md`](muq.md), [`clap.md`](clap.md).

## The space is the same and the axes rotate inside it

| | MuQ-MuLan | CLAP |
|---|---:|---:|
| Share of A's top-8 principal subspace found in B's top-8 | 0.88 | 0.79 |
| Same for the top 16 | 0.90 | 0.91 |
| Same for the top 32 | 0.97 | 0.97 |
| Within-genre spread of B along A's ten axes, relative to A | 0.85 to 1.14 | 0.82 to 1.09 |

The directions along which recordings of one genre differ span nearly the same subspace in both corpora, and the amount of spread along each of A's axes is within 18% in B. What changes from sample to sample is which directions inside that subspace a decomposition singles out.

## The seven axes that became sliders

| Slider axis | Best match among B's axes | Tags of that match | Share of the axis inside the span of B's axes |
|---|---:|---|---:|
| classical to funk (CLAP, principal) | 0.78 | sad, melodic, choir / funk, energetic, drums | 0.92 |
| arousal (MuQ, independent) | 0.76 | minor key, organ, electric guitar / aggressive, hip hop, drums | 0.94 |
| jazz to electronic (MuQ, independent) | 0.72 | latin, jazz, improvised / synthesizer, electronic dance, vocals | 0.94 |
| valence (MuQ, independent) | 0.65 | dark, choir, ambient / playful, rhythmic, energetic | 0.92 |
| piano (MuQ, independent) | 0.51 | piano, electric piano, melancholic / hand percussion, swung, mysterious | 0.97 |
| acoustic to electronic (CLAP, independent) | 0.50 | melodic, sad, mysterious / fast, drums, funk | 0.90 |
| strings to synth (MuQ, independent) | 0.43 | country, acoustic guitar, acoustic / organ, reggae, happy | 0.84 |

## Reading

- **Four of the seven come back as their own axis with the same meaning**: classical to funk, arousal, jazz to electronic, and valence, with cosines of 0.65 to 0.78 and matching tags at both ends.
- **Piano comes back at one end.** B has an axis with piano and electric piano at one end; its other end is different.
- **Strings to synth and acoustic to electronic do not come back as single axes.** Both still lie mostly inside the span of B's axes (0.84 and 0.90), so the direction exists in B and is not one that B's decomposition isolates.
- **Split-half stability overstates how well a single axis generalises.** It tests resampling from one corpus. Against a different slice of real music the matched cosine falls from 0.90 to 0.61 for the independent components used here.
- **The coverage result does not depend on which corpus defines "real".** Coverage divides by the spread of real music along an axis, and that spread differs by at most 18% between the two corpora.

## Reproduce

```bash
python experiments/run_jobs.py experiments/jobs/19_real_large_embed.tsv --gpus 0 1 2
venv-muq/bin/python experiments/muq_embed.py --corpus runs/corpus/real_large --music ../data/fma/fma_large
python experiments/replicate_axes.py --a runs/corpus/real_ace --b runs/corpus/real_large --emb muq \
    --vocab runs/reference/vocab_muq.npz --out results/discovery/replication
```
