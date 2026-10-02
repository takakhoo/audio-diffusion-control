# Axes discovered in real music

24,975 thirty-second recordings from FMA-medium, measured on their middle ten seconds. Recordings in the top 40% of a CLAP "vocals" score were left out, leaving 14,985. Each genre's mean embedding was removed first, so an axis describes how recordings of the same genre differ.

| File | Embedding | Method | Stability across two halves of the corpus |
|---|---|---|---:|
| [`pca.md`](pca.md) | CLAP | principal components | 0.94 |
| [`ica.md`](ica.md) | CLAP | independent components | 0.83 |
| [`muq_pca.md`](muq_pca.md) | MuQ-MuLan | principal components | 0.98 |
| [`muq_ica.md`](muq_ica.md) | MuQ-MuLan | independent components | 0.90 |
| [`sae.md`](sae.md) | CLAP | sparse autoencoder, 1,024 features | 9 of 1,024 features reproduced by a second seed |

Stability is the mean absolute cosine between axes fitted on two random halves, after matching them one to one. For the sparse autoencoder it is the number of features whose decoder atom has cosine above 0.8 with some atom from a second training seed.

Each table lists, per axis, the tags its two ends align with and its strongest rank correlations with the measured descriptors, the beat-tracked tempo, and the aesthetics scores.

A harder test, against 47,942 recordings from a different slice of FMA, is in [`../replication`](../replication/README.md): the leading subspace carries over and single axes do so only partly (matched cosine 0.56 to 0.66).

## What stands out

- **MuQ-MuLan gives the most reproducible and the most musical axes.** Its independent components include an arousal axis (quiet, minor key, dreamy, melancholic against aggressive, dry, rhythmic, energetic), a valence axis (dark, metal, distorted against playful, pop, simple, happy), jazz against electronic, and plucked or bowed strings against synthesizer and choir. Arousal and valence are the two dimensions that listener studies of music keep finding.
- **Principal components are more stable; independent components read more cleanly.** A principal component such as CLAP's first ("violin, distorted, flute, live" against "calm, acoustic guitar, simple, minor key") mixes instruments and moods. The independent components more often name one contrast.
- **The sparse autoencoder at this size is mostly noise between seeds.** Only 9 of 1,024 features were found twice. The few that survive are coherent (for example "orchestral film score, strings, major key, cello"), but the dictionary as a whole is not something to build on without a stability filter.
- **Several leading axes track the quality predictor.** CLAP's second principal component correlates -0.47 with content enjoyment, and MuQ's second correlates +0.63. Part of what varies most within a genre is how polished a recording sounds.
