# Research map

What already exists around slider controls for music diffusion, what we take from each piece, and what is still open. Compiled 1 to 2 October 2026. Every link was opened during compilation. Items marked (abstract only) were checked at the abstract page and not read in full.

## Where this project sits

Two papers already run LoRA Concept Sliders on Stable Audio Open as a baseline, so "port Concept Sliders to audio" is not a contribution by itself. What we could not find anywhere:

1. A slider scored against a measured signal descriptor. Published audio sliders are scored with CLAP or MuQ text alignment only.
2. Sliders for acoustic and production properties (reverberation, note density, stereo width, low end). Published ones cover instrument, genre, mood, vocal gender, tempo, and sound-effect intensity.
3. A comparison with plain signal processing. For brightness or reverb, an EQ or a convolution is the obvious baseline and nobody reports it.
4. A leakage matrix: when "brighter" moves, what else moves?
5. Sliders trained from before/after audio pairs with known DSP parameters.
6. SliderSpace-style unsupervised discovery for audio (PCA over CLAP embeddings of generated clips, one LoRA per direction).

Items 1 to 4 are the evaluation this repo runs. Items 5 and 6 are the method extensions.

## Slider methods from images

| Work | What it does | What we use |
|---|---|---|
| [Concept Sliders](https://arxiv.org/abs/2311.12092), Gandikota et al., ECCV 2024. [code](https://github.com/rohitgandikota/sliders) | LoRA trained so that its output matches the base prediction plus a scaled difference between a positive-prompt and a negative-prompt prediction. | The training objective in `audiosliders/train.py`, in v-prediction form. Their released text-slider script trains scale +1 only; we train +1 and -1 in the same batch. Also the trick of leaving the slider off for the first steps to keep layout. |
| [SliderSpace](https://arxiv.org/abs/2502.01639), Gandikota et al., ICCV 2025. [code](https://github.com/rohitgandikota/sliderspace) | PCA over CLIP embeddings of generated images, then one rank-1 LoRA per component, trained on a distilled 4-step model so the CLIP loss can see a clean image after one step. | The discovery experiment. The released code differentiates through a single denoiser step and trains only `to_k`/`to_v` of cross-attention with a frozen orthogonal up-projection. |
| [Text Slider](https://arxiv.org/abs/2509.18831), Chiu et al., WACV 2026 (abstract only) | LoRA in the text encoder, no backprop through the denoiser. | A cheaper variant to try on T5. |
| [Prompt Sliders](https://arxiv.org/abs/2409.16535), Sridhar and Vasconcelos, 2024 (abstract only) | Slider as a learned text embedding. | Reference for the embedding-interpolation baseline. |
| [Semantic directions in T2I](https://arxiv.org/abs/2403.17064), Baumann et al., CVPR 2025 (abstract only) | Token-level directions in the text embedding. | Same. |
| [FluxSpace](https://arxiv.org/abs/2412.09611), Dalva et al., CVPR 2025 (abstract only) | Training-free editing in rectified-flow transformer blocks. | Relevant if the backbone changes to a flow model. |

## Sliders and steering already done in audio

| Work | Backbone | Controls shown | How they score it | Notes for us |
|---|---|---|---|---|
| [FreeSliders](https://arxiv.org/abs/2511.00103), Ezra et al., Oct 2025. [code](https://github.com/azencot-group/Free_Sliders) | Stable Audio Open 1.0 (plus image and video models) | Ten concepts, mostly sound effects; two musical | CLAP, LPAPS, and three new measures: conceptual range, smoothness, semantic preservation | Training-free: applies the Concept Sliders target at inference. Equivalent to our `guidance` method, which costs two extra forward passes per step. Reports that the useful scale differs per concept and saturates. |
| [TADA!](https://arxiv.org/abs/2602.11910), Staniszewski et al., Feb 2026. [code](https://github.com/luk-st/steer-audio) | AudioLDM2, Stable Audio Open, ACE-Step | Tempo, mood, vocal gender, instruments, genres | CLAP and MuQ alignment against LPAPS preservation, Audiobox Aesthetics, listener study | Activation steering. Finds that concepts concentrate in a few cross-attention layers and that Concept Sliders loses 21 to 25% when confined to them. The paper to compare against on semantic concepts. |
| [Activation Patching for Interpretable Steering in Music Generation](https://arxiv.org/abs/2504.04479), Facchiano et al., Apr 2025 | Not named in the abstract | Tempo (fast/slow), timbre (bright/dark) | Steering-strength analysis | Closest attribute overlap with our brightness and tempo sliders. |
| [ZETA / ZEUS](https://arxiv.org/abs/2402.10009), Manor and Michaeli, ICML 2024 (abstract only) | Audio diffusion via DDPM inversion | Unsupervised directions per clip | | Nearest prior for unsupervised discovery. Directions are per clip, so they are not reusable sliders. |
| [MusicRFM](https://arxiv.org/abs/2510.19127), Zhao et al. (abstract only) | MusicGen-Large | Notes, chords, tempo | Probe accuracy, FD, MMD, CLAP | Autoregressive counterpart. |
| [Discovering and Steering Interpretable Concepts in Large Generative Music Models](https://arxiv.org/abs/2505.18186), Singh et al. (abstract only) | MusicGen | SAE features | CLAP, listening test | Nikhil Singh advises Noah Schaffer (below). |
| [SMITIN](https://arxiv.org/abs/2404.02252), Koo et al. (abstract only) | MusicGen | Drums present, real vs synthetic | Probe output | Monitors the probe to cap intervention strength. |
| Anonymous "Audio Concept Sliders" [repo](https://github.com/audiosliderreview2026-byte/Audio-Concept-Sliders) | AudioLDM2 | | | An anonymous 2026 submission. Rank 4, MSE concept loss plus a cosine direction term. Only skimmed. |

Editing and conditioning work that solves a neighbouring problem: [DITTO](https://arxiv.org/abs/2401.12179) (optimize the initial noise against a feature loss), [Music ControlNet](https://arxiv.org/abs/2311.07069) (time-varying melody, dynamics, rhythm; scored by re-extracting the control from the output, which is the same idea as our descriptor scoring), [MuseControlLite](https://arxiv.org/abs/2506.18729) (adapters on Stable Audio Open), [MusicMagus](https://arxiv.org/abs/2402.06178) (text-embedding direction edits), [Text2FX](https://arxiv.org/abs/2409.18847) (CLAP-guided differentiable EQ and reverb, the reference for "just use effects").

## Noah Schaffer

A CS PhD student at Dartmouth in the SAHAS Lab, advised by Nikhil Singh; previously in Bryan Pardo's Interactive Audio Lab at Northwestern ([site](https://noahschaffer.github.io/), [GitHub](https://github.com/noahschaffer)). He lists music editing, mechanistic interpretability, and generation control as interests.

- [RIME: Enabling Large-Scale Agentic Music Post-Production](https://arxiv.org/abs/2607.19605), Schaffer and Singh, Jul 2026. [code](https://github.com/sahaslab/RIME). Builds paired edit-instruction data by applying effects with known parameters. Its POEMS toolkit covers reverb, delay, chorus, distortion, filters, shelf and peak EQ, gain, pan, and stem separation. Evaluation uses FAD and KAD in MERT space plus a directional edit similarity.
- [Music Separation Enhancement with Generative Modeling](https://interactiveaudiolab.github.io/assets/papers/ismir2022-schaffer-et-al.pdf), ISMIR 2022.

What it offers this project: nothing on diffusion sliders directly. The useful idea is the RIME recipe of rendering before/after pairs with known effect parameters. Applied here, those pairs become supervision for sliders whose unit is a physical quantity (item 5 above), and the same effects are the DSP baseline. `audiosliders/dsp.py` implements the effects we need without the dependency.

## Evaluation

| Tool | Use |
|---|---|
| [Adapting Fréchet Audio Distance for Generative Music Evaluation](https://arxiv.org/abs/2311.01616), Gui et al., ICASSP 2024. [fadtk](https://github.com/microsoft/fadtk) | FAD depends on sample size, embedding, and reference set. We report it with CLAP embeddings against unsteered clips from the same prompts and treat it as relative. |
| [KAD](https://arxiv.org/abs/2502.15602), Chung et al., 2025. [kadtk](https://github.com/YoonjinXD/kadtk) | MMD-based and unbiased at small sample sizes, which fits per-scale sets of a few hundred clips. |
| [stable-audio-metrics](https://github.com/Stability-AI/stable-audio-metrics) | The protocol from the Stable Audio Open paper (FD-OpenL3, KL-PaSST, CLAP score). |
| [Audiobox Aesthetics](https://arxiv.org/abs/2502.05139) | Reference-free quality per clip; TADA uses it along the slider. |

## Backbones

| Model | Type | Access | Notes |
|---|---|---|---|
| [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) ([paper](https://arxiv.org/abs/2407.14358)) | 1.06B DiT, v-prediction, T5-base, 44.1 kHz stereo, up to 47 s | Gated, Stability AI Community License | What this repo uses. In diffusers as `StableAudioPipeline`. |
| [Stable Audio Open Small](https://huggingface.co/stabilityai/stable-audio-open-small) ([paper](https://arxiv.org/abs/2505.08175)) | Rectified flow with adversarial post-training, 8 steps, 11 s | Gated; our token has not been granted access | The natural backbone for the SliderSpace loss, which wants a clean output after one or two steps. Loads through stable-audio-tools only. |
| [ACE-Step 1.5](https://arxiv.org/abs/2602.00744) | Flow-matching DiT | MIT, ungated | A second backbone if results need to generalize. |
| AudioLDM2, MusicLDM | U-Net latent diffusion, 16 kHz | Ungated, non-commercial | Lower fidelity. |

Existing LoRA tooling for Stable Audio Open: stable-audio-tools ships native LoRA with a sigma interval gate; [LoRAW](https://github.com/NeuralNotW0rk/LoRAW) is an older community trainer. Neither exposes a signed, per-sample scale, which is why `audiosliders/lora.py` is hand-rolled.

## Implementation facts worth not rediscovering

- The diffusers pipeline always generates the full 1024-frame latent (47.5 s) and crops. The transformer accepts shorter latents; `backbone.py` sizes the latent to the requested duration. On 8 prompts in `experiments/00_smoke.py`, 10 s clips reach a mean CLAP text-audio similarity of 0.42 against 0.47 for the first 10 s of 47 s clips, a gap smaller than the spread across prompts (0.28 to 0.57). Generation takes 0.65 s per 10 s clip at 50 steps on one RTX 6000 Ada.
- Unconditional input for classifier-free guidance is all-zero cross-attention tokens, including the two duration tokens. The global duration embedding is kept.
- The sampler's sigma maps to model time by t = (2/pi) atan(sigma). In z_t = cos(pi t/2) x0 + sin(pi t/2) eps form the transformer can be called directly with no scheduler.
- `laion/larger_clap_music` loads into transformers 5.18 with every text embedding identical. `laion/larger_clap_music_and_speech` and `laion/clap-htsat-unfused` behave correctly.
- The CLAP feature extractor in transformers is numpy and non-differentiable. A CLAP-space training loss needs a torch mel front end (48 kHz, n_fft 1024, hop 480, 64 Slaney mel bins from 50 Hz to 14 kHz, log10 power).

## Second pass (2 October 2026): how the closest work measures, and what that means for us

Read from full texts and repositories. Numbers are theirs.

### TADA in detail ([arXiv:2602.11910](https://arxiv.org/abs/2602.11910), [code, MIT](https://github.com/luk-st/steer-audio))

- **Backbone and scale.** The benchmark runs on ACE-Step v1 (3.5B), with Stable Audio Open and AudioLDM2 used only to localize layers. 9 concepts, 100 test prompts, one seed, 31 strengths, 30-second clips.
- **Metric.** Alignment against preservation, summarised as an area under the curve: alignment is the change in CLAP or MuQ-MuLan similarity to a fixed query, preservation is LPAPS distance to the unsteered clip. Quality is Audiobox Aesthetics at matched preservation.
- **Result.** Localized sparse-autoencoder steering scores highest (0.118 MuQ AUC). Their LoRA Concept Sliders baseline scores 0.086 and is second on the CLAP version of the metric. Restricting the LoRA to the few "functional" cross-attention layers costs it 21 to 25%.
- **Independent checks.** They validate mood against spectral centroid and tempo against onset rate, which is the same instinct as our descriptor test, applied to two concepts.
- **Listening study.** 32 sessions, 1,279 ratings, three 1-to-5 questions per sample on an interactive strength slider. Concept Sliders was not included.

What we take: MuQ-MuLan as a second, CLAP-independent scorer (done, `experiments/muq_score.py`); activation steering as a baseline (done, `audiosliders/steer.py`); the gap that nobody has human data on LoRA music sliders.

### FreeSliders in detail ([arXiv:2511.00103](https://arxiv.org/abs/2511.00103))

- Stable Audio Open 1.0, 10-second clips, 10 concepts of which 8 are sound effects; the two musical ones are choir pitch and electric versus acoustic guitar. 10 seeds, 7 strengths. No listening study.
- Metrics: conceptual range (CLAP alignment gained toward each end), conceptual smoothness (spread of consecutive alignment gaps), semantic preservation (mean LPAPS to the unsteered clip).
- Their Concept Sliders LoRA baseline sometimes goes the wrong way (negative range on one concept). Our `guidance` method is the same training-free idea without their automatic strength search.

### Others found

| Work | What it adds |
|---|---|
| [AnchorSteer](https://arxiv.org/abs/2605.31053), Chang et al., KDD 2026 | Text-free concept modules on Stable Audio Open trained by reconstruction on 1,000 self-generated clips per concept, with a 28-person study. The nearest relative of our set trainer; it trains an injected module per concept where we train a signed low-rank update between two sets. |
| [Do Text-to-Music Models Really Follow Instructions?](https://arxiv.org/abs/2608.11899), Wang, 2026 | Scores key and beat control on ACE-Step 1.5, Stable Audio 3, and LeVo2 with a key estimator and a beat tracker against matched neutral prompts. Shows that apparent agreement with an instruction can be the model's prior. Supports measuring with independent estimators. |
| Anonymous [Audio-Concept-Sliders](https://github.com/audiosliderreview2026-byte/Audio-Concept-Sliders) repo | AudioLDM2, real-audio editing through inversion. Includes a prompt-free paired or unpaired regime with a loss toward target-domain latents. |
| Community [ACE-Step 1.5 XL Concept Sliders](https://huggingface.co/Xanthius/Ace-Step-1.5-XL-Concept-Sliders) | 19 sliders trained with ai-toolkit, no evaluation published. We load them through `SliderBank.load_peft` and run them through the same protocol as ours. |
| [A Quantized Native Runtime for On-Device Semantic Audio Generation](https://arxiv.org/abs/2607.08526), Spanio and Rodà, 2026 | Mainly a runtime paper on Stable Audio 3. Our literature pass reports from its full text that it compares difference-in-means steering with a LoRA and finds late-step injection halves the damage, which matches our gating result. Not re-read by us; check before citing for that claim. |

### Measures added because of this pass

- **MuQ-MuLan** direction score, in a separate environment because MuQ needs transformers 4.x.
- **Beat This** beat tracker for tempo, in place of the librosa tempo estimate that suffers octave errors.
- **Confidence intervals** on every rank correlation and effect size.
- Still to add: LPAPS and the TADA area-under-curve for a like-for-like row; Essentia mood and instrument classifiers; a key estimator.

### Other open backbones worth a slider study

Stable Audio 3 (flow matching, open base checkpoints, official LoRA documentation, in diffusers) is the obvious third backbone. MiniMax Music 3, YuE2, DiffRhythm 2, and Magenta RealTime 2 have open weights but are autoregressive or hybrid, so the trainers here would need more than a wrapper.

## Third pass (2 October 2026): discovery methods, embedding spaces, and how axes are judged

### The image-side analogue of our descriptor sliders

[Measured Sliders](https://arxiv.org/abs/2609.05234), Chen, Wei, and Yin, September 2026. Sliders for image models defined by closed-form *differentiable* image measurements, trained with an objective that moves the target and holds the others still. Reports 98.9% monotone sweeps, a selectivity score (own effect over drift), and composition success for pairs and triples. Our descriptors are not differentiable through the audio decoder, so we reach the same goal by sorting clips on the measurement and training between the two ends, and by balancing the two sets on the measurements that should stay still. We adopt their selectivity and monotone-share metrics.

### Discovery beyond PCA

| Method | Idea | What we took |
|---|---|---|
| SliderSpace | PCA on CLIP embeddings, one LoRA per component. Never measures whether the trained sliders are distinct; relies on PCA orthogonality in the encoder. | Our leakage matrix and selectivity score measure what it assumes. |
| ICA on embeddings ([Yamagiwa et al., EMNLP 2023](https://arxiv.org/abs/2305.13175)) | Whiten with PCA, then rotate to maximise non-Gaussianity; components are more interpretable than principal ones. | `independent_directions` in `audiosliders/contrast.py`. |
| Sparse autoencoders on music embeddings ([Guinot et al., 2026](https://arxiv.org/abs/2608.08757)) | Sparse features of CLAP and MuQ embeddings of real music, used to steer retrieval. Nobody trains generator sliders from them. | `audiosliders/sae.py`. At 1,024 features on our corpus only 9 reproduce across seeds, so they need a stability filter. |
| NoiseCLR, LatentCLR, pullback-metric and Jacobian methods | Discover directions inside the generator with contrastive or spectral objectives. | Not used yet; the cost is a joint training run per set of directions. |

No paper was found that runs PCA or ICA over CLAP or MuQ embeddings of a real music corpus to name musical dimensions, or that trains sliders from such axes.

### How people claim a set of axes is good

- **Selectivity**: own effect over drift on other measurements (Measured Sliders). Implemented in `audiosliders/metrics.py`.
- **Monotone share**: fraction of sweeps with rank correlation above 0.8 (same source). Implemented.
- **Stability**: matched cosine between axes fitted on different data or seeds. Implemented as `axis_stability`.
- **Composition**: whether two or three sliders applied together each still move their own measurement. `experiments/compose.py`.
- **Human**: SliderSpace asked raters which of two image grids was more diverse, useful, and creative; TADA used three Likert questions on an interactive strength slider.

### Perceptual dimensions to relate discovered axes to

Listener studies repeatedly reduce perceived musical attributes to arousal, valence, and depth ([Fricke, Greenberg, Rentfrow, and Herzberg](https://www.repository.cam.ac.uk/handle/1810/302164)). The independent components of MuQ-MuLan embeddings of FMA include an arousal axis and a valence axis (see [`results/discovery/real/`](../results/discovery/real/)).

### Quality predictors

Audiobox Aesthetics is the quality signal used here and by TADA. Its correlation with human preference on generated music is reported as weak by the MuQ-Eval authors, so results that depend on it (the usable span, the quality slider) should be re-checked with a second predictor such as MuQ-Eval before submission.

### Larger real-music corpora

MTG-Jamendo (55,000 full tracks with genre, instrument, and mood tags) is the cleanest step up from FMA and is the corpus other recent work uses. FMA-large (106,574 clips) is already on disk here.

## Fourth pass (2 October 2026): axes inside the generator, and coverage of real music

Two experiments were added after this pass: principal axes of ACE-Step's own cross-attention activations ([`experiments/discover_internal.py`](../experiments/discover_internal.py)), and how much of each real-music axis a model's output spans ([`results/coverage/`](../results/coverage/README.md)). This is what already exists for each.

### Unsupervised directions inside image diffusion models

| Work | What they decompose | What they report |
|---|---|---|
| [Kwon et al., 2022](https://arxiv.org/abs/2210.10960) | The U-Net bottleneck ("h-space"), edited with a small learned module | Edits there are close to linear and consistent across timesteps. |
| [Haas et al., 2023](https://arxiv.org/abs/2303.11073) | PCA of h-space activations, and spectral directions of the Jacobian | Interpretable directions with no text model and no fine-tuning. The closest image analogue to our internal axes. |
| [Park et al., 2023](https://arxiv.org/abs/2307.12868) | Local basis from the pullback metric | Directions are local: they depend on the sample and the timestep. |
| [NoiseCLR, Dalva and Yanardag, 2023](https://arxiv.org/abs/2312.05390) | Contrastive directions learned from unlabelled images | Directions compose and stay disentangled. |
| [LOCO Edit, Chen et al., 2024](https://arxiv.org/abs/2409.02374) | Singular vectors of the Jacobian | Local linearity holds only in a band of noise levels. |
| [Surkov et al., 2024](https://arxiv.org/abs/2410.22366) | Sparse autoencoders on SDXL Turbo block updates | Features are causal and different blocks do different jobs. |
| [STRIDE, Yadav et al., 2026](https://arxiv.org/abs/2605.11494) | PCA of a diffusion transformer's own activations | Used to shape noise that restores diversity in one-step models. The axes are not named or offered as controls. |
| [ELROND, Skierś et al., 2026](https://arxiv.org/abs/2602.10216) | PCA or a sparse autoencoder on gradients of the differences between seeds of one prompt | Steering directions in the text-embedding input. Their "fixed prompt, vary the seed" setup matches our removal of the prompt mean. |

### Inside music generators

| Work | Model | Method | Difference from ours |
|---|---|---|---|
| [TADA](https://arxiv.org/abs/2602.11910), section 3 | ACE-Step | TopK sparse autoencoders on cross-attention outputs; features for a concept are chosen with positive and negative prompts | Same activations. Their features are picked per named concept from prompt pairs; ours are principal axes with the prompt mean removed and no label at any point. |
| [Singh, Cherep, and Maes, 2025](https://arxiv.org/abs/2505.18186) | Autoregressive transformer music generators | Sparse autoencoders on the residual stream with automated labelling | Autoregressive models. |
| [SMITIN, Koo et al., 2024](https://arxiv.org/abs/2404.02252) | MusicGen | Probes on attention heads steer toward an instrument | Supervised. |

We found no paper that runs PCA over a music diffusion transformer's activations and steers along the unnamed components. The claim to make is that narrow one.

### Diversity and coverage against real music

| Work | What is compared | Measure |
|---|---|---|
| [Presto!, Novack et al., 2024](https://arxiv.org/abs/2410.05167) | Distilled text-to-music models against a reference set | Density, recall, and coverage in CLAP space, as single numbers. |
| [Huang et al., 2025](https://arxiv.org/abs/2503.16669) | Text-to-music metrics against human preference data | Fréchet Audio Distance is inconsistent with human preference; they propose a MAUVE-based divergence (rank correlation 0.84 against 0.49). |
| [Slendebroek and Metaxa, 2026](https://arxiv.org/abs/2608.06106) | Suno and Lyria 3 against human music in four genres | 72 hand-built MIR features with dispersion and separability diagnostics. They report homogenisation. |

The existing evidence is either one aggregate number per model, or dispersion of signal features for commercial systems. Our table differs in two ways: the axes are learned from real music and carry musical labels, and the models are open ones that a slider can then be trained on.

### Open models noted for later (from the search pass; check each model card before use)

`ACE-Step/acestep-v15-xl-sft-diffusers` (MIT, same pipeline class as the turbo model used here), `ASLP-lab/DiffRhythm2` (Apache-2.0), `declare-lab/JAM-0.5` (non-commercial, flow transformer), `HeartMuLa/HeartMuLa-oss-3B-happy-new-year` (Apache-2.0, language model plus codec).
