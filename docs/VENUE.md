# Target venue

Decided 2 October 2026 after checking the official pages of about twenty venues. Dates marked *estimate* are last cycle's, because the 2027 call is not posted yet; replace them when it is.

## Primary: ISMIR 2027

International Society for Music Information Retrieval Conference, London, 12 to 16 September 2027 ([ismir.net](https://ismir.net/)).

- **Why here.** The ISMIR 2026 call lists evaluation methodology and evaluation metrics for generative tasks as topics ([call](https://ismir2026.ismir.net/authors/call-for-papers)). The closest precedent for this paper, "Aligning Text-to-Music Evaluation with Human Preferences", was at [ISMIR 2025](https://ismir2025program.ismir.net/poster_314.html), as was the control paper [LiLAC](https://ismir2025program.ismir.net/poster_53.html). The paper's main claim is about how music control should be measured, which is this community's question.
- **Deadline.** *Estimate:* abstract around 20 April 2027, full paper around 27 April 2027, notification around 10 July 2027.
- **Rules the draft already follows** ([author guidelines](https://ismir2026.ismir.net/authors/author-guidelines), [template](https://github.com/ismir/paper_templates/releases/tag/2026v1)): at most six pages of technical content plus references; A4; double-blind with line numbers; abstract of 150 to 200 words; numbered references in order of citation; PDF under 10 MB. [`paper/audiosliders.tex`](../paper/audiosliders.tex) uses the official style file with the `submission` option and currently builds to six pages including references.
- **Rules that need action before submitting.**
  - *Anonymity.* The demo page and this repository identify the author. Submission needs an anonymized copy of both as supplementary material.
  - *Preprints.* ISMIR strongly discourages near-duplicate arXiv postings and forbids promoting the paper during review. An arXiv version posted months earlier, with the submission adding the listening study and baselines, is the low-risk route. Take the venue name off the README between submission and notification.

## What the paper still needs to be competitive there

Status on 2 October 2026:

1. **A listening study. Open, and the main gap.** It should test whether the usable span from the two quality predictors matches where listeners hear quality drop, and whether a move along the arousal or valence axis is heard as one. Needs human-subjects review at Dartmouth. Design: [`LISTENING_STUDY.md`](LISTENING_STUDY.md).
2. **The two closest methods as baselines. Done on ACE-Step.** Prompt-pair guidance (the FreeSliders recipe) and activation steering (TADA's difference-of-means vectors) run through the same protocol, with prompt interpolation as a third ([`results/ace/summary.md`](../results/ace/summary.md)). Still open: their own metrics (LPAPS, the TADA area-under-curve score) and their sparse-autoencoder variant.
3. **Non-CLAP evidence. Done.** Every direction score is repeated in MuQ-MuLan, and quality is scored by SongEval as well as Audiobox Aesthetics.
4. **Confidence intervals. Done** for rank correlations and effect sizes in the summary tables.
5. **A tighter story. Done in the current draft**: protocol and method comparison first, then axes of real music, coverage, and sliders along them. Six pages including references, 200-word abstract.
6. **A third backbone. Done for six sliders** on Stable Audio 3 small ([`results/sa3`](../results/sa3/README.md)); the gated evaluation is still to run.
7. **A bigger and cleaner sample of real music** for the axes and the coverage numbers. FMA-large is on disk; MTG-Jamendo is the corpus other recent work uses.

## Backup and fallbacks

- **TISMIR** (Transactions of ISMIR): rolling, double-blind, 8,000 words, room for all four contributions; slow (about eight months) and has a publication charge with waivers ([submissions](https://transactions.ismir.net/about/submissions)).
- **ICASSP 2028**: four pages plus references; the 2027 deadline was 23 September 2026, so expect September 2027 (*estimate*).
- **Stake a claim now**: arXiv, since FreeSliders and TADA both appeared within the past year. A non-archival workshop is optional; an archival one would count as prior publication at ISMIR.

## Considered and set aside

- **ICLR 2027 and ICASSP 2027 main tracks**: deadlines passed in September 2026.
- **ICML 2027** (*estimate* late January) and **NeurIPS 2027 Evaluations and Datasets** (*estimate* May): better known, but reviewers there would call the trainer incremental and ask for a human study, more backbones, and a packaged benchmark; the dates also collide with ISMIR's.
- **EvoMUSART 2027** (1 November 2026): would take the paper nearly as is, but it is archival and would use the paper up.
- **CHI, IUI, NIME, Creativity and Cognition**: these communities expect a study with musicians.

## Timeline

| When | What |
|---|---|
| October 2026 | Freeze current results; post arXiv v1; file for human-subjects review |
| November 2026 to January 2027 | Baselines (FreeSliders, TADA), non-CLAP measures, confidence intervals |
| January 2027 | ISMIR 2027 call appears: replace the estimated dates; run the listening study |
| February to March 2027 | Final six pages on the ISMIR 2027 template; anonymized demo and code |
| Late April 2027 (*estimate*) | Submit |
| July 2027 (*estimate*) | Notification; if rejected, extend to TISMIR |
