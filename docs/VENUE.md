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

1. **A listening study.** The main gap. It should test whether the usable span from the aesthetics predictor matches where listeners hear quality drop, and whether descriptor shifts match heard changes. Needs human-subjects review at Dartmouth.
2. **The two closest methods as baselines** on the same backbones, scored with this protocol and with their own metrics: FreeSliders ([arXiv:2511.00103](https://arxiv.org/abs/2511.00103)) and TADA activation steering ([arXiv:2602.11910](https://arxiv.org/abs/2602.11910)).
3. **Non-CLAP evidence** for the semantic sliders, since tags and direction scores both come from CLAP.
4. **Confidence intervals** on every headline number.
5. **A tighter story.** Six pages will not hold four contributions at equal weight. Lead with the protocol and the embedding-passes, waveform-fails result.

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
