# Listening study: design

Not yet run. This is the plan for the one piece of evidence the project lacks: whether people hear what the measurements say. It needs human-subjects review before any data is collected for publication.

**A pilot version is live**: [takakhoo.github.io/audio-diffusion-control/listen.html](https://takakhoo.github.io/audio-diffusion-control/listen.html). It runs tasks A and C below on the published demo clips (12 direction questions at positions -1 and +1, 10 edited-clip ratings, one hidden identical pair), takes about ten minutes, and uploads nothing: answers stay in the browser and download as a JSON file. [`experiments/listening_results.py`](../experiments/listening_results.py) turns a set of those files into accuracy with exact binomial intervals and mean ratings. The pilot is for checking the procedure and for informal feedback. It compares two ways of making a slider on some attributes and does not include the method comparison of task B.

## What it has to answer

1. **Direction.** When a slider moves, do listeners hear the named attribute move the same way?
2. **Usable span.** Does quality, as heard, fall where the aesthetics predictor says it falls?
3. **Same piece.** At the ends of the usable span, do listeners still recognise the clip as the piece they started from?
4. **Method.** Do listeners prefer the trained slider, activation steering, or guidance applied at inference, at matched strength?

## Stimuli

- 6 sliders: mood, ensemble, harmony, brightness, tempo, and one discovered axis.
- 4 held-out prompts per slider, one seed, 5 positions inside the usable span, 30-second clips cut to the first 12 seconds.
- 3 methods on the same prompt and seed: trained slider, activation steering, guidance at inference.
- All clips matched to the same loudness, as in the demo.

## Tasks

| Task | What the listener does | What it measures |
|---|---|---|
| A. Which is more X? | Hears the two ends in random order, picks the one that is more *happy* (or fuller, brighter, faster...). Two-alternative forced choice. | Direction. Reported as accuracy with a binomial interval; chance is 50%. |
| B. Rate the strip | Hears five positions of one clip on a slider like the demo's, then rates on 1 to 5: how well the clip took on the attribute without sounding unnatural or losing its identity; overall audio quality; how even the steps felt. | Comparable with the three questions in TADA's study. |
| C. Same piece? | Hears the unsteered clip and one end; rates 1 to 5 whether it is recognisably the same piece. | Preservation. |

24 trials per listener, about 20 minutes: 8 of task A, 8 of task B, 8 of task C, balanced over sliders and methods and shuffled.

## Listeners

- 30, which matches or exceeds the closest studies (28 in AnchorSteer, 32 sessions in TADA).
- Self-rated musical experience recorded on a 1 to 5 scale.
- Two attention checks: an identical pair in task C, and a clip replaced by noise in task B. Failing either drops the session.

## Analysis

- Task A: accuracy per slider and method with exact binomial intervals.
- Tasks B and C: mixed-effects model with listener and prompt as random effects; pairwise comparisons between methods with Holm correction.
- Agreement with the measurements: rank correlation between mean heard quality per position and mean aesthetics score per position, and between task A accuracy and the descriptor's rank correlation.

## What would count against the project

- Task A accuracy near 50% for a slider whose descriptor and embedding scores both pass.
- Heard quality falling well inside the span the predictor calls usable.
- Listeners preferring guidance at inference or activation steering at matched strength.

Any of these would be reported.
