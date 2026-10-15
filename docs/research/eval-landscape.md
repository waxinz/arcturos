# Eval landscape: blind A/B quality judging for local LLMs

Research notes for the Arcturos eval side. Primary question: **"which model is better, A or B, on a specific
quality dimension?"** Sources were verified during the survey (October 2026); citations inline as markdown links.

## 1. Blind pairwise preference judging

- [Chatbot Arena (LMSYS/LMArena)](https://arxiv.org/abs/2403.04132) is the canonical design: two anonymized
  models answer the same prompt side by side, a human (or judge) picks A / B / tie **without knowing which model
  is which** — blindness is what prevents brand bias from contaminating votes.
- Ratings come from pairwise wins, not absolute scores. Arena moved from online Elo to a
  **Bradley-Terry maximum-likelihood fit** over all battles, with confidence intervals via bootstrap
  resampling ([Elo system update](https://arena.ai/blog/chatbot-arena-update)). BT is order-independent and
  reproducible; online Elo depends on battle order.
- [Arena-Hard](https://arxiv.org/abs/2406.11939) ([pipeline blog](https://www.lmsys.org/blog/2024-04-19-arena-hard/))
  shows the cheap variant: 500 curated hard prompts, every candidate judged **pairwise against a fixed baseline**
  answer (gpt-4-0314), then BT coefficients over those A-vs-baseline battles. That reached ~3x better model
  separation than MT-Bench and high correlation with full Arena rankings for ~$20 per full run — you do not need
  every model to play every other model.

## 2. LLM-as-judge bias catalog and mitigations

The [Judging LLM-as-a-Judge paper (MT-Bench)](https://arxiv.org/abs/2306.05685) cataloged the failure modes; headline
result: a strong judge (GPT-4) reaches **85% agreement with human experts (81% is human-human agreement)** on MT-Bench —
so LLM judging is viable *if* the biases below are handled.

- **Position bias**: judges over-pick one side. GPT-4's swap consistency on MT-Bench pairwise was only **65%**
  (weak judges were far worse: Vicuna-13B zero-shot 15–16%); [LLMs are not Fair Evaluators](https://arxiv.org/abs/2305.17926)
  found the same and showed evaluation quality jumps after swapping answer order.
  *Fix (adopted by MT-Bench itself)*: run the judge twice with A/B swapped; count a win only if both orders agree,
  else record a tie. Randomizing order is a cheaper large-scale alternative.
- **Verbosity / length bias**: judges prefer longer answers even when they are worse; MT-Bench showed a trivial
  "repetitive list" attack flips judges (GPT-4 resists best). [AlpacaEval 2.0](https://tatsu-lab.github.io/alpaca_eval/)
  introduced the **length-controlled (LC) win rate**, a regression-corrected preference estimate: on its leaderboard
  GPT-4-Turbo scores 46.1% raw vs 55.0% LC, while Yi-Large scores 57.5% raw vs 51.9% LC — i.e. raw win rates can
  inflate or deflate by ~5-9 points purely on length. [Arena style control](https://www.lmsys.org/blog/2024-08-28-style-control/)
  does the same for human votes by adding style covariates (length, markdown density) to the BT regression.
  *Fix*: always record output token/char length; report LC-style or length-stratified win rates alongside raw ones.
- **Self-enhancement / self-preference bias**: judges favor their own family — MT-Bench measured GPT-4 favoring
  itself with a **10% higher win rate** than humans awarded it (Claude-v1: +25%).
  [LLM Evaluators Recognize and Favor Their Own Generations](https://arxiv.org/abs/2404.13076) shows judges can
  *recognize* their own outputs and that self-recognition causally drives self-preference.
  *Fix*: prefer a judge from a different model family than both candidates, and store judge identity in run metadata.
- **Knowledge bias**: a judge cannot grade facts it does not know and will prefer confident-sounding-but-wrong
  claims. MT-Bench found judges fail on math/reasoning and that **reference-guided judging** (show the judge a
  verified reference answer) sharply reduces failures. *Fix*: never ask the judge to be the source of truth — pair
  judge grading with deterministic reference checks wherever an answer is verifiable (see §4).

## 3. Single-answer rubric grading vs pairwise

- MT-Bench supports both modes: **single_v1** asks for a 1-10 score against an optional reference answer plus a
  written rubric; **pairwise_v1** asks which of two answers is better. GPT-4's single-answer agreement with humans
  (85% on MT-Bench, S2) matched pairwise — the authors note GPT-4 has a "relatively stable internal rubric" — but
  single-answer produces **more tie votes** and depends more on the judge's calibration.
- Practical split: **rubric grading wins** when the question has a correct answer or verifiable constraints
  (math, extraction, format) — a 1-10 score lets you threshold and rank many models cheaply without N² comparisons;
  **pairwise wins** for open-ended quality (writing, tone, helpfulness) where humans and judges are far more
  reliable at "which is better" than at "score this 1-10", and where relative preference is exactly the question
  being asked.
- Arcturos takes both: reference-checked items need no judge at all; open-ended items are blind pairwise.

## 4. Reference-guided / deterministic checking

The doctrine across mature benchmarks is **objective checks where possible, judge where not**:

- [GSM8K](https://arxiv.org/abs/2110.14168) (8.5K grade-school word problems) and
  [MATH](https://arxiv.org/abs/2103.03874) (12.5K competition problems) grade by **exact match of the final answer**
  after light normalization — the reasoning path may vary, the terminal number/expression must be right.
- [IFEval](https://arxiv.org/abs/2311.07911) built ~500 prompts around **25 types of verifiable instructions**
  ("write in more than 400 words", "mention keyword X ≥3 times", "all lowercase", "exactly N bullet points") so
  instruction-following is graded by program, not opinion — directly the model for our `reference.check` specs.
- [Aider's polyglot benchmark](https://aider.chat/docs/leaderboards/) runs 225 Exercism exercises across six
  languages **in a sandbox and passes/fails them with unit tests** — execution is the gold standard for code.
  Arcturos v1 approximates this with contains/function-shape checks and judge review; sandbox execution is the v2 path.
- Consequence for suite design: every item should carry a `reference` check when its answer is objectively
  verifiable, and only the genuinely open-ended remainder goes to the LLM judge.

## 5. Multi-turn eval design

- [MT-Bench](https://arxiv.org/abs/2306.05685) is 80 two-turn questions (80 single + follow-up). The judge sees
  **the full conversation plus both candidates' turns** and is asked to focus on the latest turn — showing only the
  final exchange (no context) measurably degrades judgment because follow-ups are often uninterpretable alone.
- What breaks in multi-turn judging: context bloat pushes the judge toward position/verbosity biases; judging the
  *first* turn separately is cheap but misses drift; and turn-2 quality depends on turn-1 answers the model itself
  gave (state it built up). Humans preferred proprietary models *more* on turn 2 than turn 1 — multi-turn separates
  models that single-turn can't.
- Operationally: store one result per assistant turn (Arcturos already suffixes `#t2`, `#t3` in the runner), and
  judge the last turn with full history attached.

## 6. Multimodal / vision eval practice

- [MMMU](https://arxiv.org/abs/2311.16502) (college-level, multi-discipline) and
  [MathVista](https://arxiv.org/abs/2310.02255) (math reasoning in visual contexts) grade mostly by **exact/option
  match**, which keeps vision scoring deterministic — when the question has a closed answer, vision needs no judge.
- [MM-Vet](https://arxiv.org/abs/2308.02490) covers open-ended integrated capabilities and therefore uses an
  **LLM-based evaluator for open-ended outputs** — a multimodal judge reads image + question + both candidate
  answers and scores/compares them.
- Arcturos pattern (v1): vision items are engineered so answers are objective (counts, chart values,
  rendered digits, positions) and thus reference-checked; a multimodal judge mirrors §2 for any open-ended vision items later.
  Ground truth must be derivable **only** from the image, or the item leaks.

## 7. Statistical rigor

- Pairwise win rates are binomial proportions: the 95% CI is ≈ p̂ ± 1.96·√(p̂(1−p̂)/n); at p̂=0.5 that is ±5
  points at **n≈385 decisive battles** per model-pair comparison, and ±10 points at n≈96 — below ~100 wins+losses
  a "winner" is mostly noise. Prefer [Wilson score intervals](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)
  for small n (they don't go negative), and [bootstrap over battles](https://arxiv.org/abs/2403.04132) for BT/Elo-style
  aggregates, as Arena does.
- [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) is the field's statistics checklist: treat evals as
  experiments, report CIs, and plan sample size *before* running; ties are not missing data — they carry signal.
- **Tie rate as signal**: a high tie rate means the models are genuinely close *or* the judge is under-discriminating;
  MT-Bench counts inconsistent post-swap verdicts as ties. Track ties per dimension, and split them
  (win = w / (w + l)) for ranking while reporting the raw tie share separately.

## 8. Judge prompt engineering

- Give the judge **an explicit per-dimension rubric** ("judge only on code correctness, ignore verbosity") — MT-Bench
  showed judges otherwise apply a vague internal quality notion; rubric prompts are also how we make "which is better
  on dimension X" a per-dimension measurement instead of one mushy overall score.
- Return **structured verdicts** (JSON: `{"winner": "A|B|tie", "rationale": "..."}`) rather than free text or a bare
  word: machine-parseable, forces the judge to commit, and the extracted rationale gives the human reviewer an audit
  trail. MT-Bench measured raw format-**error rates** (up to ~79% for weak judges with score prompts) — a schema plus
  a repair/retry path is not optional.
- **Temperature 0** for judging (deterministic, re-runnable; Arcturos' bench replay already uses 0.0) and keep
  judge prompts versioned alongside suites so results stay comparable over time.

## Adopted for Arcturos v1

- **Blind pairwise judging as the core open-ended protocol** — the question is literally "is A better than B";
  candidates are presented anonymously (A/B) to the judge.
- **Judge both orders (A,B) and (B,A); disagree ⇒ tie** — GPT-4's swap consistency is only ~65%; the conservative
  rule converts position bias into visible ties instead of phantom wins.
- **Reference checks (`exact/numeric/contains/one_of/regex`) on every objectively verifiable item** — deterministic
  grading is free, reproducible, and judge-free (IFEval/GSM8K doctrine).
- **Per-dimension rubric lines inside every judge prompt** — makes "better on dimension X" measurable, not a single
  mushy quality score.
- **JSON structured verdicts with rationale + temperature 0** — parseable verdicts, committed judgments, auditable
  rationale, deterministic reruns.
- **Record output length (chars/tokens) for every answer and report length alongside win rate** — length bias is the
  largest systematic distortion (±5-9 points on AlpacaEval); v1 reports it, v2 can add LC-style correction.
- **Judge from a different family than both candidates when possible; store judge model + prompt version in run
  metadata** — self-preference bias is real (+10-25% win-rate inflation) and must be traceable.
- **Multi-turn: judge the final turn with the full conversation in context; store one result per assistant turn** —
  matches MT-Bench's finding that context-less turn-2 judging degrades; Arcturos' `#t2` suffixes already support it.
- **Vision items engineered for objective ground truth (counts, chart values, rendered digits)** — deterministic
  vision scoring now; multimodal judge only for open-ended vision later.
- **Report Wilson CIs on win rates, bootstrap CIs on aggregates, and raw tie rates per dimension** — a "winner"
  below ~100 decisive battles is noise; ties must be visible, not hidden.
- **Suites versioned and immutable; judge prompts versioned with them** — win rates are only comparable within a
  fixed suite + judge-prompt version.
