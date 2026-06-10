# Roadmap: from workshop-grade to a top-venue paper (and a PhD pitch)

**Honest current standing.** This repository is a complete, rigorous, *workshop-grade* contribution
and a strong master's thesis: a parameter-free cost-aware stopping rule, an honest negative (fixed
rules don't transfer across families), and a meta-learned forward-looking policy that generalizes
across tasks / unseen families / unseen cost levels and out-ranks the PBGI competitor (Xie et al.
2025) on its GP-BO substrate. It is **not** yet a NeurIPS/ICML main-conference paper: the novelty is
incremental relative to Xie 2025, the largest empirical wins are against weak baselines on random
search, and the benchmarks are surrogate. This document is the honest plan to close that gap — and,
framed differently, it is exactly a fundable **PhD opening project**.

## The one-sentence pitch
*"Fixed cost-aware stopping rules (incl. the 2025 SOTA) are myopic and benchmark-dependent; we learn
a forward-looking stopping policy that transfers across tasks, model families, and cost regimes, and
beats the SOTA stopping rule on its own ground."* — already supported in miniature; the roadmap makes
it airtight.

## Gap analysis (what a top reviewer will demand)

| Reviewer objection | Current state | What closes it |
|---|---|---|
| "Core idea is Xie 2025." | We out-rank PBGI on GP-BO (modest). | Beat **the authors' own code** on standard benchmarks; make the *forward-looking* axis the headline, not the reservation rule. |
| "Wins are vs weak baselines." | Big wins vs fixed/patience on RS. | Compare vs **ASHA/Hyperband, BO+stopping, learning-curve early-stop** on the same budget. |
| "Surrogate benchmarks only." | LCBench/rbv2 + a small live demo. | Add **PD1, HPOBench, real deep-learning HPO**; live-training at scale. |
| "Modest GP-BO magnitudes." | +0.6–4 pp, significant in rank. | Show the regime where it matters (slow/expensive search) and *quantify $ / GPU-hours saved*. |
| "No real theory." | Honest regret reduction (Prop. 4). | Prove the **concentration** step (held-out error → uniform δ w.h.p.) via a generalization bound. |

## Workstreams (each ~1–3 months of PhD-scale effort)

**W1 — Beat the real competitor.** Integrate Xie et al.'s released PBGI/LogEIPC stopping; run
head-to-head on standard benchmarks with their code and our META, reporting cost-to-target and
anytime regret-vs-cost AUC, with critical-difference diagrams over many tasks.

**W2 — Strong baselines & substrates.** ASHA/Hyperband (multi-fidelity), GP-BO and TPE, plus
learning-curve early stopping (Domhan-style). Show META as a *layer* that improves each.

**W3 — Harder, real benchmarks.** PD1 (real DL training curves), HPOBench, and a handful of *live*
deep-learning HPO runs (vision/NLP) to validate beyond surrogates and beyond classic ML.

**W4 — Stronger policy.** Replace one-shot regression with a **sequential model** (GRU over the
trace) or a **fitted-Q / optimal-stopping RL** treatment; ablate features; calibrate stopping risk.

**W5 — Theory.** Turn Prop. 4 into a high-probability regret bound: a generalization argument on the
learned `r̂` controlling the uniform error δ, hence regret, on unseen tasks/families. This is the
piece that would make the contribution *quotable*.

**W6 — Impact framing.** Translate savings into **GPU-hours / dollars / CO₂** on real workloads; ship
the Optuna/Ray-Tune plugin as a maintained OSS tool (adoption is its own form of impact).

## Minimal path to a *main-conference submission* (if forced to prioritize)
W1 + W2 + W3-lite (PD1) + W5. That is the smallest set that flips the three fatal objections
(competitor, baselines, benchmarks) and adds the theory that distinguishes it.

## Why this is a good PhD project (the application angle)
- A **working prototype + workshop paper + public repo** already in hand — rare for an applicant.
- A crisp, *open* research question with a clear SOTA threat already mapped (research maturity).
- A 6–18 month plan with independent workstreams (de-risked, parallelizable).
- Cross-cuts optimal stopping, meta-learning, AutoML, and systems/impact — fundable and broad.

**Use this document** in PhD applications / advisor emails as: *"Here is a complete prototype and an
honest assessment of what it takes to make it a main-conference result; this is the project I want to
do my PhD on."* That sentence, backed by this repo, is stronger than most applicants' published work.
