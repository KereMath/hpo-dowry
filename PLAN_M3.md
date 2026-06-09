# M3 — Meta-Learned Forward-Looking Stopping (the independent contribution)

## Why M3 (and why it matters after Xie et al. 2025)

The competitor Xie et al. 2025 ("Cost-aware Stopping for Bayesian Optimization") and our own
rule are both **myopic**: they take a *one-step* reservation-value decision (stop when the
expected gain of the *next* evaluation falls below its cost). Our experiments exposed the
intrinsic limit of myopia: in the **low-cost regime** the one-step rule slightly underperforms
the hindsight-optimal stop, because the truly optimal decision depends on *how the search will
continue for several more steps*, not just the next one.

**M3 replaces the myopic test with a learned, forward-looking stopping policy.** This is the
part of the project that is genuinely **independent of Xie 2025** (they do not learn a policy;
they use closed-form PBGI/LogEIPC), and it is the most defensible "own contribution" for the
thesis. Goal: **close the low-cost myopic gap without losing the high-cost wins**, with a
single policy that generalizes across tasks and cost levels.

---

## Problem formulation — imitation of the hindsight-optimal stop

For a trace (a single ordered HPO run) with scores `s_1..s_T` and costs `c_1..c_T`, and a cost
weight `λ`, the realized utility of stopping at step `t` is

    U(t) = max(s_1..s_t) − λ · (c_1 + ... + c_t).

The **hindsight-optimal stop** is `t* = argmax_t U(t)` (computable only after the fact). M3
learns a *causal* policy `π(features at step t) → {stop, continue}` that imitates stopping at
`t*`, using only information available *at* step `t`.

- **Label.** `y_t = 1` if `t ≥ t*` (we are at or past the optimal stop), else `0`.
  Deploy: stop at the first `t` with `π = 1`.
- **Alternative (stretch).** Regression target = expected remaining utility gain
  `max_{u>t} U(u) − U(t)`; stop when predicted gain ≤ 0. Captures magnitude, not just sign.

---

## Causal features at step t (all available online)

- search progress: `t`, `t / T_seen`, steps-since-last-improvement, #improvements so far;
- incumbent state: incumbent `b`, recent incumbent slope, last improvement magnitude;
- distribution: mean / std / IQR / high quantiles of observed scores, `(b − mean)/std`;
- **the myopic signals** (so the policy can *refine* the one-step rule, not discard it):
  EVT estimate `EI(b)`, next cost `c_next`, the gap `EI(b) − c_next`, ratio `EI(b)/c_next`;
- **cost context (critical for cross-λ generalization):** normalized cost `λ` in score units,
  `c_next / score_spread`, cumulative cost so far.

Including `λ` as a feature is what lets a **single** policy adapt across cost regimes — directly
attacking the "no static config is optimal across all λ" finding from M1.

---

## Meta-dataset construction

- Tasks: the 34 LCBench tasks (later +rbv2_* families for generality).
- Substrates: random-search orderings **and** BO (TPE) traces — so the policy is trained to be
  optimizer-agnostic.
- Cost levels: a grid of `λ` multipliers (e.g. ×0.5, 1, 2, 4, 8) per task.
- For each (task, substrate, λ, trace): compute `t*`, emit one feature row per step with label.
- Expected size: 34 tasks × ~hundreds of traces × ~tens of steps → ~10⁵–10⁶ rows. Subsample if needed.

---

## Model

- Start: **gradient-boosted trees** (`sklearn.HistGradientBoostingClassifier`) — robust,
  handles mixed scales, gives calibrated-ish probabilities, fast.
- Stretch: small sequential model (GRU over the step history) or a fitted-Q / LSTD treatment of
  the stopping MDP (optimal stopping is a canonical RL problem). Only if the GBT policy shows
  signal worth deepening.

---

## Evaluation (must avoid leakage)

- **Leave-one-task-out (LOTO):** train on 33 tasks, test on the held-out task; repeat for all 34.
  This is the honest meta-learning protocol — the policy must generalize to an *unseen* task.
- Metrics (same as the rest of the project, for comparability):
  - **(A)** matched-regret training-time saving vs the hindsight baseline envelope;
  - **(B)** normalized utility gap to the oracle;
  - per cost regime, aggregated across tasks, with **Wilcoxon** significance and win-rate.
- **Baselines to beat / match:** `evt + k_min=4` (our myopic M1), the best hindsight-tuned
  fixed/patience, and (once implemented) Xie 2025's PBGI/LogEIPC myopic stopping.
- **Success criterion:** M3 ≥ myopic at high cost **and** strictly closes the low-cost gap
  (the regime where myopia loses), on held-out tasks, significantly.

---

## Risks & kill-criteria

- **Leakage:** any per-task tuning or non-LOTO split invalidates the result. Enforce LOTO.
- **Label noise:** `t*` varies across orderings; average / use many traces so the policy learns
  the *expected* optimal, not a single lucky run.
- **Cross-λ / cross-task shift:** if the policy fails to generalize to held-out tasks, fall back
  to a *cost-aware myopic* rule (make `k_min` and the threshold a learned function of `λ`),
  which is a weaker but still-novel result.
- **No improvement over myopic:** if LOTO M3 does not beat `evt+k=4`, report it as a negative
  result (myopia is near-optimal here) — still thesis-worthy, and consistent with Xie 2025's
  Bayesian-optimality claim. The honest negative is a legitimate outcome.

---

## Deliverables

- `meta_stopping.py` — feature extraction, label computation, meta-dataset builder, GBT policy,
  LOTO evaluation, and a `MetaStopping` rule class compatible with `core.run_trace`.
- Updated `RESULTS.md` with the M3 LOTO comparison table.
- README "Results" + "Roadmap" sections updated with the M3 outcome (positive or negative).

## Milestones

1. **M3a** — meta-dataset builder (features + `t*` labels) over RS + BO traces, all 34 tasks.
2. **M3b** — train GBT policy + LOTO evaluation harness; produce the (A)/(B) comparison vs myopic.
3. **M3c** — verdict: does forward-looking learning close the low-cost gap? Decide stretch (RL/GRU)
   vs. lock the result.
