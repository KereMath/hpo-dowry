# Learning When to Stop: A Cost-Aware, Optimizer-Agnostic Forward-Looking Stopping Policy for Hyperparameter Optimization

*Draft — master's thesis / workshop paper. Numbers are from the experiments in this repository.*

## Abstract

Most hyperparameter-optimization (HPO) runs do not decide *when to stop*; they exhaust a fixed
budget, wasting training time once the incumbent has effectively converged. We frame HPO
termination as a cost-aware optimal-stopping problem (Pandora's box / Weitzman reservation value):
stop once the expected improvement of one more evaluation falls below its training cost. The crux
is estimating, online and from few observations, the upper tail of the score-improvement
distribution; we use peaks-over-threshold Extreme Value Theory (EVT). The resulting rule is
**parameter-free** (its threshold is derived from the cost) and **optimizer-agnostic** (it needs
only the stream of observed scores and costs, not a surrogate). On 34 real LCBench tasks it saves
5–27 % of training time at matched solution quality across both random-search and Bayesian-
optimization (TPE) substrates (Wilcoxon p < 10⁻⁶ and < 10⁻², respectively). However, we find this
*fixed* myopic rule is **benchmark-dependent**: it fails to transfer from deep-learning HPO to
classic-ML model families (negative savings on XGBoost and SVM). This motivates our central
contribution: a **meta-learned forward-looking stopping policy** that imitates the hindsight-optimal
stop. Evaluated leave-one-task-out and **leave-one-family-out**, it transfers across tasks *and*
unseen model families (+17–31 % vs. the myopic rule, p < 10⁻⁶), and **significantly out-ranks the
recent PBGI cost-aware stopping rule (Xie et al. 2025) on its own GP-BO substrate** (Friedman
p = 4×10⁻⁴; Nemenyi-significant). The learned policy thus succeeds precisely where fixed rules fail.

## 1. Introduction

A sequential HPO procedure evaluates configurations one at a time, each costing real training time,
keeping the best-so-far **incumbent**. Deciding *when to stop* is economically central yet usually
handled by a fixed budget or an ad-hoc patience heuristic. We treat it as **optimal stopping**.

The classic *secretary problem* is the wrong model for HPO (no recall, ordinal-only, maximize the
probability of the single best). HPO has **recall** (all results are kept), **cardinal** scores, and
a **quality-vs-cost** objective — exactly the **Pandora's box / Weitzman** setting, whose optimal
policy is a reservation-value rule: stop when the expected improvement `EI(b)=E[(X−b)^+]` of a fresh
draw falls below its cost `c`.

**Contributions.**
1. An EVT-based, parameter-free, optimizer-agnostic cost-aware stopping rule, and a *cost-aware
   observation phase* (M1) that removes the fixed warm-up floor.
2. An honest negative: this fixed myopic rule is family-dependent and fails to transfer beyond
   deep-learning HPO.
3. A **meta-learned forward-looking stopping policy** that imitates the hindsight-optimal stop,
   transfers across tasks and **unseen model families**, and out-ranks the recent PBGI cost-aware
   stopping rule on its own substrate.
4. A reproducible evaluation on real surrogate benchmarks (LCBench, rbv2) across random-search, TPE,
   and GP-BO substrates, with Friedman/Nemenyi/Wilcoxon rigor.

## 2. Related Work

- **Cost-aware stopping for BO — Xie et al. (2025), arXiv:2507.12453 (ICML 2026).** The closest
  work: Pandora's-Box-Gittins-Index (PBGI) reservation-value stopping, cost-aware, parameter-free,
  Bayesian-optimal under independent evaluations. It is **myopic** and **GP-surrogate-tied**. We
  reconstruct it as a baseline; our META policy out-ranks it on the GP-BO substrate, and unlike it
  our rule runs on non-GP samplers.
- **PBGI acquisition — Xie et al. (2024), NeurIPS.** Pandora's-box/Gittins index as a cost-aware
  *acquisition* function (which point to try), not stopping.
- **Automatic BO termination — Makarova et al. (2022, AutoML-Conf); Ishibashi et al. (2023,
  AISTATS); Wilson (2024, NeurIPS).** Regret/confidence-based GP-specific rules (suboptimality vs.
  estimation error; regret-gap; probabilistic (ε,δ) bounds). None is cost-vs-improvement, none uses
  EVT, none is optimizer-agnostic.
- **Pandora's-box stopping with unknown distribution — Kalayci et al. (2025).** Online Weitzman
  thresholds for LLM Best-of-N inference (UCB, not EVT; not HPO).
- **EVT for HPO.** GEV block-maxima for *configuration selection* (Procedia CS 2023); cautionary
  evidence that EVT optimum-estimation is sampler-dependent (Swarm Evol. Comput. 2022). We use
  peaks-over-threshold/GPD on score *improvements* for *stopping* — a distinct use.
- **Multi-fidelity — Hyperband/ASHA/BOHB.** Allocate budget across fidelities (an inner-loop,
  which-to-train-longer question); orthogonal to outer-loop termination.

## 3. Method

### 3.1 Cost-aware stopping as a reservation rule
With incumbent `b`, the expected gain of one more evaluation is `EI(b)=E[(X−b)^+]`, `X` the score of
a fresh configuration. Stop when `EI(b) ≤ c_next` (both in score units). For i.i.d. draws this myopic
one-step rule equals the Pandora's-box reservation value `z` solving `E[(X−z)^+]=c`. It is
**parameter-free given the cost**.

### 3.2 EVT tail estimation
The empirical `EI` is degenerate above the observed max (→0 → instant stop). We model the exceedances
of a high quantile by a Generalized Pareto Distribution, fit in closed form by Probability-Weighted
Moments, and use the closed-form `EI(b)=∫_b^∞ S(x)dx`. A Gaussian-tail alternative *catastrophically
over-searches* on skewed landscapes (never stops); EVT is the robust default.

### 3.3 Cost-aware observation phase (M1)
Rather than a fixed warm-up floor, use a small seed `k_min=4` so the tail is identifiable, then let
`EI≤c` decide; the effective observation length emerges from the cost. This halves the utility gap to
the oracle vs. a fixed `k=10` floor. A shape-shrinkage "guarded" tail was tried and **hurts**
(over-conservative) — a documented negative.

### 3.4 Meta-learned forward-looking policy (META)
The myopic rule is a one-step lookahead and is sub-optimal when the search would keep improving. We
learn a policy that imitates the **hindsight-optimal** stop `t* = argmax_t U(t)`, where
`U(t)=max(s_{1:t}) − Σc_{1:t}` (costs in score units). Regression target `t*−t` ("steps-to-optimum"),
stop when predicted `≤0`. Causal features at step `t`: search progress, incumbent state, observed-
score distribution statistics, the myopic signals (`EI`, `c_next`, their gap/ratio), and cost
context — the last lets a single policy adapt across cost levels. Model: gradient-boosted trees.

## 4. Theory (framing; no new theorem claimed)

- **Myopic optimality (known).** Under i.i.d. evaluations with recall and known distribution, the
  reservation rule `stop iff EI(b)≤c` is the optimal one-step policy and coincides with Weitzman's
  Pandora's-box solution; Xie et al. (2025) prove Bayesian-optimality of the analogous GP rule under
  independent evaluations. Our contribution at this layer is the *estimator* (EVT) and *agnosticism*,
  not a new optimality theorem.
- **Where myopia is loose.** When evaluations are cheap (small `c`), the optimal stop depends on the
  multi-step continuation value, not just the next draw; the one-step rule stops too early. The gap
  is exactly the low-cost regime where we measure the myopic rule losing.
- **META as finite-horizon optimal stopping.** Stopping is a Markov decision / optimal-stopping
  problem; the hindsight-optimal `t*` is the finite-horizon optimum on a realized trace. META is an
  imitation-learning approximation to the optimal stopping *policy* (the conditional expectation of
  the continuation value), trained across traces so it targets the *expected* optimum. This explains
  (i) why it beats the myopic rule in the low-cost regime, and (ii) why, being a learned function of
  distributional + cost features rather than a fixed closed form, it transfers across landscapes
  where a fixed rule (tuned implicitly to one tail shape) does not. A formal regret bound for the
  learned policy vs. the oracle is left as future work.

## 5. Experiments

**Setup.** YAHPO Gym surrogates: LCBench (34 OpenML datasets, Auto-PyTorch MLPs) and rbv2
(XGBoost/SVM/Random-Forest, classic ML), each task = sampled configs with real `(score, training-
time)`. Substrates: random search, Optuna TPE, and a GP-BO loop with PBGI. Primary metric:
**matched-regret training-time saving** vs. a hindsight-tuned fixed-N/patience envelope; significance
by Wilcoxon, and Friedman + Nemenyi for multi-method ranking.

**5.1 Random search, LCBench (34 tasks).** Median saving 5.4/8.6/12.7/15.2 % at cost λ×1/2/4/8;
win-rate 88–100 %; Wilcoxon p ≤ 2×10⁻⁷.

**5.2 Bayesian optimization (TPE), LCBench.** Despite broken i.i.d., 8.2/9.6/17.9/26.6 % saving;
p from 1.3×10⁻³ to 2.4×10⁻⁸.

**5.3 M1 ablation.** EVT + `k_min=4` halves the oracle-utility gap (0.671→0.356); the guarded tail
is worse (0.433).

**5.4 Head-to-head vs. PBGI (Xie 2025) on GP-BO (34 tasks), proper GP-BO baseline.** Median saving —
XIE −5.6/−3.3/+0.5/0.0 %, MYOPIC −7.6/−4.3/−1.0/−1.0 %, **META +1.5/+2.0/+3.6/+4.0 %**. Aggregate
(136 cells): Friedman χ²=15.57, p=4.2×10⁻⁴; ranks META 1.75 < XIE 2.05 < MYOPIC 2.20, Nemenyi
CD=0.284 (both gaps significant); pooled META>XIE p=1.1×10⁻⁴, META>MYOPIC p=1.8×10⁻⁵.

**5.5 Cross-family generalization (rbv2).** The fixed myopic rule **fails to transfer** (negative on
XGBoost/SVM). META rescues it: leave-one-task-out +19.7/+23.8/+27.3/+31.0 %, and **leave-one-family-
out** (unseen family) +16.9/+21.1/+27.3/+30.4 %, vs. myopic ≤ +6.1 %; p < 10⁻⁶ throughout.

## 6. Limitations

- On the GP-BO substrate the absolute savings of all rules are modest; META's win is significant in
  rank but small per cell, strongest in the mid-cost regime.
- The PBGI baseline is a faithful re-implementation, not the authors' code.
- EVT tail estimation is landscape-dependent (the motivation for META); the myopic rule should not be
  deployed unconditionally across model families.
- No formal regret bound for META; evaluation is on surrogate benchmarks (real but not live training).

## 7. Conclusion

Framing HPO termination as cost-aware optimal stopping yields a simple, parameter-free,
optimizer-agnostic rule that saves real training time — but a *fixed* rule does not transfer across
problem families. A meta-learned forward-looking stopping policy does: it generalizes across tasks
and unseen model families and out-ranks a just-published cost-aware stopping competitor on its own
substrate. Learning *when to stop* is both useful and transferable; fixed reservation rules are not.

## References
See `README.md` §10 (Xie 2024/2025, Makarova 2022, Ishibashi 2023, Wilson 2024, Kalayci 2025,
Weitzman 1979, YAHPO Gym / LCBench / rbv2).
