# Learning When to Stop: Cost-Aware, Optimizer-Agnostic, Forward-Looking Stopping for Hyperparameter Optimization

**Master's Thesis (draft)** · Repository: `github.com/KereMath/hpo-dowry`
Companion documents: `PAPER.md` (condensed), `THEORY.md` (proofs), `RESULTS.md` (raw logs).

---

## Abstract

Hyperparameter optimization (HPO) spends most of its cost on training models, yet the
decision of *when to stop searching* is usually left to a fixed budget or an ad-hoc patience
heuristic. This thesis casts HPO termination as a **cost-aware optimal-stopping problem** and
develops a stopping layer that plugs on top of any sequential optimizer. The base rule is a
parameter-free Pandora's-box reservation rule — stop when the expected improvement of one more
evaluation falls below its training cost — with the improvement tail estimated online by
**Extreme Value Theory (EVT)**. On 34 real LCBench tasks it saves 5–27 % of training time at
matched solution quality, across both random search and Bayesian optimization. We then show,
honestly, that this *fixed* rule is **benchmark-dependent**: it does not transfer from deep-
learning HPO to classic-ML model families. This motivates the central contribution: a
**meta-learned forward-looking stopping policy** that imitates the hindsight-optimal stop and,
unlike any fixed rule, **generalizes across tasks, across unseen model families, and across
unseen cost levels** (+17–31 % over the myopic rule, p < 10⁻⁶), while **out-ranking the recent
PBGI cost-aware stopping rule (Xie et al., 2025) on its own GP-BO substrate**. We give a regret
reduction bounding the policy's regret by its held-out prediction error, and ship the method as
a drop-in Optuna stopper.

---

## Chapter 1 — Introduction

### 1.1 Motivation
A sequential HPO procedure (random search, Bayesian optimization, evolutionary search) proposes
configurations one at a time; each is *expensive* because evaluating it means training and
validating a model. The practitioner keeps the best configuration found so far — the
**incumbent** — and faces a recurring economic question: *is it worth paying for one more
configuration, or should the search stop now?* In practice this question is answered by a fixed
trial budget chosen up front, or by a "stop if no improvement in `p` trials" patience rule. Both
are crude: a fixed budget wastes compute once the incumbent has effectively converged, and a
good patience value is itself unknown and problem-dependent.

### 1.2 The problem is optimal stopping — but not the secretary problem
Deciding when to stop a sequential, costly search with the option to keep the best result is a
classical **optimal-stopping** problem. The famous *secretary problem* is the wrong model for
HPO: it forbids recall (you cannot return to a passed-over candidate), uses only ordinal
information (better/worse, not the actual score), and maximizes the probability of selecting the
single best candidate. HPO has the opposite structure — full **recall** (every result is
stored), **cardinal** validation scores, and an objective that trades **expected quality against
cumulative cost**. The matching theory is **search theory / Pandora's box** (Weitzman, 1979),
whose optimal policy is a *reservation value*: keep searching while the incumbent is below a
threshold `z`, where `z` equates the expected improvement of a fresh draw with its cost.

### 1.3 Contributions
1. A **parameter-free, optimizer-agnostic** cost-aware stopping rule whose reservation threshold
   is derived from the cost, with the improvement tail estimated online by EVT
   (peaks-over-threshold / Generalized Pareto), plus a **cost-aware observation phase** that
   removes the fixed warm-up floor (Chapter 4).
2. An honest **negative result**: this fixed rule is benchmark-dependent and fails to transfer
   across model families (Chapter 7.5).
3. The central contribution: a **meta-learned forward-looking stopping policy** that imitates the
   hindsight-optimal stop and **generalizes across tasks, unseen model families, and unseen cost
   levels**, and **out-ranks the PBGI competitor (Xie et al., 2025) on its own substrate**.
4. A **regret reduction** (Chapter 5) bounding the learned policy's regret by its held-out
   prediction error, explaining the empirical generalization.
5. A reproducible evaluation on real surrogate benchmarks across three substrates with
   Friedman/Nemenyi/Wilcoxon rigor, and a **drop-in Optuna stopper** (Chapter 7.7).

### 1.4 Outline
Chapter 2 gives background on optimal stopping, HPO, and EVT. Chapter 3 surveys related work and
positions the thesis against the closest competitor. Chapter 4 develops the method. Chapter 5
states the theory. Chapter 6 describes the experimental setup. Chapter 7 reports results. Chapter
8 discusses limitations; Chapter 9 concludes.

---

## Chapter 2 — Background

### 2.1 Optimal stopping and the Pandora's box problem
An optimal-stopping problem asks, at each step of a stochastic process, whether to *stop* and
collect a reward or *continue* at a cost. Wald's sequential analysis (SPRT) and the theory of
Markov optimal stopping provide the general machinery: the optimal policy stops when the reward
of stopping exceeds the *continuation value*, the expected discounted value of acting optimally
thereafter. Weitzman's **Pandora's box** specializes this to search with recall: a set of boxes,
each with a known reward distribution and opening cost; open boxes (paying their cost), observe
rewards, and stop to claim the best opened reward. The optimal policy assigns each box a
*reservation value* (its Gittins-style index) and opens in decreasing order, stopping when the
best reward in hand exceeds every unopened box's reservation value. For identical i.i.d. boxes
all reservation values coincide at `z` with `E[(X−z)^+] = c`.

### 2.2 Hyperparameter optimization
HPO searches a configuration space to maximize validation performance. **Random search** samples
i.i.d.; **Bayesian optimization (BO)** fits a surrogate (Gaussian process or TPE density model)
and proposes the maximizer of an acquisition function; **multi-fidelity** methods (Hyperband,
ASHA, BOHB) allocate budget across cheap-to-expensive fidelities. Multi-fidelity attacks the
*inner-loop* question (how long to train a given configuration); this thesis attacks the
*outer-loop* question (how many configurations to evaluate), which is orthogonal and far less
studied.

### 2.3 Extreme value theory
To stop by comparing the expected improvement `EI(b)=E[(X−b)^+]` to a cost, one must estimate the
*upper tail* of the score distribution above the incumbent from few samples. The empirical
estimator is degenerate (no observed mass above the maximum). EVT's **peaks-over-threshold**
approach models exceedances over a high threshold by a Generalized Pareto Distribution
(Pickands–Balkema–de Haan), which we fit in closed form by Probability-Weighted Moments and
integrate to obtain `EI` analytically.

---

## Chapter 3 — Related Work

**Cost-aware stopping for BO.** *Xie et al. (2025), "Cost-aware Stopping for Bayesian
Optimization"* (arXiv:2507.12453, ICML 2026) is the closest prior work: a Pandora's-Box-Gittins-
Index (PBGI) reservation stopping rule that is cost-aware, parameter-free, and Bayesian-optimal
under independent evaluations. It is **myopic** (one-step) and tied to a **GP surrogate**. Our
work differs in (i) using EVT rather than a GP posterior, (ii) being **optimizer-agnostic**, and
(iii) replacing the myopic rule with a **forward-looking learned policy** that out-ranks PBGI on
its own substrate. *Xie et al. (2024, NeurIPS)* introduced PBGI as an acquisition function (which
configuration to try), not stopping.

**Automatic BO termination.** *Makarova et al. (2022, AutoML-Conf)* stop when optimization
suboptimality (a GP-confidence regret bound) is dominated by the validation/test estimation
error. *Ishibashi et al. (2023, AISTATS)* use a regret-gap criterion with an auto-tuned
threshold. *Wilson (2024, NeurIPS)* gives a probabilistic (ε,δ) regret-bound stopping rule. All
are GP-posterior-dependent and none is a cost-vs-improvement rule.

**Pandora's box with unknown distribution.** *Kalayci et al. (2025)* learn Weitzman thresholds
online for LLM Best-of-N inference (UCB, not EVT; not HPO).

**EVT in HPO.** Prior work uses GEV *block-maxima* to model the test metric for *configuration
selection*; we use POT/GPD on score *improvements* for *stopping*. EVT optimum-estimation is known
to be sampler-dependent (Bidlingmaier et al., 2022) — a caution we address empirically.

**Multi-fidelity.** Hyperband/ASHA/BOHB handle the budget via fidelity scheduling, orthogonal to
outer-loop termination.

---

## Chapter 4 — Method

### 4.1 Cost-aware reservation rule
With incumbent `b` and next cost `c` (in score units), stop when `EI(b)=E[(X−b)^+] ≤ c`. For
i.i.d. draws this equals the Pandora's-box reservation value. It is parameter-free given the cost.

### 4.2 EVT tail estimation
Model exceedances over a high quantile `u` by `GPD(ξ,β)`, fit by Probability-Weighted Moments
(closed form, robust for small samples), giving a closed-form `EI(b)`. A Gaussian-tail
alternative over-searches catastrophically on skewed landscapes (it never stops); EVT is the
robust default. A shape-shrinkage "guarded" variant was tested and *hurts* (over-conservative).

### 4.3 Cost-aware observation phase (M1)
Use only a small seed `k_min=4` to make the tail identifiable, then let `EI≤c` decide; the
effective observation length emerges from the cost. This halves the utility gap to the oracle
versus a fixed long warm-up floor.

### 4.4 Meta-learned forward-looking policy (META)
The myopic rule is one-step and loses when the search would keep improving (cheap-cost regime,
non-i.i.d. samplers, unfamiliar tail shapes). We learn a policy imitating the hindsight-optimal
stop `t* = argmax_t U(t)`, `U(t)=max(s_{1:t})−Σγ_{1:t}`. Regression target `t*−t`; stop when the
predicted steps-to-optimum `≤ 0`. Causal features: search progress, incumbent dynamics, observed-
score distribution statistics, the myopic signals (`EI`, `c`, gap/ratio), and **cost context**
(which lets one policy adapt across cost levels). The model is a gradient-boosted regressor.

---

## Chapter 5 — Theory (summary; full proofs in `THEORY.md`)

- **Theorem 1 (Weitzman).** Under i.i.d. scores, known distribution, constant cost, unbounded
  horizon, the myopic reservation rule is globally optimal. (Nothing to beat in the ideal case.)
- **Proposition 2.** Our setting breaks all four premises (finite horizon ⇒ a *time-varying*
  optimal threshold; unknown distribution; non-i.i.d. under BO; heterogeneous cost), each of
  which makes a single constant threshold sub-optimal.
- **Proposition 4 (regret reduction).** If the realized utility curve is `L`-Lipschitz in the
  step index and the policy's prediction error of `t*−t` is bounded by `δ`, then it stops within
  `⌈δ⌉` of `t*` and its regret is `≤ L⌈δ⌉` (and exactly `0` if `δ<1`). Hence the learned policy's
  regret is controlled by its **held-out prediction error** — which is precisely what the
  leave-one-task/family/cost-out experiments measure. A high-probability concentration of the
  average error into a uniform `δ` is left as future work.

---

## Chapter 6 — Experimental Setup

**Benchmarks.** YAHPO Gym surrogates: **LCBench** (34 OpenML datasets, Auto-PyTorch MLPs) and
**rbv2** (XGBoost, SVM, Random Forest — classic-ML families over many datasets). Each task is a
set of sampled configurations with real `(validation score, training time)`; training time is
genuine and heterogeneous.

**Substrates.** (i) Random search (i.i.d. orderings); (ii) Optuna **TPE** (Bayesian optimization,
adaptive — breaks i.i.d.); (iii) a **GP-BO** loop with PBGI acquisition (for the head-to-head).

**Metric.** *Matched-regret training-time saving*: at the regret a rule achieves, how much less
training time than the best **hindsight-tuned** fixed-N/patience baseline that reaches the same
regret. Aggregated across tasks; significance by Wilcoxon signed-rank, and Friedman + Nemenyi
critical-difference for multi-method ranking. Evaluation of the learned policy is **leave-one-
task-out (LOTO)**, **leave-one-family-out (LOFO)**, and **unseen-λ** — never testing on training
data.

---

## Chapter 7 — Results

**7.1 Random search (LCBench, 34 tasks).** Median saving 5.4/8.6/12.7/15.2 % at cost λ×1/2/4/8;
win-rate 88–100 %; Wilcoxon p ≤ 2×10⁻⁷.

**7.2 Bayesian optimization (TPE, LCBench).** 8.2/9.6/17.9/26.6 % saving; p from 1.3×10⁻³ to
2.4×10⁻⁸ — robust to the dominant real optimizer despite broken i.i.d.

**7.3 Observation-phase ablation (M1).** EVT + `k_min=4` halves the oracle-utility gap
(0.671→0.356); the guarded tail is worse (0.433) — a documented negative.

**7.4 Head-to-head vs PBGI (Xie 2025) on GP-BO (34 tasks, proper baseline).** Median saving:
XIE −5.6/−3.3/+0.5/0.0, MYOPIC −7.6/−4.3/−1.0/−1.0, **META +1.5/+2.0/+3.6/+4.0**. Aggregate (136
cells): Friedman p=4.2×10⁻⁴; ranks META 1.75 < XIE 2.05 < MYOPIC 2.20 (Nemenyi CD=0.284, both
gaps significant); pooled META>XIE p=1.1×10⁻⁴, META>MYOPIC p=1.8×10⁻⁵.

**7.5 Cross-family generalization (rbv2).** The fixed myopic rule **fails to transfer** (negative
on XGBoost/SVM). META rescues it: LOTO +19.7/+23.8/+27.3/+31.0 %, **LOFO** (unseen family)
+16.9/+21.1/+27.3/+30.4 %; p < 10⁻⁶ throughout.

**7.6 Unseen-cost generalization.** Trained on a subset of cost levels and deployed (LOTO) on
held-out ones, META extrapolates to higher (+23.6/+29.3 %) and lower (+23.6/+27.3 %) unseen cost;
p < 10⁻⁶. *Generalization triad — tasks, families, costs — all hold.*

**7.6b Cross-family on a BO substrate — a boundary condition.** Repeating the cross-family test on
a GP-BO (PBGI) substrate over the rbv2 families yields an essentially **null** result: both META and
the myopic rule save ≈ 0 % (LOTO/LOFO, no significant difference, p > 0.04). The reason is
interpretable: GP-BO over these low-dimensional classic-ML spaces **converges within a handful of
steps**, so the incumbent — and the hindsight-tuned baseline, and both stopping rules — all plateau
almost immediately; there is no headroom for a stopping rule to save anything. This sharpens the
scope of the contribution rather than contradicting it: cost-aware stopping (of any kind) pays off
where search is **slow or broad and evaluations expensive** (random search, and the deep-learning
LCBench HPO where the TPE result was +8–27 %), and is simply *harmless* (≈ 0, never negative for
META) where BO already converges in a few steps. The learned policy's cross-family *advantage*
manifests exactly in the high-headroom regime where stopping matters.

**7.7 Product.** An Optuna `EVTStopper` drop-in callback auto-terminates a study; on
RandomForest/digits it saves ~85–94 % wall-clock for ≤ 0.9 % accuracy, the single knob `frac`
tracing the quality-vs-cost curve.

---

## Chapter 8 — Discussion and Limitations

The fixed myopic rule is a strong, simple, optimizer-agnostic baseline on deep-learning HPO, but
its tail estimate is landscape-specific and it should **not** be deployed unconditionally across
model families — the cross-family failure makes this concrete. The meta-learned policy removes
this fragility by *learning* the stop from distributional and cost features, and the regret
reduction (Prop. 4) ties its success to a measurable held-out error. Limitations: on the GP-BO
substrate the absolute savings of all rules are modest (META wins in rank, small per cell); the
PBGI baseline is a faithful re-implementation, not the authors' code; the benchmarks are
surrogate (real but not live training, except the Optuna demo); and the regret bound assumes a
uniform prediction-error bound whose concentration is not proved.

---

## Chapter 9 — Conclusion and Future Work

Framing HPO termination as cost-aware optimal stopping yields a simple, parameter-free,
optimizer-agnostic rule that saves real training time — but no *fixed* rule transfers across
problem families. A meta-learned forward-looking stopping policy does: it generalizes across
tasks, unseen model families, and unseen cost regimes, and out-ranks a just-published cost-aware
stopping competitor on its own substrate. **Learning *when to stop* is both useful and
transferable; fixed reservation rules are not.** Future work: a high-probability regret bound via
a generalization argument on the policy; a sequential (RL/GRU) policy beyond one-shot regression;
validation on live training at scale; and integration with multi-fidelity inner-loop schedulers.

---

## References
Weitzman (1979); Wald (1947); Pickands (1975); Balkema–de Haan (1974); Hosking & Wallis (1987);
Xie et al. (2024, 2025); Makarova et al. (2022); Ishibashi et al. (2023); Wilson (2024); Kalayci
et al. (2025); Li et al. (Hyperband, 2018); Pfisterer et al. (YAHPO Gym); Zimmer et al. (LCBench);
Binder et al. (rbv2); Bidlingmaier et al. (2022).
