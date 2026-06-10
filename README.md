# hpo-dowry — A Cost-Aware, Optimizer-Agnostic Optimal-Stopping Layer for HPO

> **One line.** Most hyperparameter-optimization (HPO) runs do not know *when to stop*:
> they burn a fixed budget. This project builds a **parameter-free, cost-aware stopping
> layer** that plugs on top of *any* sequential optimizer (random search, Bayesian
> optimization) and halts the search the moment the *expected improvement of one more
> evaluation falls below its training cost* — estimating that improvement tail with
> **Extreme Value Theory (EVT)**. On 34 real HPO tasks it saves **5–27 % of training time
> at matched solution quality**, with strong statistical significance, and **without any
> per-task tuning of the stopper itself**.

This README is the full, honest record of what was built, what the experiments show,
how the idea sits in the literature (including a recently-published near-competitor), and
how the work proceeds toward a thesis and/or a product.

---

## ⚠️ Status & open work (required to reach a top-tier venue)

Current standing: **complete master's-thesis / workshop-grade** work (working method, honest
negative, a meta-learned policy that generalizes and out-ranks the PBGI competitor, theory,
product, live validation). It is **not** yet a NeurIPS/ICML main-conference paper. The four
things that must still be done — see [`ROADMAP_TOPVENUE.md`](ROADMAP_TOPVENUE.md) for the full plan:

- [ ] **1. Beat the real competitor.** Run **Xie et al. (2025)'s released PBGI/LogEIPC code** head-to-head
      on **standard benchmarks (HPOBench, PD1, real deep-learning HPO)** — not our re-implementation.
- [ ] **2. Strong baselines.** Compare against **ASHA / Hyperband (multi-fidelity) + BO + learned
      learning-curve early stopping**, not just fixed-budget / patience.
- [ ] **3. Stronger policy + real theory.** A **sequential / RL** stopping policy (beyond one-shot
      regression), and the **concentration step of Prop. 4** (held-out error → uniform δ w.h.p., a real
      generalization theorem).
- [ ] **4. Live-training at scale.** Validate on **real (non-surrogate) deep-learning training**, beyond
      the small sklearn demo.

These are PhD-scale workstreams; this repo is their de-risked foundation.

### Honest competitive standing (what we beat, what we didn't)

- **vs fixed-budget / patience (common defaults):** clearly ahead — 5–27 % time saved at matched
  quality, p < 10⁻⁶. *(But these are weak baselines.)*
- **vs Xie 2025 PBGI (the SOTA cost-aware stopping rule):** our *myopic* rule does **not** beat PBGI
  on its GP-BO turf; our *META* policy **out-ranks** PBGI significantly in aggregate rank (Friedman
  p = 4×10⁻⁴, Nemenyi-significant) but by a **modest** per-cell margin — and against our
  **re-implementation**, not the authors' code (open-work #1).
- **vs Hyperband / ASHA / BO (dominant HPO methods):** **not compared** — these are multi-fidelity /
  proposal methods; our stopping rule is a *layer on top*, not a competitor. Untested (open-work #2).
- **The genuine, defensible edge:** META **generalizes where fixed rules (incl. SOTA-style myopic
  ones) fail** — across tasks, unseen model families, and unseen cost levels — a *robustness/transfer*
  advantage, not a raw-speed one.

**Honest headline:** *"Fixed cost-aware stopping rules (incl. 2025 SOTA) are myopic and
benchmark-dependent; we offer a learned policy that transfers."* We have **not** shown "beats SOTA on
every metric," and should not claim it; closing open-work #1–#2 is what would let us say "ahead" cleanly.

---

---

## 1. Motivation

A sequential HPO run evaluates configurations one at a time; each evaluation costs real
training time. You keep the best-so-far **incumbent** `b`. The economic question is:

> *Is it worth paying for one more evaluation, or should I stop now?*

This is a classic **optimal-stopping** problem. The textbook *secretary problem* is the
wrong variant for HPO (it assumes no recall, ordinal-only information, and maximizes the
probability of picking the single best). HPO instead has **recall** (you keep all results),
**cardinal** scores (you see the actual validation metric), and you care about the
**expected quality vs. cost trade-off**. The right machinery is **search theory /
Pandora's Box (Weitzman's reservation value)**:

- Expected improvement of one more draw: `EI(b) = E[(X - b)^+]` where `X` is the score of
  a fresh configuration.
- **Stopping rule:** stop as soon as `EI(b) ≤ c_next`, the cost of the next evaluation
  (both in the same "score units"). For i.i.d. draws this myopic one-step rule coincides
  with the Pandora's-box reservation value `z` solving `E[(X - z)^+] = c`.

It is **parameter-free given the cost**: the threshold is *derived* from the data and the
cost, not hand-tuned like a `patience` value.

**The crux** is estimating the *upper tail* of the score distribution above the current
incumbent from only a handful of observations. The naive empirical estimator is degenerate
(no mass above the observed max → `EI = 0` → stop immediately). This is where **Extreme
Value Theory** enters: we model the tail with a **peaks-over-threshold / Generalized Pareto
Distribution (GPD)**, fit online by closed-form **Probability-Weighted Moments** (fast
enough to run inside the loop).

---

## 2. Method

For an incumbent `b`, observed scores `S`, and per-evaluation cost `c_next`:

1. **Seed (cost-aware observation phase).** Evaluate a small seed of `k_min = 4` configs so
   the tail model is identifiable. There is *no* fixed long observation floor: the effective
   observation length **emerges from the cost** — it shortens automatically when evaluations
   are expensive and lengthens when they are cheap.
2. **Estimate the improvement tail.** Fit a GPD to the exceedances of `S` over a high
   quantile (peaks-over-threshold), via closed-form PWM. Compute
   `EI(b) = ∫_b^∞ P(X > x) dx` in closed form.
3. **Stop test.** Stop when `EI(b) ≤ c_next`.

The cost knob `λ` converts seconds → score units; the utility being optimized is
`U = best_accuracy − λ · total_training_time`.

**Tail models implemented** (`core.py`): `empirical` (naive baseline, fails),
`gaussian` (simple parametric — *blows up* on skewed landscapes), `evt` (GPD/PWM — the
robust default), `guarded` (EVT + shape-shrinkage — an ablation that turned out to be
*over-conservative*; see results).

**Baselines** (`core.py`): `FixedBudget(N)`, `Patience(p)`, and an `OracleMarginal` gold
standard that knows the true score distribution (upper bound on achievable utility).

---

## 3. What is in this repo

| File | Purpose |
|---|---|
| `core.py` | Tail models, stopping rules (`MarginalStopping`, `FixedBudget`, `Patience`), trace simulator, reservation-value solver. |
| `benchmark.py` | Standardized loader over **YAHPO Gym** surrogate benchmarks → `task → (scores, costs)`. Caches landscapes. |
| `synthetic.py` | Synthetic validation on known distributions; oracle comparison; adaptivity claim. |
| `real_hpo.py` | Proof-of-concept on real **sklearn** RandomForest landscapes (3 datasets), with real CV scores + fit times. |
| `pareto.py` | λ-swept quality-vs-cost frontier vs. the hindsight-tuned baseline envelope. |
| `experiment_suite.py` | **Suite-level statistical verdict** across all 34 LCBench tasks (win-rate + Wilcoxon). |
| `experiment_m1.py` | M1 ablation: cost-aware observation phase + tail hardening (`evt,k=4` vs `guarded` vs old). |
| `bo_substrate.py` | **Phase 2-ext**: runs the stopper on top of a real **Optuna TPE (Bayesian optimization)** substrate. |
| `meta_stopping.py` | **M3**: meta-learned forward-looking policy (target `t*−t`), LOTO evaluation. |
| `xie_baseline.py` | Faithful **Xie 2025 PBGI** GP-BO loop + PBGI stopping (the competitor). |
| `decisive_m3_vs_xie.py` | Head-to-head META vs PBGI vs myopic on the GP-BO substrate. |
| `cross_family.py`, `cross_family_m3.py` | Cross-family (rbv2) generalization; LOTO + leave-one-family-out. |
| `rigor.py` | Friedman / Nemenyi critical-difference / pooled Wilcoxon + figures. |
| `optuna_plugin.py` | **Product**: drop-in `EVTStopper` / `MetaStopper` callback for Optuna + demo. |
| `PAPER.md` | Paper/thesis-style consolidation (abstract, method, theory, experiments). |
| `PLAN.md`, `PLAN_M3.md` | Phased research/product plan and the M3 plan. |
| `RESULTS.md` | Live results log (raw numbers). |
| `USPexperiment.py` | The original toy script that started this line of thinking (a brute-force secretary-problem-with-threshold experiment). Kept for provenance. |

**Product usage:**
```python
import optuna
from optuna_plugin import EVTStopper
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500, callbacks=[EVTStopper(frac=0.05)])  # auto-stops
```
Demo (`python optuna_plugin.py`, RandomForest/digits): **~85–94 % less wall-clock for ≤ 0.9 %
accuracy lost**, the single knob `frac` tracing the quality-vs-cost operating curve.

Caches (`cache/`, `landscapes.npz`) and the YAHPO data (`c:/tmp/yahpo_data`, ~500 MB) are
*not* committed.

---

## 4. Experimental setup

- **Benchmark.** YAHPO Gym `lcbench` — Auto-PyTorch MLPs over **34 real OpenML datasets**.
  Each task = 400 sampled configurations evaluated at full fidelity, giving
  `(val_accuracy, training_time)`. The training time is genuine and **heterogeneous**
  (median per-config cost ranges from ~42 s to ~2339 s across tasks).
- **Substrates.** (a) Random search = random orderings of a task's landscape;
  (b) Bayesian optimization = Optuna **TPE** proposing configs over the real YAHPO space
  (15 seeds/task), the adaptive setting where the i.i.d. assumption is *violated*.
- **Metric (primary).** *Matched-regret training-time saving*: at the regret the stopper
  achieves, how much **less training time** than the best **hindsight-tuned** baseline
  (best `FixedBudget`/`Patience`) that reaches the same regret. Aggregated across tasks,
  with **Wilcoxon signed-rank** significance and win-rate.
- **Metric (secondary).** *Cost-aware utility gap to the oracle*, normalized by score spread.

---

## 5. Results

### 5.1 Proof-of-concept (sklearn, 3 datasets)
- The optimal-stopping rule beats untuned defaults (patience/fixed) by large margins.
- **`gaussian` tail catastrophically fails** on the skewed `digits` landscape (never stops,
  95 % of budget) → **tail choice is safety-critical**; EVT is robust.

### 5.2 Suite verdict — Random Search, 34 LCBench tasks (MS2 go/no-go: **PASSED**)
Matched-regret training-time saving of `opt-evt` (parameter-free) vs. the hindsight baseline
envelope:

| cost λ× | median saving | win-rate | Wilcoxon p |
|---|---|---|---|
| 1.0 | 5.4 % | 88 % | 2.0 × 10⁻⁷ |
| 2.0 | 8.6 % | 94 % | 5.8 × 10⁻¹⁰ |
| 4.0 | 12.7 % | 100 % | 1.2 × 10⁻¹⁰ |
| 8.0 | 15.2 % | 100 % | 1.2 × 10⁻¹⁰ |

Even against a *per-task hindsight-tuned* envelope, the parameter-free rule is more
time-efficient at matched quality, significantly, on essentially every task.

### 5.3 M1 — cost-aware observation phase + tail hardening
- **Winner: `evt` + small seed `k_min = 4`.** Roughly **halves** the normalized
  utility gap to the oracle (0.671 → **0.356**), best/tied at every cost level; fixes the
  high-cost regime where the old fixed `k=10` floor was wasteful.
- **Negative result (ablation): the `guarded` shape-shrinkage tail back-fired** (0.433) —
  it is over-conservative and stops too early when evaluations are cheap. Since EVT did not
  blow up, the extra shrinkage only hurts. (A clean documented negative for the thesis.)
- **Residual limitation:** in the *low-cost* regime the small seed slightly underperforms on
  matched-regret — an intrinsic limit of the *myopic* one-step rule, motivating M3 below.

### 5.4 Phase 2-ext — Bayesian-optimization substrate (Optuna TPE), 34 tasks
The critical stress test: BO samples adaptively, breaking the i.i.d. assumption. The stopper
**still** saves time at matched quality, and *more* at high cost (BO converges fast → the
incumbent plateaus early → the stopper exploits the plateau):

| cost λ× | median saving | win-rate | Wilcoxon p |
|---|---|---|---|
| 1.0 | 8.2 % | 79 % | 1.3 × 10⁻³ |
| 2.0 | 9.6 % | 71 % | 4.5 × 10⁻² |
| 4.0 | 17.9 % | 91 % | 3.9 × 10⁻⁵ |
| 8.0 | 26.6 % | 94 % | 2.4 × 10⁻⁸ |

**Takeaway:** the layer is robust to the dominant real optimizer — closing the single most
important reviewer objection ("does it work on real BO, not just random search?").

### 5.5 M3 — meta-learned forward-looking stopping (the independent contribution)
The myopic one-step rule (and Xie 2025) leave value on the table in the *low-cost* regime. M3
trains a policy to imitate the **hindsight-optimal** stop (regression target `t* − t`,
"steps-to-optimum"; stop when predicted `≤ 0`), on a 523k-row meta-dataset built from random
-search *and* BO traces across the 34 tasks and 5 cost levels. Evaluated **leave-one-task-out**
(generalizing to *unseen* tasks). Matched-regret time saving — META vs. the MYOPIC `evt,k=4` rule:

| cost λ× | META (forward-looking) | MYOPIC | paired Wilcoxon p |
|---|---|---|---|
| 0.5 | **+14.6 %** (94 % win) | −11.0 % (24 %) | 1.6 × 10⁻⁹ |
| 1.0 | **+17.8 %** (97 %) | −0.8 % (41 %) | 1.7 × 10⁻⁷ |
| 2.0 | **+23.6 %** (97 %) | +1.8 % (56 %) | 9.0 × 10⁻⁶ |
| 4.0 | **+32.2 %** (100 %) | +11.2 % (76 %) | 6.4 × 10⁻⁹ |
| 8.0 | **+33.1 %** (100 %) | +27.0 % (76 %) | 5.8 × 10⁻¹⁰ |

The low-cost regime where the myopic rule *lost* (−11 %, −0.8 %) becomes a strong win; META beats
MYOPIC at every cost level (p < 10⁻⁵), on held-out tasks. **This forward-looking axis is exactly
what Xie 2025 (myopic) does not have**, and is the thesis's primary own contribution. (Remaining
hardening: held-out BO traces, generalization to unseen λ, more model families.)

### 5.6 Decisive head-to-head vs. the competitor (Xie 2025 PBGI), GP-BO substrate
A faithful GP-BO loop with **Pandora's-Box Gittins-Index** acquisition + PBGI stopping
(`xie_baseline.py`): stop when `max_x g(x) ≤ incumbent`, `g` the reservation value under the GP
posterior. We compare three stopping rules against a **proper baseline computed on the GP-BO
traces** (fixed-N / patience over the selection order) — `decisive_m3_vs_xie.py`. Matched-regret
training-time saving, **all 34 tasks**, median (win-rate):

| cost λ× | XIE (PBGI) | MYOPIC (EVT, ours) | **META (M3, LOTO)** | META>XIE (paired p) |
|---|---|---|---|---|
| 1.0 | −5.6 % (35 %) | −7.6 % (29 %) | **+1.5 % (53 %)** | 0.125 (n.s.) |
| 2.0 | −3.3 % (44 %) | −4.3 % (38 %) | **+2.0 % (68 %)** | 0.024 |
| 4.0 | +0.5 % (53 %) | −1.0 % (50 %) | **+3.6 % (82 %)** | 0.010 |
| 8.0 | 0.0 % (50 %) | −1.0 % (41 %) | **+4.0 % (74 %)** | 0.122 (n.s.) |

**Aggregate rigor** (`rigor.py`, 136 task×cost cells): Friedman χ²=15.57, **p=4.2×10⁻⁴**;
average ranks (1=best) **META 1.75** < XIE 2.05 < MYOPIC 2.20, with Nemenyi **CD=0.284** —
both gaps (META–XIE 0.298, META–MYOPIC 0.452) exceed CD, so **META significantly outranks both**.
Pooled paired Wilcoxon: META > XIE **p=1.1×10⁻⁴**, META > MYOPIC **p=1.8×10⁻⁵**.
(Figures: `figures/avg_ranks.png`, `figures/saving_by_cost.png`.)

**Honest verdict.** Against a *properly-tuned* GP-BO-trace baseline, PBGI and our myopic EVT rule
are both roughly break-even (PBGI even negative at low cost); **only the forward-looking META policy
is consistently positive, and it significantly out-ranks both PBGI and the myopic rule** on the
GP-BO substrate (Friedman + Nemenyi + pooled Wilcoxon). Per-cost magnitudes are **modest**
(median +0.6 to +4 pp) and per-cell significance holds firmly only in the mid-cost regime; the
**large** wins remain in the optimizer-agnostic regime (random search, §5.5: +15–33 % vs myopic,
where PBGI is undefined). META was trained on random-search + TPE traces and tested
**leave-one-task-out on GP-BO traces** — generalizing across *both* tasks and samplers.

> An earlier 12-task version of this table used a random-search baseline and reported inflated
> 60–69 % savings; that was a methodological artifact, replaced by the correctly-baselined 34-task
> numbers above.

**Caveats (kept explicit):** the PBGI baseline is our faithful *re-implementation*, not the
authors' code; GP-BO magnitudes are modest; the strongest evidence is the rank-aggregate, not any
single cost cell. Robust claim: *a meta-learned forward-looking stopping policy significantly
out-ranks the myopic PBGI rule on its own GP-BO substrate and beats the myopic EVT rule by 15–33 %
on optimizer settings where PBGI is undefined.*

---

### 5.7 Cross-family generalization — the myopic rule fails, the learned policy rescues it
We rebuilt landscapes on three **classic-ML families** from `rbv2_*` (XGBoost, SVM, Random Forest;
12 tasks each, real `timetrain` cost) — `cross_family.py`, `cross_family_m3.py`.

**The fixed myopic EVT rule does NOT transfer beyond LCBench MLPs.** Matched-regret saving is
positive on Random Forest but **negative on XGBoost and SVM** (e.g. −24 % / −22 % at low cost),
overall break-even-to-negative. An honest, scope-defining limitation: the fixed rule is
landscape/family-dependent.

**The meta-learned forward-looking policy (META) rescues generalization** — even to an entirely
unseen *family*:

| | cost λ× | META | MYOPIC | META > MYOPIC p |
|---|---|---|---|---|
| **leave-one-task-out** | 1.0 / 4.0 / 8.0 | +19.7 % / +27.3 % / +31.0 % | −5.5 % / −1.3 % / +6.1 % | 5.4×10⁻⁸ … 6.5×10⁻⁹ |
| **leave-one-FAMILY-out** | 1.0 / 4.0 / 8.0 | +16.9 % / +27.3 % / +30.4 % | (same myopic) | 6.3×10⁻¹⁰ … 2.0×10⁻⁹ |

Trained on two families, deployed on the held-out third (**LOFO**), META still saves **17–31 %** at
matched quality (p < 10⁻⁶ at every cost), ≈ its leave-one-task-out performance — so it transfers
across *model families*, not just tasks. **This turns the myopic rule's failure into the motivation
for the learned approach, and makes META the central contribution.**

## 6. Where this sits in the literature (honest novelty assessment)

A verified multi-source literature review (21 primary sources, 25 adversarially-checked
claims) produced a **qualified-novel** verdict. The individual ingredients are **all prior
art**, and one very recent paper overlaps heavily:

- **⚠ Closest competitor — Xie et al. 2025, "Cost-aware Stopping for Bayesian Optimization"**
  (arXiv:2507.12453, ICML 2026). Already unites **Pandora's-box reservation-value stopping +
  heterogeneous cost + parameter-free ("no heuristic tuning")**, proven Bayesian-optimal in
  the independent-evaluation case, grounded in the Pandora's-Box Gittins Index (PBGI) /
  LogEIPC. This is the **primary novelty threat** and must be engaged head-on.
- **Xie et al. 2024 (NeurIPS)** — Pandora's-Box/Gittins-index as a cost-aware *acquisition
  function* (which config to try next), not stopping.
- **Kalayci et al. 2025** — distribution-free online Pandora's-box stopping with thresholds
  learned on the fly, but for **LLM Best-of-N inference**, not HPO; uses UCB, not EVT.
- **Makarova et al. 2022 (AutoML-Conf)** — the canonical HPO automatic-termination rule, but
  a *regret-vs-estimation-error* criterion (validation/test gap), GP-specific, no cost, no EVT.
- **Ishibashi et al. 2023 (AISTATS)**, **Wilson 2024 (NeurIPS)** — BO stopping via
  regret-gap / probabilistic (ε,δ) bounds; GP-posterior-dependent; not cost-vs-improvement.
- **EVT-for-HPO** exists but via **GEV block-maxima for configuration selection**, *not*
  peaks-over-threshold on score *improvements* for a *stopping* decision.
- **Caution (Swarm Evol. Comput. 2022):** EVT estimates of the optimum can be unreliable and
  depend on the sampler — a transferable risk we must address (our RS+BO robustness is
  evidence against the pessimism, but it must be argued explicitly).

### Defensible, currently-unclaimed contributions
1. **EVT (POT/GPD) online estimation of the improvement tail for the stopping decision** —
   no surveyed stopping rule does this.
2. **An optimizer-AGNOSTIC layer** validated on **random search *and* BO simultaneously** —
   competitors are tied to a GP surrogate or a specific cost-aware acquisition function.
3. **(Planned, M3) a meta-learned forward-looking stopping policy** — beyond the myopic
   one-step rule that all the above (including Xie 2025) effectively use.

---

## 7. Honest outlook

- **"New optimizer that beats SOTA":** not the goal, and not realistic — a stopping layer
  does not choose configs; BO + multi-fidelity own that axis.
- **"Beat the just-published cost-aware stopping SOTA (Xie 2025) head-to-head":** the *myopic*
  EVT rule does **not** beat PBGI on its GP-BO turf. But the **forward-looking META policy
  does** — modestly and significantly at low/medium cost on the GP-BO substrate (§5.6), while
  also running where PBGI is undefined (random search, TPE). This is the strongest card: a
  *different axis* (learned, forward-looking) rather than a variant of the same myopic rule.
  Still to harden: all 34 tasks for high-cost power, more benchmarks, and the authors' own code.
- **Master's thesis:** **viable and well-supported, with a clear central result.** The arc is:
  (i) a parameter-free cost-aware EVT stopping rule that works and is optimizer-agnostic, but
  (ii) is *benchmark/family-dependent* (fails to transfer from LCBench MLPs to XGBoost/SVM) —
  which (iii) motivates the **meta-learned forward-looking policy (META)**, the central
  contribution, which **transfers across tasks and across model families (leave-one-family-out,
  +17–31 % vs the myopic rule, p < 10⁻⁶)** and **significantly out-ranks the competitor's PBGI
  rule on its own GP-BO substrate** (Friedman p = 4×10⁻⁴). This is a self-contained, honest,
  publishable-grade story (workshop → possibly main conference), with documented negatives and a
  competitor baseline engaged head-on.
- **Product:** a drop-in, optimizer-agnostic, zero-tuning **stopper plugin** (Optuna
  `StudyStopper` / Ray Tune `Stopper`) that cuts cloud training cost — a credible OSS artifact
  regardless of the publication outcome.

---

## 8. Roadmap

- **M3 (highest upside, most independent of Xie 2025):** meta-learned *forward-looking*
  stopping. Build a meta-dataset of (online features → hindsight-optimal stop) across tasks
  and cost levels; train a policy (gradient-boosted / small sequential model) evaluated
  **leave-one-task-out**. Target the low-cost regime where the myopic rule loses.
- **Engage the competitor:** implement Xie 2025's PBGI/LogEIPC stopping as a baseline and
  identify where EVT + agnosticism wins (RS, non-GP, heavy tails).
- **Broaden benchmarks:** more YAHPO scenarios (rbv2_* → SVM/RF/XGBoost/glmnet = multiple
  model families, hundreds of tasks) for statistical power and cross-family generality.
- **Theory:** competitive-ratio / regret of the EVT-based myopic rule vs. the oracle; PAC-style
  bound from tail-estimation error.
- **Rigor:** anytime regret-vs-cost AUC, critical-difference (Nemenyi) diagrams, error bars.
- **Product track:** package the stopper as an Optuna/Ray-Tune plugin + a cloud-cost-saving report.

Milestones, kill-criteria, and per-phase "done" definitions are in [`PLAN.md`](PLAN.md);
the running numbers are in [`RESULTS.md`](RESULTS.md).

---

## 9. Reproducing

```bash
pip install numpy scipy scikit-learn optuna yahpo-gym
# YAHPO surrogate data (~500 MB), then point benchmark.py:DATA_PATH at it:
git clone --depth 1 https://github.com/slds-lmu/yahpo_data.git

python synthetic.py            # synthetic validation
python real_hpo.py             # sklearn proof-of-concept (trains real models)
python benchmark.py lcbench    # build + cache the 34 real landscapes
python experiment_suite.py     # suite-level statistical verdict (random search)
python experiment_m1.py        # M1 ablation
python bo_substrate.py gen     # generate BO traces (slow)
python bo_substrate.py analyze # BO-substrate verdict
```

Requires Python 3.10+. The YAHPO data path is set in `benchmark.py` (`DATA_PATH`).

---

## 10. Key references

- Xie, Scully, Terenin, Frazier et al. *Cost-aware Stopping for Bayesian Optimization*,
  arXiv:2507.12453 (ICML 2026). — **primary related work / baseline.**
- Xie et al. *Cost-aware Bayesian Optimization via the Pandora's Box Gittins Index*,
  NeurIPS 2024 (arXiv:2406.20062).
- Kalayci et al. *Pandora's-box inference-time stopping*, arXiv:2510.01394 (2025).
- Makarova et al. *Automatic Termination for Hyperparameter Optimization*, AutoML-Conf 2022
  (arXiv:2104.08166).
- Ishibashi et al. *A stopping criterion for BO by the gap of expected minimum simple regrets*,
  AISTATS 2023.
- Wilson. *Stopping Bayesian Optimization with Probabilistic Regret Bounds*, NeurIPS 2024
  (arXiv:2402.16811).
- Weitzman. *Optimal Search for the Best Alternative* (Pandora's Box), Econometrica 1979.
- Pfisterer et al. *YAHPO Gym — Surrogate HPO Benchmarks*; Zimmer et al. *Auto-PyTorch / LCBench*.

---

*Status: proof-of-concept complete; MS2 (statistical signal on real benchmarks) passed on
both random-search and BO substrates; M1 done; literature positioning mapped. Next: M3
(meta-learned forward-looking stopping) and head-to-head vs. Xie 2025.*
