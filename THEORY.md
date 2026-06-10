# Theory

This note states what can be proved honestly about the stopping rules studied here.
It does **not** claim a new optimality theorem for the myopic rule (that is Weitzman's,
restated below); the positive contribution is (i) a precise account of why a fixed
myopic threshold is sub-optimal in the *realistic* setting, and (ii) a clean regret
reduction for the meta-learned policy that connects test-time regret to the policy's
held-out prediction error — explaining the empirical LOTO / LOFO / unseen-λ results.

## Setup

A run produces a stream of configuration scores `X_1, X_2, …` (higher is better) at
per-evaluation costs `γ_1, γ_2, …` (already in *score units*: `γ_i = λ · time_i`). After
`t` evaluations the incumbent is `M(t) = max_{i≤t} X_i` and the cumulative cost is
`Γ(t) = Σ_{i≤t} γ_i`. A stopping rule chooses a stopping time `τ`; its **utility** is

    U(τ) = M(τ) − Γ(τ).

The oracle (hindsight) optimum on a realized trace is `t* = argmax_t U(t)`, with utility
`U(t*)`. **Regret** of a rule that stops at `τ` is `R(τ) = U(t*) − U(τ) ≥ 0`.

The *myopic reservation rule* stops at `τ_z = min{t : M(t) ≥ z}` for the reservation value
`z` solving `E[(X − z)^+] = c` (one-step indifference); equivalently it stops once the
one-step expected improvement `EI(M(t)) = E[(X − M(t))^+]` drops below the next cost `c`.

---

## 1. Myopic optimality under idealized assumptions (Weitzman; restated)

**Theorem 1 (Weitzman 1979; Pandora's box with identical boxes).**
*If the scores `X_i` are i.i.d. with known distribution `F`, recall is allowed, the
per-evaluation cost is a constant `c`, and the horizon is unbounded, then the myopic
reservation rule `τ_z` maximizes `E[U(τ)]` over all stopping times.*

*Proof sketch.* Define the continuation value with incumbent `b`,
`V(b) = sup_τ E[U(τ) − b | M(0)=b]`. By the optimality principle,
`V(b) = max{ 0, −c + E[(X−b)^+] + E[V(max(b,X))] }`. `V` is non-increasing and convex in
`b`; the stopping region is `{b : V(b)=0}`. For identical i.i.d. boxes Weitzman's exchange
argument shows the index of every unopened box equals the same reservation value `z` with
`E[(X−z)^+]=c`, and opening is worthwhile iff the incumbent is below `z`. Hence `τ_z` is
optimal. ∎

So in the idealized regime there is **nothing to beat**: the myopic rule is globally
optimal, not merely one-step optimal.

---

## 2. Why a fixed myopic threshold is sub-optimal in our setting

Our experiments violate every premise of Theorem 1, and each violation breaks the
optimality of a single constant threshold.

**Proposition 2.** *Each of the following makes the constant-`z` myopic rule sub-optimal:*

1. **Finite horizon / sampling without replacement.** The candidate pool is finite (e.g.
   400 configs) and the trace is capped. With a deadline `T`, the optimal reservation value
   is **time-varying and decreasing**: `z_t` solves a backward recursion
   `z_t = b : E[(X−b)^+] + E[V_{t+1}(max(b,X))] = c`, with `V_T ≡ 0`, so `z_t ↓` as `t→T`
   (less remaining opportunity). A single `z` cannot match a decreasing `z_t`; it stops too
   late early on and too early near the deadline. *Proof:* monotonicity of `V_t` in the
   horizon (`V_t ≥ V_{t+1}` pointwise by adding one admissible action) gives `z_t ≥ z_{t+1}`.
2. **Unknown distribution.** `F` is estimated online; the plug-in `\hat z` solving
   `\widehat{EI}(\hat z)=c` carries the estimation error of `\widehat{EI}` (Prop. 3).
3. **Non-i.i.d. evaluations.** Under Bayesian optimization the `X_i` are adaptively chosen,
   so the i.i.d. premise fails and the exchange argument does not apply.
4. **Heterogeneous cost.** With `γ_i` non-constant, the constant-`c` reservation value is
   only an average-case surrogate; the per-step indifference cost is `γ_{t+1}`, not `c`.

None of these is a defect of the idea — they define the regime where a *learned* policy
can help. ∎

---

## 3. EVT plug-in estimate of the improvement tail

`EI(b) = ∫_b^∞ S(x)\,dx`, `S = 1−F`. Above a high threshold `u`, by the
Pickands–Balkema–de Haan theorem the exceedance distribution converges to a Generalized
Pareto `GPD(ξ, β)`, giving the closed form (for `b ≥ u`, `ξ<1`)

    EI(b) = p_u · β/(1−ξ) · (1 + ξ(b−u)/β)^{1 − 1/ξ},   p_u = P(X>u).

**Remarks (honest).** The PWM estimator of `(ξ,β)` is consistent as the number of
exceedances `→∞` and is well-behaved for small samples, but for very few exceedances the
tail estimate has high variance; our `guarded` shrinkage variant traded this variance for
bias and *empirically hurt* (it over-stops). EVT optimum-estimation is also known to be
sampler-dependent (Bidlingmaier et al., Swarm Evol. Comput. 2022) — a caution that the
myopic plug-in inherits and that motivates the learned policy.

---

## 4. Regret reduction for the meta-learned policy

The meta-policy predicts `r̂(t) ≈ r(t) = t* − t` (steps to the hindsight optimum) and stops
at `τ = min{t ≥ k₀ : r̂(t) ≤ 0}`. The following bounds its regret by its prediction error
and the smoothness of the realized utility curve — quantities we *measure* on held-out
tasks/families/costs.

**Proposition 4 (policy-error ⇒ regret).**
*Suppose on a trace the utility `U(·)` is `L`-Lipschitz in the step index
(`|U(s)−U(t)| ≤ L|s−t|`), and the policy's prediction error is uniformly bounded,
`sup_t |r̂(t) − (t*−t)| ≤ δ` with `δ < 1` on the relevant range. Then the policy stops at
`τ = t*` and incurs zero regret. More generally, for any `δ ≥ 0`, `|τ − t*| ≤ ⌈δ⌉` and*

    R(τ) = U(t*) − U(τ) ≤ L · ⌈δ⌉.

*Proof.* `r(t)=t*−t` is strictly decreasing with a single sign change at `t=t*`
(`r(t)>0` for `t<t*`, `r(t)≤0` for `t≥t*`). If `|r̂(t)−r(t)|<1` then `sgn(r̂(t))=sgn(r(t))`
wherever `|r(t)|≥1`, so the first `t` with `r̂(t)≤0` lies in `{t*−1,…}`; combined with
`r̂(t)≤0 ⇒ r(t)≤δ ⇒ t ≥ t*−δ`, we get `t* − ⌈δ⌉ ≤ τ ≤ t*` (the policy never overshoots a
strictly-decreasing target by more than the error). Lipschitzness gives
`R(τ) ≤ L|τ−t*| ≤ L⌈δ⌉`. The `δ<1` case yields `τ=t*`. ∎

**Why this explains the experiments.** `δ` is exactly the policy's prediction error on the
held-out evaluation (leave-one-task-out, leave-one-family-out, unseen-λ). The empirical
finding — small regret / large savings on entirely unseen tasks, families, and cost levels —
is the statement that the *trained `r̂` transfers with small `δ`*. Proposition 4 turns that
empirical transfer into a regret guarantee: **whatever the benchmark, the meta-policy's
regret is controlled by its held-out prediction error times the utility-curve smoothness.**
By contrast the fixed myopic rule has no such adaptivity: its error is the fixed gap of
Prop. 2, which is benchmark-dependent and, as measured, large on XGBoost/SVM.

**Limitations of the theory.** Prop. 4 bounds *per-trace* regret given a uniform error bound
`δ`; turning the *average* held-out error into a high-probability uniform `δ` needs a
concentration / generalization argument (a learning-theoretic bound on `r̂`), which we do not
prove and leave as future work. `L` (utility smoothness) is bounded by the score range plus
the per-step cost and is finite but benchmark-dependent. We make **no** claim of beating the
idealized Weitzman optimum; the contribution is in the realistic estimated / finite /
non-i.i.d. / heterogeneous-cost regime, where Theorem 1 does not apply.

---

## References
Weitzman (1979) *Optimal Search for the Best Alternative*, Econometrica. Pickands (1975);
Balkema–de Haan (1974) — GPD tail limit. Hosking & Wallis (1987) — PWM. Xie et al. (2025) —
PBGI cost-aware stopping (myopic, GP-tied). Bidlingmaier et al. (2022) — EVT optimum-
estimation reliability.
