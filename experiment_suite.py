"""
Suite-level decisive experiment (MS2 go/no-go).

Across ALL tasks of a real benchmark (LCBench, 34 OpenML datasets), test whether
the parameter-free cost-aware stopping rule (opt-evt) Pareto-dominates the
HINDSIGHT-TUNED baseline envelope (best fixed-N or patience-p per task) on the
quality-vs-training-cost frontier -- and whether the win is statistically
significant across tasks.

Two views:
  (A) Matched-regret time saving: for each cost level (lambda), at the regret the
      stopper achieves, how much LESS training time than the best baseline that
      reaches the same regret. Aggregated over tasks; Wilcoxon signed-rank.
  (B) Cost-aware utility vs oracle: U = best_acc - lambda*time. Normalized regret
      to the oracle stopper; opt-evt (one setting) vs best-tuned baseline (per
      lambda). Aggregated; Wilcoxon.
"""

import numpy as np
from scipy import stats as sstats

from benchmark import get_suite
from core import FixedBudget, MarginalStopping, Patience, StoppingRule, run_trace


class OracleMarginal(StoppingRule):
    def __init__(self, pool, cost):
        self.pool = np.asarray(pool, float)
        self.cost = cost
        self.reset()

    def should_stop(self):
        ei = float(np.mean(np.maximum(self.pool - self.incumbent, 0.0)))
        return ei <= self.cost


def eval_rule(scores, costs, make_rule, lam, n_orderings=300, seed=7):
    r = np.random.default_rng(seed)
    M = len(scores)
    best = scores.max()
    times, regrets, utils = [], [], []
    for _ in range(n_orderings):
        perm = r.permutation(M)
        out = run_trace(scores[perm], lam * costs[perm], make_rule())
        t = costs[perm][:out["n_evals"]].sum()
        times.append(t)
        regrets.append(best - out["incumbent"])
        utils.append(out["incumbent"] - lam * t)
    return np.mean(times), np.mean(regrets), np.mean(utils)


def baseline_points(scores, costs, lam, n_orderings=300):
    N_grid = list(range(3, 40, 3)) + list(range(40, min(len(scores), 200) + 1, 15))
    p_grid = [3, 5, 8, 12, 18, 25, 35, 50, 70]
    pts = []  # (time, regret, util, label)
    for N in N_grid:
        t, r, u = eval_rule(scores, costs, lambda N=N: FixedBudget(N), lam, n_orderings)
        pts.append((t, r, u))
    for p in p_grid:
        t, r, u = eval_rule(scores, costs, lambda p=p: Patience(p), lam, n_orderings)
        pts.append((t, r, u))
    return pts


def time_at_regret(pts, target):
    cand = [t for t, r, _ in pts if r <= target + 1e-9]
    return min(cand) if cand else np.inf


def main(scenario="lcbench", M=400):
    suite = get_suite(scenario, M=M, seed=0)
    tasks = list(suite.keys())
    lam_mults = [0.5, 1.0, 2.0, 4.0, 8.0]

    # storage: per (lam) lists across tasks
    save_by_lam = {m: [] for m in lam_mults}   # matched-regret savings (%)
    util_opt_by_lam = {m: [] for m in lam_mults}
    util_base_by_lam = {m: [] for m in lam_mults}
    util_norm_opt = {m: [] for m in lam_mults}
    util_norm_base = {m: [] for m in lam_mults}

    for ti, t in enumerate(tasks):
        s, c = suite[t]
        spread = max(s.max() - np.median(s), 1e-6)
        base_lam = 0.03 * spread / c.mean()
        bpts_cache = {}
        for m in lam_mults:
            lam = base_lam * m
            # opt-evt: parameter-free (cost=None uses running mean cost)
            ot, orr, ou = eval_rule(s, c, lambda: MarginalStopping("evt", 10, None), lam)
            # oracle utility (gold standard for this lambda)
            _, _, oracle_u = eval_rule(s, c, lambda: OracleMarginal(s, lam * c.mean()), lam)
            # baselines (regret/time independent of lambda; util depends on lambda)
            if m not in bpts_cache:
                bpts_cache[m] = baseline_points(s, c, lam)
            bpts = bpts_cache[m]

            # (A) matched-regret time saving vs best baseline reaching opt's regret
            bt = time_at_regret(bpts, orr)
            if np.isfinite(bt) and bt > 0:
                save_by_lam[m].append(100 * (1 - ot / bt))
            # (B) utility: opt vs best-tuned baseline utility
            best_base_u = max(u for _, _, u in bpts)
            util_opt_by_lam[m].append(ou)
            util_base_by_lam[m].append(best_base_u)
            # normalized utility-regret to oracle (lower=better)
            util_norm_opt[m].append((oracle_u - ou) / spread)
            util_norm_base[m].append((oracle_u - best_base_u) / spread)
        print(f"  [{ti+1}/{len(tasks)}] task {t} done", flush=True)

    print(f"\n================ SUITE VERDICT: {scenario}, {len(tasks)} tasks ================")
    print("\n(A) Matched-regret TRAINING-TIME SAVING of opt-evt vs best baseline at same regret")
    print(f"{'lam x':>7} {'median%':>9} {'mean%':>8} {'win-rate':>9} {'Wilcoxon p':>12} {'n':>4}")
    for m in lam_mults:
        sv = np.array(save_by_lam[m])
        if len(sv) < 3:
            continue
        win = np.mean(sv > 0)
        try:
            p = sstats.wilcoxon(sv).pvalue
        except Exception:
            p = float("nan")
        print(f"{m:7} {np.median(sv):9.1f} {np.mean(sv):8.1f} {win:9.0%} {p:12.2e} {len(sv):4d}")

    print("\n(B) Cost-aware UTILITY: opt-evt (parameter-free) vs best hindsight-tuned baseline")
    print(f"{'lam x':>7} {'opt>base win':>12} {'Wilcoxon p':>12} {'norm-reg opt':>13} {'norm-reg base':>14}")
    for m in lam_mults:
        uo = np.array(util_opt_by_lam[m]); ub = np.array(util_base_by_lam[m])
        win = np.mean(uo > ub)
        try:
            p = sstats.wilcoxon(uo - ub).pvalue
        except Exception:
            p = float("nan")
        print(f"{m:7} {win:12.0%} {p:12.2e} {np.mean(util_norm_opt[m]):13.4f} "
              f"{np.mean(util_norm_base[m]):14.4f}")
    print("\n(norm-reg = utility gap to oracle / score-spread; lower is better)")


if __name__ == "__main__":
    main()
