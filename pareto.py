"""
Decisive test: does the cost-aware optimal-stopping rule's quality-vs-cost
frontier DOMINATE the best-tuned baseline frontier?

Baselines (fixed-N, patience-p) need a hyperparameter; sweeping it traces a
quality-vs-time frontier -- but that frontier is the BEST-CASE, hindsight-tuned
envelope. Our rule has one knob too (the cost lambda). We sweep lambda to trace
ITS frontier. If the optimal-stopping frontier sits on or below the baseline
envelope, the rule matches/beats hindsight tuning WITHOUT per-dataset tuning --
that is the real contribution.

For each operating point we report (mean training time used, mean regret).
Lower-left is better. We then compute, at matched regret, the time saving of the
optimal-stopping frontier over the baseline envelope.
"""

import numpy as np

from core import FixedBudget, MarginalStopping, Patience, run_trace

CACHE = "landscapes.npz"


def load():
    d = np.load(CACHE)
    return {k.replace("_s", ""): (d[k], d[k.replace("_s", "_c")])
            for k in d.files if k.endswith("_s")}


def eval_rule(scores, costs, make_rule, lam, n_orderings=3000, seed=7):
    r = np.random.default_rng(seed)
    M = len(scores)
    best = scores.max()
    times, regrets = [], []
    for _ in range(n_orderings):
        perm = r.permutation(M)
        out = run_trace(scores[perm], lam * costs[perm], make_rule())
        times.append(costs[perm][:out["n_evals"]].sum())
        regrets.append(best - out["incumbent"])
    return float(np.mean(times)), float(np.mean(regrets))


def frontier_envelope(points):
    """Lower-left Pareto envelope: for each, min regret achievable at <= its time."""
    pts = sorted(points)  # by time
    env = []
    best_r = np.inf
    for t, r in pts:
        best_r = min(best_r, r)
        env.append((t, best_r))
    return env


def time_at_regret(points, target_regret):
    """Min time among points achieving regret <= target."""
    cand = [t for t, r in points if r <= target_regret + 1e-9]
    return min(cand) if cand else np.inf


def main():
    lands = load()
    # sweep the cost knob for the optimal-stopping rule
    lam_mults = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    N_grid = list(range(3, 40, 2)) + list(range(40, 201, 10))
    p_grid = [3, 5, 8, 12, 18, 25, 35, 50, 70, 100]

    for name, (s, c) in lands.items():
        spread = s.max() - np.median(s)
        base_lam = 0.03 * spread / c.mean()

        opt_pts = []
        for m in lam_mults:
            lam = base_lam * m
            t, r = eval_rule(s, c, lambda: MarginalStopping("evt", k_obs=10, cost=None), lam)
            opt_pts.append((t, r))

        # baselines: lambda is irrelevant to their stopping (they ignore cost),
        # so use any lam (regret/time invariant). Use base_lam.
        fixed_pts = [eval_rule(s, c, lambda N=N: FixedBudget(N), base_lam) for N in N_grid]
        pat_pts = [eval_rule(s, c, lambda p=p: Patience(p), base_lam) for p in p_grid]
        baseline_pts = fixed_pts + pat_pts
        baseline_env = frontier_envelope(baseline_pts)

        print(f"\n=== {name} (full grid={c.sum():.1f}s, best acc={s.max():.4f}) ===")
        print("optimal-stopping (EVT) frontier  [time(s), regret, time%]:")
        for (t, r), m in zip(opt_pts, lam_mults):
            print(f"   lam x{m:<4}  t={t:7.2f}  ({100*t/c.sum():4.1f}%)  regret={r:.5f}")

        # Decisive comparison: at each opt point's regret, baseline's best time.
        print("verdict (at matched regret, optimal-stopping time vs best baseline time):")
        wins = 0
        for t, r in opt_pts:
            bt = time_at_regret(baseline_pts, r)
            if np.isinf(bt):
                print(f"   regret={r:.5f}: no baseline reaches this quality "
                      f"(opt uses {t:.2f}s)  -> opt-only region")
                wins += 1
                continue
            save = 100 * (1 - t / bt)
            flag = "WIN " if save > 1 else ("tie " if save > -1 else "lose")
            print(f"   regret={r:.5f}: opt={t:7.2f}s  best-baseline={bt:7.2f}s "
                  f"-> {save:+5.1f}%  [{flag}]")
            if save > 1:
                wins += 1
        print(f"   => optimal-stopping wins/ties on {wins}/{len(opt_pts)} operating points")


if __name__ == "__main__":
    main()
