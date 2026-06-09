"""
M1 evaluation: cost-aware observation phase + guarded tail.

Compare the original rule (opt-evt, fixed k_obs=10) against the M1 rule
(opt-guarded, small seed k_min=4 + shape-shrunk EVT tail) across ALL tasks, on
both axes:
  (A) matched-regret training-time saving vs hindsight baseline envelope  [want: keep]
  (B) cost-aware utility gap to the oracle stopper (normalized)           [want: shrink]
Goal: improve (B) -- especially in the high-cost regime where the fixed floor
hurt -- WITHOUT giving back the (A) win.
"""

import numpy as np
from scipy import stats as sstats

from benchmark import get_suite
from core import FixedBudget, MarginalStopping, Patience
from experiment_suite import OracleMarginal, baseline_points, eval_rule, time_at_regret

RULES = {
    "old(evt,k=10)": lambda: MarginalStopping(tail="evt", k_min=10, cost=None),
    "evt,k=4": lambda: MarginalStopping(tail="evt", k_min=4, cost=None),
    "guarded,k=4": lambda: MarginalStopping(tail="guarded", k_min=4, cost=None),
}


def main(scenario="lcbench", M=400):
    suite = get_suite(scenario, M=M, seed=0)
    tasks = list(suite.keys())
    lam_mults = [0.5, 1.0, 2.0, 4.0, 8.0]

    # per rule, per lam: lists across tasks
    save = {r: {m: [] for m in lam_mults} for r in RULES}
    nreg = {r: {m: [] for m in lam_mults} for r in RULES}

    for ti, t in enumerate(tasks):
        s, c = suite[t]
        spread = max(s.max() - np.median(s), 1e-6)
        base_lam = 0.03 * spread / c.mean()
        for m in lam_mults:
            lam = base_lam * m
            bpts = baseline_points(s, c, lam)
            _, _, oracle_u = eval_rule(s, c, lambda: OracleMarginal(s, lam * c.mean()), lam)
            for rname, mk in RULES.items():
                ot, orr, ou = eval_rule(s, c, mk, lam)
                bt = time_at_regret(bpts, orr)
                if np.isfinite(bt) and bt > 0:
                    save[rname][m].append(100 * (1 - ot / bt))
                nreg[rname][m].append((oracle_u - ou) / spread)
        print(f"  [{ti+1}/{len(tasks)}] {t}", flush=True)

    rnames = list(RULES)
    print(f"\n========== M1 COMPARISON: {scenario}, {len(tasks)} tasks ==========")
    print("\n(A) matched-regret time saving %  [median (win-rate)]   -- higher better")
    print(f"{'lam x':>7} | " + " | ".join(f"{r:>20}" for r in rnames))
    for m in lam_mults:
        row = f"{m:7} | "
        row += " | ".join(f"{np.median(np.array(save[r][m])):7.1f} (w{np.mean(np.array(save[r][m])>0):3.0%})"
                          for r in rnames)
        print(row)

    print("\n(B) utility gap to oracle (normalized)  [mean]   -- LOWER better")
    print(f"{'lam x':>7} | " + " | ".join(f"{r:>16}" for r in rnames))
    for m in lam_mults:
        row = f"{m:7} | " + " | ".join(f"{np.mean(nreg[r][m]):16.4f}" for r in rnames)
        print(row)

    print("\noverall mean utility-gap (all tasks x lambdas), lower=better:")
    for r in rnames:
        allr = np.concatenate([nreg[r][m] for m in lam_mults])
        print(f"   {r:>16}: {allr.mean():.4f}")


if __name__ == "__main__":
    main()
