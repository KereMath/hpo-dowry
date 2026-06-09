"""
Synthetic validation of the cost-aware optimal-stopping layer.

Streams are i.i.d. draws from a known score distribution (random search over an
effectively infinite config space). Because the distribution is known, we can
compute the GOLD-STANDARD oracle stopper (true expected improvement) and measure
how close each *online* rule gets.

Two claims under test:
  (1) The optimal-stopping rule (with a sound tail model) nearly matches the
      oracle utility, and the naive empirical tail fails.
  (2) Adaptivity: with a SINGLE setting (just the cost c), it tracks the
      per-distribution best, while any fixed patience value wins on one
      distribution and loses on another.

Utility:  U = incumbent_score - cost * n_evals   (the quantity the rule maximizes)
"""

import numpy as np

from core import (
    FixedBudget,
    MarginalStopping,
    Patience,
    StoppingRule,
    run_trace,
)

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Score distributions (higher = better), each with a big reference pool for    #
# computing the true expected improvement used by the oracle.                  #
# --------------------------------------------------------------------------- #
def make_dists():
    g = np.random.default_rng(12345)
    dists = {}

    # 1) Gaussian-ish landscape
    dists["gauss"] = lambda n, r: np.clip(r.normal(0.70, 0.10, n), 0, 1)

    # 2) HPO-like: a broad mediocre plateau with a thin slice of good configs
    def hpo_like(n, r):
        good = r.random(n) < 0.10
        s = np.where(good, r.normal(0.88, 0.04, n), r.normal(0.60, 0.06, n))
        return np.clip(s, 0, 1)

    dists["hpo_like"] = hpo_like

    # 3) Heavy upper tail: rare configs are MUCH better (EVT should matter)
    def heavy(n, r):
        base = r.normal(0.55, 0.05, n)
        bump = r.pareto(2.5, n) * 0.05  # heavy positive tail
        return np.clip(base + bump, 0, 1)

    dists["heavy_tail"] = heavy

    return dists


class OracleMarginal(StoppingRule):
    """Gold standard: knows the true distribution via a large reference pool."""

    def __init__(self, pool, cost):
        self.pool = np.asarray(pool, dtype=float)
        self.cost = cost
        self.reset()

    def should_stop(self):
        ei = float(np.mean(np.maximum(self.pool - self.incumbent, 0.0)))
        return ei <= self.cost


def eval_rule(sampler, ref_pool, make_rule, cost, n_streams=3000, t_max=400, seed=1):
    r = np.random.default_rng(seed)
    utils, evals, incs = [], [], []
    for _ in range(n_streams):
        scores = sampler(t_max, r)
        costs = np.full(t_max, cost)
        rule = make_rule()
        out = run_trace(scores, costs, rule)
        u = out["incumbent"] - cost * out["n_evals"]
        utils.append(u)
        evals.append(out["n_evals"])
        incs.append(out["incumbent"])
    return {
        "utility": float(np.mean(utils)),
        "evals": float(np.mean(evals)),
        "incumbent": float(np.mean(incs)),
    }


def main():
    dists = make_dists()
    cost = 0.004  # per-eval price in score units
    big = np.random.default_rng(999)
    pools = {name: s(400_000, big) for name, s in dists.items()}

    print(f"\n=== Claim 1: online optimal-stopping vs oracle (cost={cost}) ===")
    header = f"{'dist':12} {'oracle':>8} {'evt':>8} {'gauss':>8} {'empirical':>10} {'patience20':>11} {'fixed50':>8}"
    print(header)
    print("-" * len(header))

    for name, sampler in dists.items():
        pool = pools[name]
        rules = {
            "oracle": lambda p=pool: OracleMarginal(p, cost),
            "evt": lambda: MarginalStopping(tail="evt", k_obs=10, cost=cost),
            "gauss": lambda: MarginalStopping(tail="gaussian", k_obs=10, cost=cost),
            "empirical": lambda: MarginalStopping(tail="empirical", k_obs=10, cost=cost),
            "patience20": lambda: Patience(p=20),
            "fixed50": lambda: FixedBudget(50),
        }
        res = {k: eval_rule(sampler, pool, mk, cost) for k, mk in rules.items()}
        row = (
            f"{name:12} "
            f"{res['oracle']['utility']:8.4f} "
            f"{res['evt']['utility']:8.4f} "
            f"{res['gauss']['utility']:8.4f} "
            f"{res['empirical']['utility']:10.4f} "
            f"{res['patience20']['utility']:11.4f} "
            f"{res['fixed50']['utility']:8.4f}"
        )
        print(row)
        # also show evals for the EVT rule vs oracle
        print(f"{'  (evals)':12} {res['oracle']['evals']:8.1f} {res['evt']['evals']:8.1f}"
              f" {res['gauss']['evals']:8.1f} {res['empirical']['evals']:10.1f}"
              f" {res['patience20']['evals']:11.1f} {res['fixed50']['evals']:8.1f}")

    # ----------------------------------------------------------------------- #
    # Claim 2: adaptivity. Best-tuned-by-hindsight patience PER distribution   #
    # vs a single optimal-stopping setting used everywhere.                    #
    # ----------------------------------------------------------------------- #
    print(f"\n=== Claim 2: adaptivity (single setting across distributions, cost={cost}) ===")
    patience_grid = [5, 10, 20, 40, 80]
    fixed_grid = [10, 25, 50, 100, 200]

    print(f"{'dist':12} {'opt(evt)':>9} | best-tuned patience       | best-tuned fixed")
    print("-" * 70)
    per_dist_best_patience = {}
    for name, sampler in dists.items():
        pool = pools[name]
        opt = eval_rule(sampler, pool, lambda: MarginalStopping("evt", 10, cost), cost)["utility"]
        pat = {p: eval_rule(sampler, pool, lambda p=p: Patience(p), cost)["utility"] for p in patience_grid}
        fix = {n: eval_rule(sampler, pool, lambda n=n: FixedBudget(n), cost)["utility"] for n in fixed_grid}
        best_p = max(pat, key=pat.get)
        best_n = max(fix, key=fix.get)
        per_dist_best_patience[name] = best_p
        print(f"{name:12} {opt:9.4f} | p*={best_p:<3} U={pat[best_p]:7.4f}  (p=20:{pat.get(20,float('nan')):7.4f}) "
              f"| N*={best_n:<3} U={fix[best_n]:7.4f}")

    print("\nNote: a single patience value cannot be optimal on all distributions;")
    print(f"per-distribution best p* = {per_dist_best_patience}")
    print("The optimal-stopping rule uses the SAME cost setting everywhere.")


if __name__ == "__main__":
    main()
