"""
Kill-experiment on REAL hyperparameter-optimization landscapes.

We build genuine HPO landscapes: sample N random RandomForest configurations,
evaluate each by cross-validation on real sklearn datasets, and record both the
score (CV accuracy) AND the real cost (total fit time in seconds). Random search
= random orderings of these configs. We then ask the practical question:

    At equal final quality, how much TRAINING TIME does the cost-aware
    optimal-stopping layer save versus the standard defaults?

Cost-awareness knob: lambda converts seconds -> score units. The utility being
maximized is   U = incumbent_accuracy - lambda * total_fit_time.
"""

import os
import time

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from core import FixedBudget, MarginalStopping, Patience, StoppingRule, run_trace

CACHE = os.path.join(os.path.dirname(__file__), "landscapes.npz")


def sample_configs(n, seed):
    r = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        configs.append(dict(
            n_estimators=int(r.choice([20, 50, 100, 200, 400])),
            max_depth=int(r.choice([2, 3, 5, 8, 12, 20])),
            max_features=float(r.choice([0.2, 0.4, 0.6, 0.8, 1.0])),
            min_samples_leaf=int(r.choice([1, 2, 4, 8, 16])),
        ))
    return configs


def build_landscape(X, y, n_configs=200, seed=0):
    """Return (scores, costs) over n_configs real CV evaluations."""
    configs = sample_configs(n_configs, seed)
    scores, costs = [], []
    for cfg in configs:
        clf = RandomForestClassifier(random_state=0, n_jobs=1, **cfg)
        t0 = time.perf_counter()
        acc = cross_val_score(clf, X, y, cv=3, scoring="accuracy").mean()
        dt = time.perf_counter() - t0
        scores.append(acc)
        costs.append(dt)
    return np.array(scores), np.array(costs)


def get_landscapes():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return {k.replace("_s", ""): (d[k], d[k.replace("_s", "_c")])
                for k in d.files if k.endswith("_s")}
    datasets = {
        "digits": load_digits(return_X_y=True),
        "wine": load_wine(return_X_y=True),
        "breast_cancer": load_breast_cancer(return_X_y=True),
    }
    out, save = {}, {}
    for name, (X, y) in datasets.items():
        print(f"  building landscape: {name} ...", flush=True)
        s, c = build_landscape(X, y, n_configs=200, seed=42)
        out[name] = (s, c)
        save[f"{name}_s"] = s
        save[f"{name}_c"] = c
    np.savez(CACHE, **save)
    return out


class OracleMarginal(StoppingRule):
    """Gold standard for a finite landscape: true EI from the full score pool."""

    def __init__(self, pool, cost):
        self.pool = np.asarray(pool, dtype=float)
        self.cost = cost
        self.reset()

    def should_stop(self):
        ei = float(np.mean(np.maximum(self.pool - self.incumbent, 0.0)))
        return ei <= self.cost


def evaluate(scores, costs, make_rule, lam, n_orderings=3000, seed=7):
    """Average a rule over random search orderings of a finite landscape.

    costs are raw seconds; the rule sees cost-in-score-units = lam * seconds.
    """
    r = np.random.default_rng(seed)
    M = len(scores)
    best = scores.max()
    used_time, regret, n_evals, util = [], [], [], []
    for _ in range(n_orderings):
        perm = r.permutation(M)
        s_ord = scores[perm]
        c_ord_su = lam * costs[perm]  # score-unit cost stream
        rule = make_rule()
        out = run_trace(s_ord, c_ord_su, rule)
        t = costs[perm][:out["n_evals"]].sum()
        used_time.append(t)
        regret.append(best - out["incumbent"])
        n_evals.append(out["n_evals"])
        util.append(out["incumbent"] - lam * t)
    return dict(time=np.mean(used_time), regret=np.mean(regret),
               evals=np.mean(n_evals), util=np.mean(util))


def main():
    lands = get_landscapes()
    for name, (s, c) in lands.items():
        print(f"\nlandscape {name}: {len(s)} configs | best acc={s.max():.4f} "
              f"median acc={np.median(s):.4f} | fit time/config: "
              f"median={np.median(c):.3f}s total={c.sum():.1f}s")

    # Choose lambda so the per-eval cost is a small fraction of the score spread.
    for name, (s, c) in lands.items():
        spread = s.max() - np.median(s)
        mean_cost = c.mean()
        # price one mean-config of training at ~3% of the score spread
        lam = 0.03 * spread / mean_cost
        full_time = c.sum()

        rules = {
            "oracle": lambda: OracleMarginal(s, lam * c.mean()),
            "opt-evt": lambda: MarginalStopping("evt", k_obs=10, cost=None),
            "opt-gauss": lambda: MarginalStopping("gaussian", k_obs=10, cost=None),
            "patience15": lambda: Patience(p=15),
            "patience30": lambda: Patience(p=30),
            "fixed50": lambda: FixedBudget(50),
            "fixed100": lambda: FixedBudget(100),
        }
        res = {k: evaluate(s, c, mk, lam) for k, mk in rules.items()}

        print(f"\n=== {name}  (lambda={lam:.4g}, full-grid time={full_time:.1f}s) ===")
        print(f"{'rule':12} {'evals':>6} {'time(s)':>9} {'time%':>7} {'regret':>9} {'util':>9}")
        print("-" * 56)
        for k, rr in res.items():
            print(f"{k:12} {rr['evals']:6.1f} {rr['time']:9.2f} "
                  f"{100*rr['time']/full_time:6.1f}% {rr['regret']:9.5f} {rr['util']:9.5f}")

        # Budget-saved-at-equal-quality: match opt-gauss regret with a fixed-N.
        opt = res["opt-gauss"]
        target = opt["regret"]
        # find smallest fixed N whose mean regret <= target
        eqN, eqtime = None, None
        for N in range(2, len(s) + 1):
            rr = evaluate(s, c, lambda N=N: FixedBudget(N), lam, n_orderings=1500)
            if rr["regret"] <= target:
                eqN, eqtime = N, rr["time"]
                break
        if eqtime:
            saved = 100 * (1 - opt["time"] / eqtime)
            print(f"  -> to MATCH opt-gauss regret ({target:.5f}), fixed search needs "
                  f"N={eqN} ({eqtime:.2f}s); opt-gauss uses {opt['time']:.2f}s "
                  f"=> {saved:.1f}% less training time at equal quality")


if __name__ == "__main__":
    main()
