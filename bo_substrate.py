"""
Phase 2-ext: run the cost-aware stopper on top of a REAL Bayesian-optimization
substrate (Optuna TPE), not just random search.

This is the critical stress test: BO samples adaptively, so the i.i.d.
assumption behind the optimal-stopping theory is violated. If the stopper still
saves training time at matched quality on BO traces, the layer is robust to the
dominant real optimizer.

We run TPE over the YAHPO lcbench surrogate (real score + real runtime), cache
the proposal-order traces, then apply the stopper and baselines.
"""

import os
import sys

import numpy as np

os.environ["PYTHONWARNINGS"] = "ignore"
import optuna

from benchmark import _bench
from core import FixedBudget, MarginalStopping, Patience, run_trace

optuna.logging.set_verbosity(optuna.logging.WARNING)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SCORE_KEY, COST_KEY = "val_accuracy", "time"


def _suggest_space(trial, cs, fixed):
    cfg = dict(fixed)
    for hp in cs.get_hyperparameters():
        n = hp.name
        if n in fixed:
            continue
        if hasattr(hp, "choices"):
            cfg[n] = trial.suggest_categorical(n, list(hp.choices))
        elif hasattr(hp, "lower"):
            log = bool(getattr(hp, "log", False))
            if "Integer" in type(hp).__name__:
                cfg[n] = trial.suggest_int(n, int(hp.lower), int(hp.upper), log=log)
            else:
                cfg[n] = trial.suggest_float(n, float(hp.lower), float(hp.upper), log=log)
    return cfg


def bo_traces(scenario, instance, n_seeds=15, n_trials=80, seed0=0):
    b = _bench(scenario)
    b.set_instance(instance)
    cs = b.get_opt_space()
    # fixed dims: fidelity at max + the instance id
    fixed = {}
    for fp in b.config.fidelity_params:
        hp = cs.get_hyperparameter(fp)
        fixed[fp] = hp.upper if hasattr(hp, "upper") else hp.sequence[-1]
    fixed[b.config.instance_names] = instance

    S = np.zeros((n_seeds, n_trials))
    C = np.zeros((n_seeds, n_trials))
    for si in range(n_seeds):
        sampler = optuna.samplers.TPESampler(seed=seed0 + si, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        rec = []

        def objective(trial):
            cfg = _suggest_space(trial, cs, fixed)
            out = b.objective_function([cfg])[0]
            sc = out[SCORE_KEY] / 100.0
            ct = max(out[COST_KEY], 1e-6)
            trial.set_user_attr("cost", ct)
            rec.append((sc, ct))
            return sc

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        for j, (sc, ct) in enumerate(rec):
            S[si, j] = sc
            C[si, j] = ct
    return S, C


def get_bo_suite(scenario="lcbench", n_seeds=15, n_trials=80, max_tasks=None):
    cache = os.path.join(CACHE_DIR, f"BO_{scenario}_seeds{n_seeds}_t{n_trials}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return {t: (d[f"{t}__S"], d[f"{t}__C"]) for t in d["__tasks__"]}
    b = _bench(scenario)
    tasks = list(b.instances)
    if max_tasks:
        tasks = tasks[:max_tasks]
    suite, save = {}, {}
    for i, t in enumerate(tasks):
        S, C = bo_traces(scenario, t, n_seeds, n_trials)
        suite[t] = (S, C)
        save[f"{t}__S"] = S
        save[f"{t}__C"] = C
        print(f"  [{i+1}/{len(tasks)}] BO {t}: best={S.max():.4f} "
              f"med-final-incumbent={np.median(S.max(axis=1)):.4f}", flush=True)
    save["__tasks__"] = np.array(list(suite.keys()))
    np.savez(cache, **save)
    return suite


# ----------------------------- analysis ---------------------------------- #
def eval_on_traces(S, C, make_rule, lam, ref_best):
    times, regrets = [], []
    for si in range(S.shape[0]):
        out = run_trace(S[si], lam * C[si], make_rule())
        n = out["n_evals"]
        times.append(C[si, :n].sum())
        regrets.append(ref_best - out["incumbent"])
    return np.mean(times), np.mean(regrets)


def baseline_env_bo(S, C, lam, ref_best):
    pts = []
    for N in list(range(3, 40, 3)) + list(range(40, S.shape[1] + 1, 10)):
        pts.append(eval_on_traces(S, C, lambda N=N: FixedBudget(N), lam, ref_best))
    for p in [3, 5, 8, 12, 18, 25, 35]:
        pts.append(eval_on_traces(S, C, lambda p=p: Patience(p), lam, ref_best))
    return pts


def time_at_regret(pts, target):
    cand = [t for t, r in pts if r <= target + 1e-9]
    return min(cand) if cand else np.inf


def analyze(scenario="lcbench"):
    from scipy import stats as sstats
    suite = get_bo_suite(scenario)
    tasks = list(suite.keys())
    lam_mults = [1.0, 2.0, 4.0, 8.0]
    save = {m: [] for m in lam_mults}
    for ti, t in enumerate(tasks):
        S, C = suite[t]
        ref_best = S.max()
        spread = max(ref_best - np.median(S[:, 0]), 1e-6)
        base_lam = 0.03 * spread / C.mean()
        for m in lam_mults:
            lam = base_lam * m
            ot, orr = eval_on_traces(S, C, lambda: MarginalStopping("evt", k_min=4), lam, ref_best)
            bpts = baseline_env_bo(S, C, lam, ref_best)
            bt = time_at_regret(bpts, orr)
            if np.isfinite(bt) and bt > 0:
                save[m].append(100 * (1 - ot / bt))
    print(f"\n===== BO SUBSTRATE verdict: {scenario}, {len(tasks)} tasks =====")
    print("matched-regret training-time saving of stopper(evt,k=4) vs hindsight baseline on BO traces")
    print(f"{'lam x':>7} {'median%':>9} {'mean%':>8} {'win-rate':>9} {'Wilcoxon p':>12} {'n':>4}")
    for m in lam_mults:
        sv = np.array(save[m])
        if len(sv) < 3:
            continue
        try:
            p = sstats.wilcoxon(sv).pvalue
        except Exception:
            p = float("nan")
        print(f"{m:7} {np.median(sv):9.1f} {np.mean(sv):8.1f} {np.mean(sv>0):9.0%} {p:12.2e} {len(sv):4d}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if mode == "gen":
        get_bo_suite("lcbench")
    else:
        analyze("lcbench")
