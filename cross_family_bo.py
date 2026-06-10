"""
Cross-family generalization ON A BAYESIAN-OPTIMIZATION SUBSTRATE.

The cross-family rescue (cross_family_m3.py) was shown on random search. Here we
repeat it on adaptive Optuna-TPE traces over the rbv2 classic-ML families, to
confirm the meta-learned policy also generalizes across model families when the
underlying optimizer is BO (i.i.d. assumption broken).
"""

import os

import numpy as np
from scipy import stats as sstats
from sklearn.ensemble import HistGradientBoostingRegressor

from benchmark import _bench
from core import MarginalStopping, run_trace
from meta_stopping import MetaStopping, _trace_rows
from xie_baseline import gp_pbgi_trace, sample_pool


def time_at_regret(pts, target):
    cand = [t for t, r in pts if r <= target + 1e-9]
    return min(cand) if cand else np.inf

FAMILIES = {"rbv2_xgboost": "XGBoost", "rbv2_svm": "SVM", "rbv2_ranger": "RandomForest"}
N_PER, N_SEEDS, N_STEPS = 8, 8, 50
LAM_MULTS = [1.0, 2.0, 4.0, 8.0]
CACHE = os.path.join("cache", "cfbo_traces.npz")


def bo_traces(scenario, instance):
    """Adaptive GP-BO (PBGI) traces over a valid config pool. Sampling via
    cs.sample_configuration() respects conditionals, sidestepping rbv2's
    inactive-hyperparameter validation issues."""
    F, s, c = sample_pool(scenario, instance, M=300, seed=0,
                          score_key="acc", cost_key="timetrain", score_div=1.0)
    acq_lam = 0.03 * max(s.max() - np.median(s), 1e-6) / c.mean() * 2.0  # fixed mid-cost ordering
    S = np.zeros((N_SEEDS, N_STEPS)); C = np.zeros((N_SEEDS, N_STEPS))
    for si in range(N_SEEDS):
        so, co, _, _ = gp_pbgi_trace(F, s, c, acq_lam, N_STEPS, seed=si)
        n = len(so)
        S[si, :n] = so; C[si, :n] = co
        if n < N_STEPS:                      # pad short traces with the last value
            S[si, n:] = so[-1]; C[si, n:] = co[-1]
    return S, C


def get_cfbo():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return {k: (d[f"{k}__S"], d[f"{k}__C"], str(d[f"{k}__f"])) for k in d["__k"]}
    suite, save, ks = {}, {}, []
    for sc, fam in FAMILIES.items():
        for t in list(_bench(sc).instances)[:N_PER]:
            key = f"{sc}:{t}"
            S, C = bo_traces(sc, t)
            suite[key] = (S, C, fam)
            save[f"{key}__S"] = S; save[f"{key}__C"] = C; save[f"{key}__f"] = fam; ks.append(key)
            print(f"  BO {key}: best={S.max():.4f}", flush=True)
    save["__k"] = np.array(ks); np.savez(CACHE, **save)
    return suite


def build_meta(suite):
    keys = list(suite.keys())
    X, y, gk, gf = [], [], [], []
    for ki, k in enumerate(keys):
        S, C, fam = suite[k]
        blam = 0.03 * max(S.max() - np.median(S[:, 0]), 1e-6) / C.mean()
        for si in range(S.shape[0]):
            for m in LAM_MULTS:
                rx, ry = _trace_rows(S[si], (blam * m) * C[si])
                X += rx; y += ry; gk += [ki] * len(ry); gf += [fam] * len(ry)
    return np.array(X), np.array(y), np.array(gk), np.array(gf)


def base_env(S, C, lam, ref):
    pts = []
    for N in list(range(3, 30, 2)) + list(range(30, S.shape[1] + 1, 6)):
        t = [C[si, :min(N, S.shape[1])].sum() for si in range(S.shape[0])]
        r = [ref - S[si, :min(N, S.shape[1])].max() for si in range(S.shape[0])]
        pts.append((np.mean(t), np.mean(r)))
    return pts


def saving(S, C, mk, lam, ref):
    ts, rs = [], []
    for si in range(S.shape[0]):
        o = run_trace(S[si], lam * C[si], mk()); n = o["n_evals"]
        ts.append(C[si, :n].sum()); rs.append(ref - o["incumbent"])
    bt = time_at_regret(base_env(S, C, lam, ref), np.mean(rs))
    return 100 * (1 - np.mean(ts) / bt) if np.isfinite(bt) and bt > 0 else np.nan


def main():
    suite = get_cfbo()
    X, y, gk, gf = build_meta(suite)
    keys = list(suite.keys())
    print(f"\ncross-family BO meta-dataset: {len(y)} rows")
    for tag, fold_fn in [("LOTO", lambda ki, fam: gk != ki),
                         ("LOFO", lambda ki, fam: gf != fam)]:
        meta, myop = {m: [] for m in LAM_MULTS}, {m: [] for m in LAM_MULTS}
        for ki in range(len(keys)):
            S, C, fam = suite[keys[ki]]
            clf = HistGradientBoostingRegressor(max_depth=4, max_iter=200,
                                                learning_rate=0.08, l2_regularization=1.0)
            clf.fit(X[fold_fn(ki, fam)], y[fold_fn(ki, fam)])
            ref = S.max(); blam = 0.03 * max(S.max() - np.median(S[:, 0]), 1e-6) / C.mean()
            for m in LAM_MULTS:
                lam = blam * m
                meta[m].append(saving(S, C, lambda: MetaStopping(clf), lam, ref))
                myop[m].append(saving(S, C, lambda: MarginalStopping("evt", k_min=4), lam, ref))
        print(f"\n=== {tag} on BO substrate (cross-family): META vs MYOPIC saving % ===")
        print(f"{'lam x':>6} | {'META':>16} | {'MYOPIC':>16} | {'p':>10}")
        for m in LAM_MULTS:
            a = np.array(meta[m]); b = np.array(myop[m]); ok = ~(np.isnan(a) | np.isnan(b))
            try:
                p = sstats.wilcoxon(a[ok] - b[ok], alternative="greater").pvalue
            except Exception:
                p = float("nan")
            print(f"{m:6} | {np.nanmedian(a):7.1f}(w{np.nanmean(a>0):3.0%}) | "
                  f"{np.nanmedian(b):7.1f}(w{np.nanmean(b>0):3.0%}) | {p:10.2e}")


if __name__ == "__main__":
    main()
