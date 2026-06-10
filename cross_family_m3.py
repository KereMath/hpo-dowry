"""
Does the meta-learned forward-looking policy (M3) GENERALIZE across model families,
where the fixed myopic EVT rule does not (see cross_family.py: negative on XGBoost/SVM)?

Builds a cross-family meta-dataset (random-search traces over rbv2 XGBoost / SVM /
RandomForest landscapes) and evaluates META vs MYOPIC under:
  - leave-one-task-out (LOTO, 36 folds), and
  - leave-one-FAMILY-out (LOFO, 3 folds) -- the strongest test: train on two model
    families, deploy on a held-out family never seen in training.
"""

import numpy as np
from scipy import stats as sstats
from sklearn.ensemble import HistGradientBoostingRegressor

from core import MarginalStopping
from cross_family import get_crossfam
from experiment_suite import baseline_points, eval_rule, time_at_regret
from meta_stopping import MetaStopping, _trace_rows

LAM_MULTS = [1.0, 2.0, 4.0, 8.0]
N_RS = 40


def build_cf_meta(suite, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(suite.keys())
    X, y, gkey, gfam = [], [], [], []
    for ki, k in enumerate(keys):
        s, c, fam = suite[k]
        spread = max(s.max() - np.median(s), 1e-6)
        base_lam = 0.03 * spread / c.mean()
        for _ in range(N_RS):
            perm = rng.permutation(len(s))
            for m in LAM_MULTS:
                su = (base_lam * m) * c[perm]
                rx, ry = _trace_rows(s[perm], su)
                X += rx; y += ry; gkey += [ki] * len(ry); gfam += [fam] * len(ry)
    return np.array(X), np.array(y), np.array(gkey), np.array(gfam)


def _saving(s, c, make_rule, lam):
    rt, rr, _ = eval_rule(s, c, make_rule, lam)
    bpts = baseline_points(s, c, lam)
    bt = time_at_regret(bpts, rr)
    return 100 * (1 - rt / bt) if np.isfinite(bt) and bt > 0 else np.nan


def evaluate(suite, X, y, gkey, gfam, holdout_mask_fn, fold_name):
    """holdout_mask_fn(ki, fam) -> True if row belongs to the held-out fold."""
    keys = list(suite.keys())
    meta, myop = {m: [] for m in LAM_MULTS}, {m: [] for m in LAM_MULTS}
    folds = sorted(set(gfam)) if fold_name == "LOFO" else range(len(keys))
    for fold in folds:
        if fold_name == "LOFO":
            test_keys = [ki for ki in range(len(keys)) if suite[keys[ki]][2] == fold]
            train = gfam != fold
        else:
            test_keys = [fold]
            train = gkey != fold
        clf = HistGradientBoostingRegressor(max_depth=4, max_iter=200,
                                            learning_rate=0.08, l2_regularization=1.0)
        clf.fit(X[train], y[train])
        for ki in test_keys:
            s, c, _ = suite[keys[ki]]
            spread = max(s.max() - np.median(s), 1e-6)
            base_lam = 0.03 * spread / c.mean()
            for m in LAM_MULTS:
                lam = base_lam * m
                meta[m].append(_saving(s, c, lambda: MetaStopping(clf), lam))
                myop[m].append(_saving(s, c, lambda: MarginalStopping("evt", k_min=4), lam))
    print(f"\n=== {fold_name}: META vs MYOPIC across families, matched-regret saving % ===")
    print(f"{'lam x':>6} | {'META':>16} | {'MYOPIC':>16} | {'META>MYOPIC p':>13}")
    for m in LAM_MULTS:
        a = np.array(meta[m]); b = np.array(myop[m])
        ok = ~(np.isnan(a) | np.isnan(b))
        try:
            p = sstats.wilcoxon(a[ok] - b[ok], alternative="greater").pvalue
        except Exception:
            p = float("nan")
        print(f"{m:6} | {np.nanmedian(a):7.1f}(w{np.nanmean(a>0):3.0%}) | "
              f"{np.nanmedian(b):7.1f}(w{np.nanmean(b>0):3.0%}) | {p:13.2e}")
    return meta, myop


def main():
    suite = get_crossfam()
    X, y, gkey, gfam = build_cf_meta(suite)
    print(f"cross-family meta-dataset: {len(y)} rows, frac(t*-t<=0)={np.mean(y<=0):.3f}")
    evaluate(suite, X, y, gkey, gfam, None, "LOTO")
    evaluate(suite, X, y, gkey, gfam, None, "LOFO")


if __name__ == "__main__":
    main()
