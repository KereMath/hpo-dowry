"""
Rigor: does the META policy generalize to UNSEEN cost levels (lambda)?

A single learned policy is supposed to adapt across cost regimes via its cost
features. We test this directly: train on a subset of cost multipliers and deploy
on held-out ones (and leave-one-task-out simultaneously, so both the task AND the
cost level are unseen at test time). Cross-family rbv2 landscapes.
"""

import numpy as np
from scipy import stats as sstats
from sklearn.ensemble import HistGradientBoostingRegressor

from core import MarginalStopping
from cross_family import get_crossfam
from experiment_suite import baseline_points, eval_rule, time_at_regret
from meta_stopping import MetaStopping, _trace_rows

ALL_LAM = [1.0, 2.0, 4.0, 8.0]
N_RS = 40


def build_tagged(suite, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(suite.keys())
    X, y, gkey, glam = [], [], [], []
    for ki, k in enumerate(keys):
        s, c, _ = suite[k]
        base_lam = 0.03 * max(s.max() - np.median(s), 1e-6) / c.mean()
        for _ in range(N_RS):
            perm = rng.permutation(len(s))
            for m in ALL_LAM:
                rx, ry = _trace_rows(s[perm], (base_lam * m) * c[perm])
                X += rx; y += ry; gkey += [ki] * len(ry); glam += [m] * len(ry)
    return np.array(X), np.array(y), np.array(gkey), np.array(glam)


def _saving(s, c, make_rule, lam):
    rt, rr, _ = eval_rule(s, c, make_rule, lam)
    bt = time_at_regret(baseline_points(s, c, lam), rr)
    return 100 * (1 - rt / bt) if np.isfinite(bt) and bt > 0 else np.nan


def run(suite, X, y, gkey, glam, train_lams, test_lams, tag):
    keys = list(suite.keys())
    meta, myop = {m: [] for m in test_lams}, {m: [] for m in test_lams}
    for ki in range(len(keys)):
        # double held-out: exclude the test task AND keep only training cost levels
        mask = (gkey != ki) & np.isin(glam, train_lams)
        clf = HistGradientBoostingRegressor(max_depth=4, max_iter=200,
                                            learning_rate=0.08, l2_regularization=1.0)
        clf.fit(X[mask], y[mask])
        s, c, _ = suite[keys[ki]]
        base_lam = 0.03 * max(s.max() - np.median(s), 1e-6) / c.mean()
        for m in test_lams:
            lam = base_lam * m
            meta[m].append(_saving(s, c, lambda: MetaStopping(clf), lam))
            myop[m].append(_saving(s, c, lambda: MarginalStopping("evt", k_min=4), lam))
    print(f"\n=== {tag}: train lam={train_lams} -> test UNSEEN lam={test_lams} (+ LOTO) ===")
    print(f"{'test lam':>9} | {'META':>16} | {'MYOPIC':>16} | {'META>MYOPIC p':>13}")
    for m in test_lams:
        a = np.array(meta[m]); b = np.array(myop[m])
        ok = ~(np.isnan(a) | np.isnan(b))
        try:
            p = sstats.wilcoxon(a[ok] - b[ok], alternative="greater").pvalue
        except Exception:
            p = float("nan")
        print(f"{m:9} | {np.nanmedian(a):7.1f}(w{np.nanmean(a>0):3.0%}) | "
              f"{np.nanmedian(b):7.1f}(w{np.nanmean(b>0):3.0%}) | {p:13.2e}")


def main():
    suite = get_crossfam()
    X, y, gkey, glam = build_tagged(suite)
    run(suite, X, y, gkey, glam, [1.0, 2.0], [4.0, 8.0], "extrapolate to HIGHER cost")
    run(suite, X, y, gkey, glam, [4.0, 8.0], [1.0, 2.0], "extrapolate to LOWER cost")


if __name__ == "__main__":
    main()
