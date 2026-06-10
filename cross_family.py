"""
Cross-family generalization: does the optimizer-agnostic myopic EVT stopping rule
keep saving training time on classic-ML model families (not just Auto-PyTorch MLPs)?

Builds landscapes from three rbv2_* YAHPO scenarios -- XGBoost, SVM, Random Forest
(ranger) -- each a different model family over many OpenML datasets, with real
`timetrain` cost. Runs the same matched-regret training-time-saving analysis used
for LCBench, aggregated overall and broken down per family.
"""

import os

import numpy as np
from scipy import stats as sstats

from benchmark import build_landscape
from core import MarginalStopping
from experiment_suite import baseline_points, eval_rule, time_at_regret

FAMILIES = {"rbv2_xgboost": "XGBoost", "rbv2_svm": "SVM", "rbv2_ranger": "RandomForest"}
CACHE = os.path.join("cache", "crossfam_M300.npz")
N_PER = 12
M = 300


def get_crossfam():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return {k: (d[f"{k}__s"], d[f"{k}__c"], str(d[f"{k}__fam"])) for k in d["__keys__"]}
    from benchmark import _bench
    suite, save, keys = {}, {}, []
    for sc, fam in FAMILIES.items():
        tasks = list(_bench(sc).instances)[:N_PER]
        for t in tasks:
            key = f"{sc}:{t}"
            s, c = build_landscape(sc, t, M=M, seed=0)
            suite[key] = (s, c, fam)
            save[f"{key}__s"] = s; save[f"{key}__c"] = c; save[f"{key}__fam"] = fam
            keys.append(key)
            print(f"  {key}: best={s.max():.4f} med={np.median(s):.4f} "
                  f"cost-med={np.median(c):.3f}", flush=True)
    save["__keys__"] = np.array(keys)
    np.savez(CACHE, **save)
    return suite


def main():
    suite = get_crossfam()
    lam_mults = [1.0, 2.0, 4.0, 8.0]
    by_fam = {f: {m: [] for m in lam_mults} for f in set(FAMILIES.values())}
    overall = {m: [] for m in lam_mults}

    for ki, (key, (s, c, fam)) in enumerate(suite.items()):
        spread = max(s.max() - np.median(s), 1e-6)
        base_lam = 0.03 * spread / c.mean()
        for m in lam_mults:
            lam = base_lam * m
            ot, orr, _ = eval_rule(s, c, lambda: MarginalStopping("evt", k_min=4), lam)
            bpts = baseline_points(s, c, lam)
            bt = time_at_regret(bpts, orr)
            if np.isfinite(bt) and bt > 0:
                v = 100 * (1 - ot / bt)
                by_fam[fam][m].append(v); overall[m].append(v)
        print(f"  [{ki+1}/{len(suite)}] {key}", flush=True)

    print(f"\n===== CROSS-FAMILY: agnostic EVT stopper, matched-regret saving % =====")
    print(f"{'lam x':>6} | " + " | ".join(f"{f:>14}" for f in by_fam) + f" | {'OVERALL':>16}")
    for m in lam_mults:
        cells = " | ".join(
            f"{np.median(by_fam[f][m]):6.1f}(w{np.mean(np.array(by_fam[f][m])>0):3.0%})"
            for f in by_fam)
        ov = np.array(overall[m])
        try:
            p = sstats.wilcoxon(ov).pvalue
        except Exception:
            p = float("nan")
        print(f"{m:6} | {cells} | {np.median(ov):6.1f}(w{np.mean(ov>0):3.0%}) p={p:.1e}")
    print("\n(positive => agnostic EVT rule saves training time vs hindsight baseline,"
          " confirming generalization beyond MLPs)")


if __name__ == "__main__":
    main()
