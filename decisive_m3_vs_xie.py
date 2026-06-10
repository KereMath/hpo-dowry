"""
The decisive experiment: on a GP-BO substrate (the competitor's home turf),
compare three stopping rules against a PROPER GP-BO-trace baseline envelope:

  - XIE   : Pandora's-Box Gittins-Index stopping (max_x g(x) <= incumbent)  [Xie 2025]
  - MYOPIC: our optimizer-agnostic EVT one-step rule (EI <= cost)
  - META  : our M3 meta-learned forward-looking policy (LOTO: trained on OTHER tasks)

Fixes the earlier caveat: the matched-regret baseline is now fixed-N / patience computed
ON THE GP-BO TRACES (the selection order), not random-search orderings.

Question: can the forward-looking learned policy (META) beat the myopic PBGI rule (XIE)
on the GP-BO substrate where PBGI uses the sharp GP posterior?
"""

import numpy as np
from scipy import stats as sstats
from sklearn.ensemble import HistGradientBoostingRegressor

from core import FixedBudget, MarginalStopping, Patience, run_trace
from meta_stopping import MetaStopping, build_meta_dataset
from xie_baseline import gp_pbgi_trace, sample_pool, xie_stop_index


def baseline_env_traces(traces, lam, ref_best):
    pts = []
    maxlen = max(len(so) for so, _ in traces)
    for N in list(range(3, 30, 2)) + list(range(30, maxlen + 1, 6)):
        ts, rs = [], []
        for so, co in traces:
            n = min(N, len(so))
            ts.append(co[:n].sum()); rs.append(ref_best - so[:n].max())
        pts.append((np.mean(ts), np.mean(rs)))
    for p in [3, 5, 8, 12, 18, 25]:
        ts, rs = [], []
        for so, co in traces:
            o = run_trace(so, lam * co, Patience(p))
            n = o["n_evals"]
            ts.append(co[:n].sum()); rs.append(ref_best - o["incumbent"])
        pts.append((np.mean(ts), np.mean(rs)))
    return pts


def time_at_regret(pts, target):
    cand = [t for t, r in pts if r <= target + 1e-9]
    return min(cand) if cand else np.inf


def saving(rule_time, rule_regret, bpts):
    bt = time_at_regret(bpts, rule_regret)
    return 100 * (1 - rule_time / bt) if np.isfinite(bt) and bt > 0 else np.nan


def main(scenario="lcbench", n_tasks=12, n_seeds=8, n_steps=60):
    # meta-dataset for the LOTO META policy
    X, y, groups = build_meta_dataset(scenario)
    from benchmark import _bench
    all_tasks = list(_bench(scenario).instances)
    tasks = all_tasks[:n_tasks]
    lam_mults = [1.0, 2.0, 4.0, 8.0]
    S = {r: {m: [] for m in lam_mults} for r in ("XIE", "MYOPIC", "META")}

    for ti, t in enumerate(tasks):
        gi = all_tasks.index(t)
        clf = HistGradientBoostingRegressor(max_depth=4, max_iter=200,
                                            learning_rate=0.08, l2_regularization=1.0)
        clf.fit(X[groups != gi], y[groups != gi])     # LOTO: held-out task excluded

        F, s, c = sample_pool(scenario, t, M=400, seed=0)
        ref_best = s.max()
        spread = max(s.max() - np.median(s), 1e-6)
        base_lam = 0.03 * spread / c.mean()
        for m in lam_mults:
            lam = base_lam * m
            traces, xie_pts = [], []
            for si in range(n_seeds):
                so, co, gmax, inc = gp_pbgi_trace(F, s, c, lam, n_steps, seed=si)
                traces.append((so, co))
                xs = min(xie_stop_index(gmax, inc), len(so))
                xie_pts.append((co[:xs].sum(), ref_best - so[:xs].max()))
            bpts = baseline_env_traces(traces, lam, ref_best)

            # XIE
            xt = np.mean([p[0] for p in xie_pts]); xr = np.mean([p[1] for p in xie_pts])
            S["XIE"][m].append(saving(xt, xr, bpts))
            # MYOPIC and META on the same GP-BO traces
            for label, mk in [("MYOPIC", lambda: MarginalStopping("evt", k_min=4)),
                              ("META", lambda: MetaStopping(clf))]:
                ts, rs = [], []
                for so, co in traces:
                    o = run_trace(so, lam * co, mk())
                    n = o["n_evals"]
                    ts.append(co[:n].sum()); rs.append(ref_best - o["incumbent"])
                S[label][m].append(saving(np.mean(ts), np.mean(rs), bpts))
        print(f"  [{ti+1}/{len(tasks)}] {t}", flush=True)

    # persist per-task savings for the rigor layer (CD diagram, significance)
    import os
    save_npz = {f"{r}_{m}": np.array(S[r][m]) for r in S for m in lam_mults}
    save_npz["lam_mults"] = np.array(lam_mults)
    save_npz["tasks"] = np.array(tasks)
    np.savez(os.path.join("cache", f"decisive_{scenario}_{len(tasks)}t.npz"), **save_npz)

    print(f"\n===== DECISIVE: stopping rules on GP-BO substrate, {len(tasks)} tasks =====")
    print("matched-regret time saving % vs GP-BO-trace baseline  [median (win-rate)]")
    print(f"{'lam x':>6} | {'XIE (PBGI)':>16} | {'MYOPIC (EVT)':>16} | {'META (M3, LOTO)':>17} | {'META>XIE p':>11}")
    for m in lam_mults:
        x = np.array([v for v in S["XIE"][m] if not np.isnan(v)])
        my = np.array([v for v in S["MYOPIC"][m] if not np.isnan(v)])
        me = np.array([v for v in S["META"][m] if not np.isnan(v)])
        # paired META vs XIE (same tasks, drop nan in either)
        xa = np.array(S["XIE"][m]); ma = np.array(S["META"][m])
        ok = ~(np.isnan(xa) | np.isnan(ma))
        try:
            p = sstats.wilcoxon(ma[ok] - xa[ok]).pvalue if ok.sum() > 2 else float("nan")
        except Exception:
            p = float("nan")
        print(f"{m:6} | {np.median(x):7.1f} (w{np.mean(x>0):3.0%}) | "
              f"{np.median(my):7.1f} (w{np.mean(my>0):3.0%}) | "
              f"{np.median(me):7.1f} (w{np.mean(me>0):3.0%})  | {p:11.2e}")


if __name__ == "__main__":
    main()
