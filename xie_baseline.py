"""
Head-to-head against the primary competitor: Xie et al. 2025,
"Cost-aware Stopping for Bayesian Optimization" (arXiv:2507.12453).

Their stopping rule is the Pandora's-Box optimal policy applied to a GP
surrogate: maintain the Pandora's-Box Gittins Index (PBGI) g(x) of each
candidate -- the reservation value solving E[(Y_x - g)^+] = lambda * c, with
Y_x ~ GP posterior N(mu(x), sigma(x)) -- select argmax_x g(x), and STOP when
max_x g(x) <= incumbent (no candidate's fair value beats what we already have).

This is sharper than our rule because it uses the surrogate's per-candidate
posterior, but it REQUIRES a surrogate, so it is tied to GP-BO. We:
  (1) implement a faithful GP-BO + PBGI acquisition + PBGI stopping;
  (2) run our optimizer-agnostic EVT rule on the SAME GP-BO traces;
  (3) compare matched-regret time saving;
  (4) make the optimizer-agnostic point explicit: on random search there is no
      surrogate, so the PBGI rule does not apply at all, while ours does.
"""

import os

import numpy as np
from scipy import stats as sstats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from benchmark import _bench
from core import MarginalStopping, run_trace
from experiment_suite import baseline_points, eval_rule, time_at_regret

# ---- psi(z) = phi(z) + z*Phi(z);  E[(Y-g)^+] = sigma*psi((mu-g)/sigma) ----
_ZGRID = np.linspace(-9, 12, 4000)
_PSI = sstats.norm.pdf(_ZGRID) + _ZGRID * sstats.norm.cdf(_ZGRID)


def _psi_inv(tau):
    """Invert psi (increasing, psi(-inf)=0) at targets tau>=0 -> d."""
    tau = np.clip(tau, _PSI[0], _PSI[-1])
    return np.interp(tau, _PSI, _ZGRID)


def gittins_index(mu, sigma, lam_c):
    """PBGI reservation value g: E[(Y-g)^+]=lam_c, Y~N(mu,sigma)."""
    sigma = np.maximum(sigma, 1e-9)
    d = _psi_inv(lam_c / sigma)
    return mu - sigma * d


def _encode(cfgs, cs, num_names, cat_names, cat_maps):
    rows = []
    for cfg in cfgs:
        v = []
        for n in num_names:
            hp = cs.get_hyperparameter(n)
            x = float(cfg[n])
            if getattr(hp, "log", False):
                x = np.log(max(x, 1e-12))
            v.append(x)
        for n in cat_names:
            v.append(cat_maps[n].get(cfg[n], 0))
        rows.append(v)
    X = np.array(rows, dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X


def sample_pool(scenario, instance, M=400, seed=0,
                score_key="val_accuracy", cost_key="time", score_div=100.0):
    b = _bench(scenario)
    b.set_instance(instance)
    cs = b.get_opt_space()
    try:
        cs.seed(seed)
    except Exception:
        pass
    fixed = {}
    for fp in b.config.fidelity_params:
        hp = cs.get_hyperparameter(fp)
        fixed[fp] = hp.upper if hasattr(hp, "upper") else hp.sequence[-1]
    fixed[b.config.instance_names] = instance
    num_names, cat_names, cat_maps = [], [], {}
    for hp in cs.get_hyperparameters():
        if hp.name in fixed:
            continue
        if hasattr(hp, "choices"):
            cat_names.append(hp.name)
            cat_maps[hp.name] = {c: i for i, c in enumerate(hp.choices)}
        elif hasattr(hp, "lower"):
            num_names.append(hp.name)
    cfgs = []
    for _ in range(M):
        d = cs.sample_configuration().get_dictionary()  # always valid (respects conditionals)
        d.update(fixed)
        cfgs.append(d)
    out = b.objective_function(cfgs)
    s = np.array([o[score_key] / score_div for o in out])
    c = np.array([max(o[cost_key], 1e-6) for o in out])
    # missing (inactive) features -> 0 before encoding
    for d in cfgs:
        for n in num_names + cat_names:
            d.setdefault(n, 0)
    F = _encode(cfgs, cs, num_names, cat_names, cat_maps)
    return F, s, c


def gp_pbgi_trace(F, s, c, lam, n_steps=60, k0=4, seed=0):
    """GP-BO with PBGI acquisition. Returns selection-order scores, costs,
    and per-step (max Gittins index, incumbent) used by the Xie stopping rule."""
    rng = np.random.default_rng(seed)
    M = len(s)
    n_steps = min(n_steps, M)
    sel = list(rng.choice(M, size=k0, replace=False))
    remaining = set(range(M)) - set(sel)
    gmax_hist, inc_hist = [], []
    kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)
    while len(sel) < n_steps and remaining:
        rem = np.array(sorted(remaining))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=0, alpha=1e-6)
        gp.fit(F[sel], s[sel])
        mu, sigma = gp.predict(F[rem], return_std=True)
        lam_c = lam * c[rem]                       # cost-aware reservation
        g = gittins_index(mu, sigma, lam_c)
        inc = s[sel].max()
        gmax_hist.append(float(g.max()))
        inc_hist.append(float(inc))
        nxt = rem[int(np.argmax(g))]
        sel.append(int(nxt))
        remaining.discard(int(nxt))
    sel = np.array(sel)
    return s[sel], c[sel], np.array(gmax_hist), np.array(inc_hist)


def xie_stop_index(gmax, inc, k_min=4):
    """First step (>=k_min) where max Gittins index <= incumbent."""
    for t in range(max(k_min, 1), len(gmax) + 1):
        if t - 1 < len(gmax) and gmax[t - 1] <= inc[t - 1]:
            return t
    return len(inc) + k_min  # never triggered -> ran full trace


def main(scenario="lcbench", n_tasks=12, n_seeds=8, n_steps=60):
    b = _bench(scenario)
    tasks = list(b.instances)[:n_tasks]
    lam_mults = [1.0, 2.0, 4.0, 8.0]
    save_ours = {m: [] for m in lam_mults}
    save_xie = {m: [] for m in lam_mults}

    for ti, t in enumerate(tasks):
        F, s, c = sample_pool(scenario, t, M=400, seed=0)
        spread = max(s.max() - np.median(s), 1e-6)
        base_lam = 0.03 * spread / c.mean()
        ref_best = s.max()
        for m in lam_mults:
            lam = base_lam * m
            # build GP-BO + PBGI traces (regenerated per lambda: acquisition depends on it)
            ours_t, ours_r, xie_t, xie_r = [], [], [], []
            for si in range(n_seeds):
                so, co, gmax, inc = gp_pbgi_trace(F, s, c, lam, n_steps, seed=si)
                # Xie stop
                xs = xie_stop_index(gmax, inc)
                xs = min(xs, len(so))
                xie_t.append(co[:xs].sum()); xie_r.append(ref_best - so[:xs].max())
                # our EVT rule on the SAME trace
                o = run_trace(so, lam * co, MarginalStopping("evt", k_min=4))
                n = o["n_evals"]
                ours_t.append(co[:n].sum()); ours_r.append(ref_best - o["incumbent"])
            # baseline envelope on these GP-BO traces (fixed-N / patience over the trace)
            # reuse simple fixed-N envelope via the landscape ordering proxy:
            bpts = baseline_points(s, c, lam, n_orderings=150)
            for (rt, rr), store in [((np.mean(ours_t), np.mean(ours_r)), save_ours),
                                    ((np.mean(xie_t), np.mean(xie_r)), save_xie)]:
                bt = time_at_regret(bpts, rr)
                if np.isfinite(bt) and bt > 0:
                    store[m].append(100 * (1 - rt / bt))
        print(f"  [{ti+1}/{len(tasks)}] {t}", flush=True)

    print(f"\n===== Xie-2025 (PBGI) vs OURS (EVT), GP-BO substrate, {len(tasks)} tasks =====")
    print("matched-regret time saving %  [median (win-rate)]")
    print(f"{'lam x':>7} | {'OURS (EVT, agnostic)':>22} | {'XIE (PBGI, GP-only)':>22} | {'paired p':>10}")
    for m in lam_mults:
        a = np.array(save_ours[m]); x = np.array(save_xie[m])
        try:
            p = sstats.wilcoxon(a - x).pvalue if len(a) == len(x) and len(a) > 2 else float("nan")
        except Exception:
            p = float("nan")
        print(f"{m:7} | {np.median(a):8.1f} (w{np.mean(a>0):3.0%}) {'':5}| "
              f"{np.median(x):8.1f} (w{np.mean(x>0):3.0%}) {'':5}| {p:10.2e}")
    print("\nNote: on a RANDOM-SEARCH substrate the PBGI rule has no surrogate and "
          "cannot be computed at all; ours runs unchanged (the optimizer-agnostic point).")


if __name__ == "__main__":
    main()
