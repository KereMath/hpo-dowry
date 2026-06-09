"""
M3 -- meta-learned forward-looking stopping.

Replace the myopic one-step rule with a policy trained to imitate the
HINDSIGHT-OPTIMAL stop t* = argmax_t U(t), where U(t) = max(scores[:t]) -
cumulative_cost[:t] (costs already in score units, so the cost level is encoded
in their magnitude -- no separate lambda needed at decision time).

Everything is computed causally from the observed prefix, so the trained policy
is deployable. Evaluation is LEAVE-ONE-TASK-OUT to test generalization to unseen
tasks (the honest meta-learning protocol).
"""

import os

import numpy as np
from scipy import stats as sstats
from sklearn.ensemble import HistGradientBoostingClassifier

from benchmark import get_suite
from bo_substrate import get_bo_suite
from core import GPDTail, StoppingRule, run_trace
from experiment_suite import OracleMarginal, baseline_points, eval_rule, time_at_regret

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
LAM_MULTS = [0.5, 1.0, 2.0, 4.0, 8.0]
K_MIN = 4
STEP_CAP = 60

FEATURES = [
    "t", "since_improve_frac", "n_improve_frac", "incumbent", "z_incumbent",
    "std", "iqr", "slope", "ei", "c_next", "gap", "log_ratio", "cumcost", "ei_over_std",
]


def step_features(scores, sucosts):
    """Causal feature vector from the observed prefix (costs in score units)."""
    t = len(scores)
    b = scores.max()
    mean = scores.mean()
    std = scores.std() + 1e-9
    q75, q25 = np.percentile(scores, [75, 25])
    arg = int(np.argmax(scores))
    since = (t - 1 - arg) / max(t, 1)
    n_imp = np.sum(scores == np.maximum.accumulate(scores)) / max(t, 1)
    half = max(1, t // 2)
    slope = b - scores[:half].max()
    ei = GPDTail().fit(scores).expected_improvement(b) if t >= 2 else std
    c_next = sucosts.mean()
    gap = ei - c_next
    log_ratio = np.log((ei + 1e-12) / (c_next + 1e-12))
    return np.array([
        t, since, n_imp, b, (b - mean) / std, std, q75 - q25, slope,
        ei, c_next, gap, log_ratio, sucosts.sum(), ei / std,
    ], dtype=float)


def _optimal_stop(scores, sucosts):
    util = np.maximum.accumulate(scores) - np.cumsum(sucosts)
    return int(np.argmax(util))  # 0-based index of the optimal stop step


def _trace_rows(scores, sucosts):
    """Emit (feature, label) rows for decision steps k_min..min(T,STEP_CAP)."""
    T = min(len(scores), STEP_CAP)
    tstar = _optimal_stop(scores[:T], sucosts[:T])
    X, y = [], []
    for t in range(K_MIN, T):
        X.append(step_features(scores[:t], sucosts[:t]))
        y.append(1 if t >= tstar else 0)
    return X, y


def build_meta_dataset(scenario="lcbench", n_rs=40, seed=0):
    cache = os.path.join(CACHE_DIR, f"meta_{scenario}_rs{n_rs}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], d["groups"]
    land = get_suite(scenario, M=400, seed=0)
    bo = get_bo_suite(scenario)
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for gi, t in enumerate(land.keys()):
        s, c = land[t]
        spread = max(s.max() - np.median(s), 1e-6)
        base_lam = 0.03 * spread / c.mean()
        # random-search traces
        for _ in range(n_rs):
            perm = rng.permutation(len(s))
            for m in LAM_MULTS:
                su = (base_lam * m) * c[perm]
                rx, ry = _trace_rows(s[perm], su)
                X += rx; y += ry; groups += [gi] * len(ry)
        # BO traces
        if t in bo:
            S, C = bo[t]
            blam = 0.03 * max(S.max() - np.median(S[:, 0]), 1e-6) / C.mean()
            for si in range(S.shape[0]):
                for m in LAM_MULTS:
                    su = (blam * m) * C[si]
                    rx, ry = _trace_rows(S[si], su)
                    X += rx; y += ry; groups += [gi] * len(ry)
        print(f"  meta rows so far after task {gi+1}: {len(y)}", flush=True)
    X = np.array(X); y = np.array(y); groups = np.array(groups)
    np.savez(cache, X=X, y=y, groups=groups)
    return X, y, groups


class MetaStopping(StoppingRule):
    """Forward-looking learned stopping policy (deployable, causal)."""

    def __init__(self, model, k_min=K_MIN):
        self.model = model
        self.k_min = k_min
        self.reset()

    def should_stop(self):
        n = len(self.scores)
        if n < self.k_min:
            return False
        f = step_features(np.array(self.scores), np.array(self.costs))
        return int(self.model.predict(f.reshape(1, -1))[0]) == 1


def loto_evaluate(scenario="lcbench"):
    X, y, groups = build_meta_dataset(scenario)
    land = get_suite(scenario, M=400, seed=0)
    tasks = list(land.keys())
    print(f"\nmeta-dataset: {len(y)} rows, {X.shape[1]} features, "
          f"positive rate={y.mean():.3f}")

    # Leave-one-task-out: train on all other tasks, evaluate the learned policy
    # on the held-out task by actually running it (random-search orderings).
    save_meta = {m: [] for m in LAM_MULTS}
    save_myopic = {m: [] for m in LAM_MULTS}
    from core import MarginalStopping
    for gi, t in enumerate(tasks):
        mask = groups != gi
        clf = HistGradientBoostingClassifier(
            max_depth=4, max_iter=150, learning_rate=0.08, l2_regularization=1.0)
        clf.fit(X[mask], y[mask])

        s, c = land[t]
        spread = max(s.max() - np.median(s), 1e-6)
        base_lam = 0.03 * spread / c.mean()
        for m in LAM_MULTS:
            lam = base_lam * m
            bpts = baseline_points(s, c, lam)
            # held-out task: run META policy and MYOPIC rule
            mt, mr = eval_rule(s, c, lambda: MetaStopping(clf), lam)
            yt, yr = eval_rule(s, c, lambda: MarginalStopping("evt", k_min=4), lam)
            for (rt, rr), store in [((mt, mr), save_meta), ((yt, yr), save_myopic)]:
                bt = time_at_regret(bpts, rr)
                if np.isfinite(bt) and bt > 0:
                    store[m].append(100 * (1 - rt / bt))
        print(f"  [LOTO {gi+1}/{len(tasks)}] {t}", flush=True)

    print(f"\n===== M3 LOTO verdict: {scenario} =====")
    print("matched-regret time saving %  [median (win-rate)]  -- META(forward-looking) vs MYOPIC(evt,k=4)")
    print(f"{'lam x':>7} | {'META':>22} | {'MYOPIC':>22} | {'paired p':>10}")
    for m in LAM_MULTS:
        a = np.array(save_meta[m]); b = np.array(save_myopic[m])
        try:
            p = sstats.wilcoxon(a - b).pvalue
        except Exception:
            p = float("nan")
        print(f"{m:7} | {np.median(a):8.1f} (w{np.mean(a>0):3.0%}) {'':6}| "
              f"{np.median(b):8.1f} (w{np.mean(b>0):3.0%}) {'':6}| {p:10.2e}")


if __name__ == "__main__":
    loto_evaluate("lcbench")
