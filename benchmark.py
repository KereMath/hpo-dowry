"""
Standardized HPO-landscape loader built on YAHPO Gym surrogate benchmarks.

A "landscape" for a task = M sampled hyperparameter configurations evaluated at
full fidelity, each giving (score, cost):
    score = validation accuracy (higher is better)
    cost  = surrogate-predicted training time in seconds (the real cost)

Random search over a task = a random ordering of its landscape. This is the
substrate for the cost-aware stopping experiments.

Tasks come from real AutoML benchmarks (LCBench = Auto-PyTorch MLPs over 34
OpenML datasets; rbv2_* = classic ML model families over many datasets), each
with genuine recorded runtime -> heterogeneous, realistic cost.
"""

import os

import numpy as np

DATA_PATH = "c:/tmp/yahpo_data"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_BENCH = {}


def _bench(scenario):
    if scenario not in _BENCH:
        from yahpo_gym import BenchmarkSet, local_config
        local_config.init_config()
        local_config.set_data_path(DATA_PATH)
        _BENCH[scenario] = BenchmarkSet(scenario)
    return _BENCH[scenario]


# score target (higher=better) and cost target per scenario
SCORE_KEY = {
    "lcbench": "val_accuracy",
    "rbv2_ranger": "acc", "rbv2_svm": "acc", "rbv2_xgboost": "acc",
    "rbv2_rpart": "acc", "rbv2_glmnet": "acc", "rbv2_aknn": "acc",
}
COST_KEY = {
    "lcbench": "time",
    "rbv2_ranger": "timetrain", "rbv2_svm": "timetrain", "rbv2_xgboost": "timetrain",
    "rbv2_rpart": "timetrain", "rbv2_glmnet": "timetrain", "rbv2_aknn": "timetrain",
}


def list_tasks(scenario):
    return list(_bench(scenario).instances)


def build_landscape(scenario, instance, M=400, seed=0):
    """Return (scores in [0,1], costs in seconds) for M configs at full fidelity."""
    b = _bench(scenario)
    b.set_instance(instance)
    cs = b.get_opt_space()
    rng = np.random.default_rng(seed)
    # ConfigSpace sampling uses its own RNG; seed it for reproducibility
    try:
        cs.seed(int(seed))
    except Exception:
        pass

    # set fidelity to max
    fid = b.config.fidelity_params  # e.g. ['epoch'] or ['trainsize','repl']
    fid_max = {}
    for fp in fid:
        hp = cs.get_hyperparameter(fp)
        fid_max[fp] = hp.upper if hasattr(hp, "upper") else hp.sequence[-1]

    cfgs = []
    for _ in range(M):
        d = cs.sample_configuration().get_dictionary()
        d[b.config.instance_names] = instance  # ensure correct task id
        for fp, v in fid_max.items():
            d[fp] = v
        cfgs.append(d)

    out = b.objective_function(cfgs)
    sk, ck = SCORE_KEY[scenario], COST_KEY[scenario]
    scores = np.array([o[sk] for o in out], dtype=float)
    costs = np.array([o[ck] for o in out], dtype=float)
    # lcbench accuracy is in [0,100]
    if scores.max() > 1.5:
        scores = scores / 100.0
    costs = np.clip(costs, 1e-6, None)
    return scores, costs


def get_suite(scenario="lcbench", M=400, seed=0, max_tasks=None):
    """Load (cached) landscapes for all tasks of a scenario.

    Returns dict: task_id -> (scores, costs).
    """
    cache = os.path.join(CACHE_DIR, f"{scenario}_M{M}_s{seed}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return {k: (d[f"{k}__s"], d[f"{k}__c"]) for k in d["__tasks__"]}

    tasks = list_tasks(scenario)
    if max_tasks:
        tasks = tasks[:max_tasks]
    suite, save = {}, {}
    for i, t in enumerate(tasks):
        s, c = build_landscape(scenario, t, M=M, seed=seed)
        suite[t] = (s, c)
        save[f"{t}__s"] = s
        save[f"{t}__c"] = c
        print(f"  [{i+1}/{len(tasks)}] {scenario}/{t}: "
              f"best={s.max():.4f} med={np.median(s):.4f} "
              f"cost med={np.median(c):.2f}s", flush=True)
    save["__tasks__"] = np.array(list(suite.keys()))
    np.savez(cache, **save)
    return suite


if __name__ == "__main__":
    import sys
    scenario = sys.argv[1] if len(sys.argv) > 1 else "lcbench"
    suite = get_suite(scenario, M=400, seed=0)
    print(f"\n{scenario}: {len(suite)} tasks cached.")
    spreads = [s.max() - np.median(s) for s, _ in suite.values()]
    print(f"score spread (best-median): mean={np.mean(spreads):.4f} "
          f"min={np.min(spreads):.4f} max={np.max(spreads):.4f}")
