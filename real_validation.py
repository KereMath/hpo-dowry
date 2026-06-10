"""
Live-training validation (no surrogate).

Everything else in this repo uses YAHPO surrogate benchmarks. Here we close the
loop on REAL model training: run Optuna TPE with a full budget vs. with the
cost-aware EVTStopper, training actual scikit-learn models with cross-validation,
and measure the real wall-clock saved and the actual quality lost -- across
several (dataset, model) combinations.
"""

import time

import numpy as np

import optuna
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from optuna_plugin import EVTStopper

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATASETS = {"digits": load_digits, "wine": load_wine, "breast_cancer": load_breast_cancer}


def make_objective(model, X, y):
    def objective(trial):
        if model == "rf":
            clf = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 20, 400),
                max_depth=trial.suggest_int("max_depth", 2, 20),
                max_features=trial.suggest_float("max_features", 0.2, 1.0),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 16),
                random_state=0, n_jobs=1)
        elif model == "gb":
            clf = GradientBoostingClassifier(
                n_estimators=trial.suggest_int("n_estimators", 20, 300),
                max_depth=trial.suggest_int("max_depth", 2, 6),
                learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                random_state=0)
        else:  # svc
            clf = SVC(
                C=trial.suggest_float("C", 1e-2, 1e2, log=True),
                gamma=trial.suggest_float("gamma", 1e-4, 1e0, log=True))
        return cross_val_score(clf, X, y, cv=3).mean()
    return objective


def run(model, ds_name, loader, budget=30, frac=0.02):
    X, y = loader(return_X_y=True)
    obj = make_objective(model, X, y)

    t0 = time.perf_counter()
    full = optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=0))
    full.optimize(obj, n_trials=budget, show_progress_bar=False)
    t_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(obj, n_trials=budget, callbacks=[EVTStopper(frac=frac)],
                show_progress_bar=False)
    t_stop = time.perf_counter() - t0

    return dict(model=model, ds=ds_name,
                full_trials=len(full.trials), stop_trials=len(st.trials),
                full_best=full.best_value, stop_best=st.best_value,
                t_full=t_full, t_stop=t_stop)


def main():
    rows = []
    for model in ["rf", "gb", "svc"]:
        for ds, loader in DATASETS.items():
            r = run(model, ds, loader)
            rows.append(r)
            print(f"  {model:4} / {ds:13}: full {r['full_trials']:3d} trials "
                  f"({r['t_full']:6.1f}s, best {r['full_best']:.4f}) -> "
                  f"stop {r['stop_trials']:3d} ({r['t_stop']:6.1f}s, best {r['stop_best']:.4f})",
                  flush=True)

    print("\n===== LIVE-TRAINING VALIDATION (real sklearn models) =====")
    print(f"{'model/dataset':20} {'time saved':>11} {'trials saved':>13} {'quality lost':>13}")
    ts, qs = [], []
    for r in rows:
        tsave = 100 * (1 - r["t_stop"] / r["t_full"])
        trsave = 100 * (1 - r["stop_trials"] / r["full_trials"])
        qloss = r["full_best"] - r["stop_best"]
        ts.append(tsave); qs.append(qloss)
        print(f"{r['model']+'/'+r['ds']:20} {tsave:10.0f}% {trsave:12.0f}% {qloss:+13.4f}")
    print("-" * 60)
    print(f"{'MEAN':20} {np.mean(ts):10.0f}% {'':13} {np.mean(qs):+13.4f}")
    print(f"\n=> mean {np.mean(ts):.0f}% less wall-clock on live training "
          f"for {np.mean(qs):+.4f} mean accuracy change (frac=0.02)")


if __name__ == "__main__":
    main()
