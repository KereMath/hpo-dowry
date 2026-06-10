"""
Product track: a drop-in cost-aware stopping callback for Optuna.

Add `EVTStopper()` to `study.optimize(..., callbacks=[stopper])` and the study
halts automatically once one more trial is no longer worth its training cost --
no need to guess `n_trials`. Parameter-free by default: it is willing to pay a
fraction `frac` of the observed score spread per evaluation, and stops when the
EVT-estimated expected improvement of the incumbent drops below that.

  import optuna
  from optuna_plugin import EVTStopper
  study = optuna.create_study(direction="maximize")
  study.optimize(objective, n_trials=500, callbacks=[EVTStopper(frac=0.05)])

`MetaStopper(model_path)` instead uses a trained forward-looking policy (M3).
"""

import numpy as np

from core import GPDTail

try:
    import optuna
except Exception:
    optuna = None


class EVTStopper:
    """Cost-aware EVT stopping callback. Stops the study when EI(incumbent) <= cost.

    frac: per-evaluation price as a fraction of the observed score spread
          (the single intuitive knob; smaller => search longer).
    k_min: minimal seed before stopping is allowed.
    cost_per_eval: optional explicit per-eval cost in score units (overrides frac).
    """

    def __init__(self, frac=0.05, k_min=4, cost_per_eval=None):
        self.frac = frac
        self.k_min = k_min
        self.cost_per_eval = cost_per_eval
        self.values = []

    def _val(self, study, trial):
        v = trial.value
        if v is None:
            return None
        return v if study.direction == optuna.study.StudyDirection.MAXIMIZE else -v

    def __call__(self, study, trial):
        v = self._val(study, trial)
        if v is not None:
            self.values.append(float(v))
        n = len(self.values)
        if n < self.k_min:
            return
        s = np.array(self.values)
        best = s.max()
        spread = max(best - np.median(s), 1e-9)
        c_next = self.cost_per_eval if self.cost_per_eval is not None else self.frac * spread
        ei = GPDTail().fit(s).expected_improvement(best)
        if ei <= c_next:
            study.stop()


class MetaStopper:
    """Forward-looking learned stopping callback (M3). Loads a saved policy."""

    def __init__(self, model_path, k_min=4, frac=0.05):
        import joblib
        self.model = joblib.load(model_path)
        self.k_min = k_min
        self.frac = frac
        self.values, self.costs = [], []

    def __call__(self, study, trial):
        from meta_stopping import step_features
        v = trial.value
        if v is None:
            return
        if study.direction != optuna.study.StudyDirection.MAXIMIZE:
            v = -v
        dur = 1.0
        if trial.datetime_complete and trial.datetime_start:
            dur = max((trial.datetime_complete - trial.datetime_start).total_seconds(), 1e-6)
        self.values.append(float(v)); self.costs.append(dur)
        n = len(self.values)
        if n < self.k_min:
            return
        s = np.array(self.values)
        spread = max(s.max() - np.median(s), 1e-9)
        lam = self.frac * spread / max(np.mean(self.costs), 1e-9)
        su = lam * np.array(self.costs)
        f = step_features(s, su)
        if float(self.model.predict(f.reshape(1, -1))[0]) <= 0.0:
            study.stop()


# --------------------------------------------------------------------------- #
# Demo: real sklearn study, with vs without the cost-aware stopper.            #
# --------------------------------------------------------------------------- #
def _demo():
    import time
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X, y = load_digits(return_X_y=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 20, 400),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            max_features=trial.suggest_float("max_features", 0.2, 1.0),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 16),
            random_state=0, n_jobs=1)
        return cross_val_score(clf, X, y, cv=3).mean()

    BUDGET = 100
    t0 = time.perf_counter()
    full = optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=0))
    full.optimize(objective, n_trials=BUDGET, show_progress_bar=False)
    t_full = time.perf_counter() - t0

    print(f"\n=== Optuna cost-aware stopping demo (RandomForest / digits) ===")
    print(f"full budget : {len(full.trials):3d} trials, best={full.best_value:.4f}, "
          f"{t_full:6.1f}s  (reference)")
    print(f"{'frac':>7} | {'trials':>6} | {'best':>7} | {'wall-clock':>10} | "
          f"{'time saved':>10} | {'quality lost':>12}")
    # the single knob `frac` traces a quality-vs-cost operating curve
    for frac in [0.10, 0.03, 0.01, 0.003]:
        t0 = time.perf_counter()
        st = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=0))
        st.optimize(objective, n_trials=BUDGET, callbacks=[EVTStopper(frac=frac)],
                    show_progress_bar=False)
        dt = time.perf_counter() - t0
        print(f"{frac:7.3f} | {len(st.trials):6d} | {st.best_value:7.4f} | "
              f"{dt:9.1f}s | {100*(1-dt/t_full):9.0f}% | "
              f"{full.best_value-st.best_value:+12.4f}")


if __name__ == "__main__":
    if optuna is None:
        raise SystemExit("pip install optuna")
    _demo()
