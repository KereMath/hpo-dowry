"""
Cost-aware optimal stopping layer for hyperparameter optimization (HPO).

Idea: an HPO run evaluates configurations sequentially; each evaluation costs
training time. Keep the best-so-far incumbent `b`. The expected gain of doing
ONE more evaluation is the expected improvement EI(b) = E[(X - b)^+], where X is
the (unknown) score of a fresh configuration. The myopic-optimal / Weitzman
reservation rule says:

    STOP as soon as   EI(b) <= c_next         (cost of the next evaluation)

For i.i.d. draws this one-step rule coincides with the Pandora's-box reservation
value z solving E[(X - z)^+] = c. It is PARAMETER-FREE given the cost: the
threshold is derived from data + cost, not hand-tuned like a patience value.

The crux is estimating the UPPER TAIL of the score distribution above the
current incumbent from only a handful of observations. The empirical estimator
is degenerate (no mass above the max -> EI=0 -> stop immediately). We compare:
  - EmpiricalTail : naive, expected to fail
  - GaussianTail  : simple parametric baseline
  - GPDTail       : extreme-value-theory peaks-over-threshold (principled)
"""

import numpy as np
from scipy import integrate, stats


# --------------------------------------------------------------------------- #
# Tail models: each estimates EI(b) = E[(X - b)^+] from observed scores.       #
# Scores: higher is better.                                                    #
# --------------------------------------------------------------------------- #
class EmpiricalTail:
    """Naive plug-in. EI(b) = mean(max(x - b, 0)) over observed scores.

    Degenerate above the observed maximum (EI = 0), which is exactly the
    failure mode we want to expose: it makes the stopper quit the moment it
    sets a new incumbent.
    """

    name = "empirical"

    def fit(self, scores):
        self.scores = np.asarray(scores, dtype=float)
        return self

    def expected_improvement(self, b):
        return float(np.mean(np.maximum(self.scores - b, 0.0)))


class GaussianTail:
    """Fit X ~ N(mu, sigma); closed-form EI(b) = (mu-b)Phi(d) + sigma*phi(d)."""

    name = "gaussian"

    def fit(self, scores):
        s = np.asarray(scores, dtype=float)
        self.mu = float(s.mean())
        self.sigma = float(s.std(ddof=1)) if len(s) > 1 else 1e-9
        self.sigma = max(self.sigma, 1e-12)
        return self

    def expected_improvement(self, b):
        d = (self.mu - b) / self.sigma
        return float((self.mu - b) * stats.norm.cdf(d) + self.sigma * stats.norm.pdf(d))


class GPDTail:
    """Peaks-over-threshold EVT model of the upper tail (closed form, fast).

    Pick a high threshold u (quantile q of observed scores), model the
    exceedances (x - u) with a Generalized Pareto Distribution fitted by
    Probability-Weighted Moments (Hosking & Wallis 1987 -- closed form, robust
    for small samples, no iterative MLE), and use the closed-form expected
    improvement of a GPD tail. This is what makes the per-step stopping check
    cheap enough to run inside an HPO loop.

    For b >= u and shape xi < 1:
        EI(b) = p_u * beta/(1-xi) * (1 + xi*(b-u)/beta) ** (1 - 1/xi)
    with the xi -> 0 limit  EI(b) = p_u * beta * exp(-(b-u)/beta).
    """

    name = "evt"

    def __init__(self, q=0.6, n0=0, xi_cap=0.95):
        self.q = q
        self.n0 = n0          # shrink shape toward 0 (exponential) for small samples
        self.xi_cap = xi_cap  # hard cap on tail shape to keep the mean finite

    @staticmethod
    def _pwm_gpd(y):
        """Closed-form GPD (xi, beta) from exceedances y > 0 via PWM."""
        y = np.sort(np.asarray(y, dtype=float))
        m = len(y)
        a0 = y.mean()
        j = np.arange(1, m + 1)
        a1 = np.sum(((m - j) / (m - 1)) * y) / m  # E[Y (1-F)] estimator
        denom = a0 - 2.0 * a1
        if abs(denom) < 1e-12:
            return 0.0, max(a0, 1e-12)  # exponential fallback
        xi = 2.0 - a0 / denom          # scipy genpareto shape convention
        beta = 2.0 * a0 * a1 / denom
        if not np.isfinite(beta) or beta <= 0:
            return 0.0, max(a0, 1e-12)
        return float(xi), float(beta)

    def fit(self, scores):
        s = np.asarray(scores, dtype=float)
        self.s = s
        self.u = float(np.quantile(s, self.q))
        exc = s[s > self.u] - self.u
        self.p_u = float(np.mean(s > self.u))
        if len(exc) >= 5 and exc.std() > 0:
            self.xi, self.beta = self._pwm_gpd(exc)
        else:
            self.xi = 0.0
            self.beta = max(s.std(ddof=1) if len(s) > 1 else 1e-9, 1e-9)
        # small-sample guard: shrink the shape toward 0 (exponential tail) so a
        # noisy estimate from few exceedances cannot produce a runaway heavy tail
        # (the failure mode that makes a stopper never stop).
        if self.n0 > 0:
            m_exc = max(int(np.sum(s > self.u)), 1)
            self.xi *= m_exc / (m_exc + self.n0)
        self.xi = float(np.clip(self.xi, -1.0, self.xi_cap))
        self.beta = max(self.beta, 1e-12)
        return self

    def _ei_above_u(self, b):
        """Closed-form EI for b >= u."""
        t = b - self.u
        if abs(self.xi) < 1e-6:
            return self.p_u * self.beta * np.exp(-t / self.beta)
        base = 1.0 + self.xi * t / self.beta
        if base <= 0:
            return 0.0
        return self.p_u * self.beta / (1.0 - self.xi) * base ** (1.0 - 1.0 / self.xi)

    def expected_improvement(self, b):
        if b >= self.u:
            return float(max(self._ei_above_u(b), 0.0))
        # b < u: empirical mass in [b, u], then stitch the EVT tail at u.
        emp_above_b = float(np.mean(np.maximum(self.s - b, 0.0)))
        emp_above_u = float(np.mean(np.maximum(self.s - self.u, 0.0)))
        return float(max(emp_above_b - emp_above_u + self._ei_above_u(self.u), 0.0))


class GuardedTail(GPDTail):
    """EVT tail with small-sample shape shrinkage -- the robust default (M1)."""

    name = "guarded"

    def __init__(self):
        super().__init__(q=0.6, n0=8, xi_cap=0.5)


TAIL_MODELS = {
    "empirical": EmpiricalTail,
    "gaussian": GaussianTail,
    "evt": GPDTail,
    "guarded": GuardedTail,
}


# --------------------------------------------------------------------------- #
# Stopping rules. Each is a stateful object: feed scores one at a time,        #
# should_stop() returns True when the search should terminate.                 #
# --------------------------------------------------------------------------- #
class StoppingRule:
    def reset(self):
        self.scores = []
        self.costs = []
        return self

    def observe(self, score, cost=1.0):
        self.scores.append(float(score))
        self.costs.append(float(cost))

    @property
    def incumbent(self):
        return max(self.scores)

    def should_stop(self):
        raise NotImplementedError


class FixedBudget(StoppingRule):
    """Stop after N evaluations (the standard AutoML default)."""

    def __init__(self, n):
        self.n = n
        self.reset()

    def should_stop(self):
        return len(self.scores) >= self.n

    def __repr__(self):
        return f"fixed(N={self.n})"


class Patience(StoppingRule):
    """Stop if no improvement to the incumbent in the last `p` evaluations."""

    def __init__(self, p, warmup=5):
        self.p = p
        self.warmup = warmup
        self.reset()

    def should_stop(self):
        n = len(self.scores)
        if n < self.warmup:
            return False
        best_idx = int(np.argmax(self.scores))
        return (n - 1 - best_idx) >= self.p

    def __repr__(self):
        return f"patience(p={self.p})"


class MarginalStopping(StoppingRule):
    """Cost-aware optimal stopping: stop when EI(incumbent) <= next cost.

    Parameter-free given the cost. The observation phase is COST-AWARE (M1):
    instead of a fixed floor, we use only a small seed `k_min` to make the tail
    model identifiable, then let the EI<=cost test decide. The effective
    observation length then emerges from the cost -- it shortens automatically
    when evaluations are expensive (high cost) and lengthens when they are cheap.
    `cost` is the per-eval price in *score units*; if None we use the running
    mean of observed (score-unit) costs.
    """

    def __init__(self, tail="evt", k_min=4, cost=None, k_obs=None):
        self.tail_name = tail
        self.k_min = k_obs if k_obs is not None else k_min  # k_obs kept for back-compat
        self.cost = cost
        self.reset()

    def should_stop(self):
        n = len(self.scores)
        if n < self.k_min:
            return False
        tail = TAIL_MODELS[self.tail_name]().fit(self.scores)
        ei = tail.expected_improvement(self.incumbent)
        c_next = self.cost if self.cost is not None else float(np.mean(self.costs))
        return ei <= c_next

    def __repr__(self):
        return f"optimal({self.tail_name},k_min={self.k_min},c={self.cost})"


# --------------------------------------------------------------------------- #
# Simulation: run one ordered trace through a rule.                            #
# --------------------------------------------------------------------------- #
def run_trace(scores, costs, rule):
    """Feed (scores, costs) sequentially until the rule stops.

    Returns dict with: n_evals, total_cost, incumbent, stop_index.
    """
    rule.reset()
    for i, (s, c) in enumerate(zip(scores, costs)):
        rule.observe(s, c)
        if rule.should_stop():
            break
    n = len(rule.scores)
    return {
        "n_evals": n,
        "total_cost": float(np.sum(rule.costs)),
        "incumbent": float(rule.incumbent),
        "stop_index": n - 1,
    }


def reservation_value(samples, cost, tail="evt"):
    """Solve EI(z) = cost for the reservation value z (diagnostic / theory check)."""
    model = TAIL_MODELS[tail]().fit(samples)
    lo, hi = float(np.min(samples)), float(np.max(samples)) + 10 * np.std(samples)
    # EI is decreasing in z. Bracket a root of EI(z) - cost.
    f = lambda z: model.expected_improvement(z) - cost
    if f(lo) <= 0:
        return lo
    if f(hi) >= 0:
        return hi
    return float(stats.brentq(f, lo, hi))
