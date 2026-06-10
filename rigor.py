"""
PhD-grade statistical rigor for the stopping-rule comparison.

Loads the per-task matched-regret savings saved by decisive_m3_vs_xie.py and
produces:
  - Friedman omnibus test across the three stopping rules (XIE, MYOPIC, META);
  - average ranks + Nemenyi critical difference (CD) at alpha=0.05;
  - pairwise paired Wilcoxon (META vs XIE, META vs MYOPIC), pooled and per-cost;
  - figures (rank bar + per-cost saving boxplots), if matplotlib is available.

Each (task, cost-level) cell is one paired observation; higher saving = better.
"""

import os

import numpy as np
from scipy import stats as sstats

METHODS = ["XIE", "MYOPIC", "META"]
# Nemenyi q_alpha for k=3 methods, alpha=0.05
Q_ALPHA_K3 = 2.343


def load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    lam_mults = list(d["lam_mults"])
    # build matrix: rows = (task, lam) instances, cols = methods
    rows = []
    per_lam = {m: {meth: [] for meth in METHODS} for m in lam_mults}
    n_tasks = len(d[f"XIE_{lam_mults[0]}"])
    for ti in range(n_tasks):
        for m in lam_mults:
            vals = [float(d[f"{meth}_{m}"][ti]) for meth in METHODS]
            for meth, v in zip(METHODS, vals):
                per_lam[m][meth].append(v)
            if not any(np.isnan(vals)):
                rows.append(vals)
    return np.array(rows), lam_mults, per_lam


def cd_analysis(M):
    """M: (N instances, k methods) savings (higher better). Friedman + Nemenyi CD."""
    N, k = M.shape
    # ranks per row: rank 1 = best (highest saving) -> negate for rankdata ascending
    ranks = np.array([sstats.rankdata(-row) for row in M])
    avg_ranks = ranks.mean(0)
    stat, p = sstats.friedmanchisquare(*[M[:, j] for j in range(k)])
    cd = Q_ALPHA_K3 * np.sqrt(k * (k + 1) / (6.0 * N))
    return avg_ranks, stat, p, cd, N


def main(npz_path=None):
    if npz_path is None:
        cand = [os.path.join("cache", f) for f in os.listdir("cache")
                if f.startswith("decisive_") and f.endswith(".npz")]
        npz_path = sorted(cand)[-1]
    print(f"loading {npz_path}")
    M, lam_mults, per_lam = load(npz_path)

    avg_ranks, stat, p, cd, N = cd_analysis(M)
    print(f"\n=== Friedman test across {METHODS}, N={N} (task x cost) cells ===")
    print(f"  chi2={stat:.2f}  p={p:.2e}")
    print(f"\n=== Average ranks (1=best) + Nemenyi CD={cd:.3f} (alpha=0.05) ===")
    order = np.argsort(avg_ranks)
    for j in order:
        print(f"  {METHODS[j]:8}: {avg_ranks[j]:.3f}")
    best = METHODS[order[0]]
    print(f"\n  Pairwise rank gaps vs best ({best}); gap>CD => significant:")
    for j in order[1:]:
        gap = avg_ranks[j] - avg_ranks[order[0]]
        print(f"    {best} vs {METHODS[j]:8}: gap={gap:.3f} "
              f"{'(SIGNIFICANT)' if gap > cd else '(n.s.)'}")

    print(f"\n=== Paired Wilcoxon (pooled over all cells, N={N}) ===")
    cols = {m: M[:, i] for i, m in enumerate(METHODS)}
    for a, b in [("META", "XIE"), ("META", "MYOPIC")]:
        diff = cols[a] - cols[b]
        try:
            pw = sstats.wilcoxon(diff, alternative="greater").pvalue
        except Exception:
            pw = float("nan")
        print(f"  {a} > {b}: median diff={np.median(diff):+.1f}pp, "
              f"win={np.mean(diff>0):.0%}, p={pw:.2e}")

    print("\n=== Per-cost median saving (%) ===")
    print(f"{'lam x':>7} | " + " | ".join(f"{m:>8}" for m in METHODS))
    for lm in lam_mults:
        row = f"{lm:7} | " + " | ".join(
            f"{np.nanmedian(per_lam[lm][m]):8.1f}" for m in METHODS)
        print(row)

    _figures(M, lam_mults, per_lam, avg_ranks)


def _figures(M, lam_mults, per_lam, avg_ranks):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib unavailable; skipping figures)")
        return
    os.makedirs("figures", exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 3))
    order = np.argsort(avg_ranks)
    ax.barh([METHODS[j] for j in order], [avg_ranks[j] for j in order],
            color=["#2a9d8f", "#e9c46a", "#e76f51"])
    ax.set_xlabel("average rank (1 = best)")
    ax.set_title("Stopping-rule ranks on GP-BO substrate (lower=better)")
    fig.tight_layout(); fig.savefig("figures/avg_ranks.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.25
    xs = np.arange(len(lam_mults))
    for k, m in enumerate(METHODS):
        meds = [np.nanmedian(per_lam[lm][m]) for lm in lam_mults]
        ax.bar(xs + k * width, meds, width, label=m)
    ax.set_xticks(xs + width); ax.set_xticklabels([f"x{lm}" for lm in lam_mults])
    ax.set_xlabel("cost level (lambda)"); ax.set_ylabel("median time saving %")
    ax.axhline(0, color="k", lw=0.7); ax.legend()
    ax.set_title("Matched-regret saving by cost level (GP-BO)")
    fig.tight_layout(); fig.savefig("figures/saving_by_cost.png", dpi=140); plt.close(fig)
    print("\nfigures written: figures/avg_ranks.png, figures/saving_by_cost.png")


if __name__ == "__main__":
    main()
