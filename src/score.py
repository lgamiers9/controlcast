import numpy as np


def brier_skill_score(y_true, p):
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    r = y_true.mean()
    bs = np.mean((p - y_true) ** 2)
    baseline_bs = r * (1 - r)
    return max(0.0, 100000 * (1 - bs / baseline_bs))


def decompose(y_true, p, n_bins=20):
    """Brier = REL - RES + UNC 분해 (보정 문제 vs 분별력 문제 진단용)."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    r, n = y_true.mean(), len(y_true)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    idx = np.digitize(p, edges[1:-1])
    rel = res = 0.0
    for k in np.unique(idx):
        m = idx == k
        w = m.sum() / n
        rel += w * (p[m].mean() - y_true[m].mean()) ** 2
        res += w * (y_true[m].mean() - r) ** 2
    return {
        "BSS": max(0.0, 100000 * (res - rel) / (r * (1 - r))),
        "REL": rel,
        "RES": res,
        "mean_pred - r": p.mean() - r,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000)
    p = np.clip(y * 0.6 + rng.normal(0.4, 0.1, size=1000), 0, 1)
    print(brier_skill_score(y, p))
    print(decompose(y, p))
