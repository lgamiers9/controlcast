def decompose(y_true, p, n_bins=20):
    """Brier = REL - RES + UNC"""
    y_true = np.asarray(y_true, float); p = np.asarray(p, float)
    rr, n = y_true.mean(), len(y_true)
    edges = np.quantile(p, np.linspace(0, 1, n_bins+1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    idx = np.digitize(p, edges[1:-1])
    rel = res = 0.0
    for k in np.unique(idx):
        m = idx == k
        w = m.sum() / n
        rel += w * (p[m].mean() - y_true[m].mean())**2
        res += w * (y_true[m].mean() - rr)**2
    return {"BSS": 100000*(res-rel)/(rr*(1-rr)), "REL": rel, "RES": res,
            "mean_pred - r": p.mean() - rr}