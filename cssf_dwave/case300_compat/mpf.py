from __future__ import annotations
import numpy as np

def compute_mpf(n: int, edges, slack_buses) -> np.ndarray:
    """Compute a static electrical-distance prior from diag(B_r^{-1})."""
    n = int(n); B = np.zeros((n, n), dtype=float)
    for i, j, susceptance in edges:
        i = int(i); j = int(j); w = abs(float(susceptance))
        if not np.isfinite(w) or w <= 0:
            continue
        B[i, i] += w; B[j, j] += w; B[i, j] -= w; B[j, i] -= w
    slack = set(int(x) for x in slack_buses)
    keep = [i for i in range(n) if i not in slack]
    Br = B[np.ix_(keep, keep)]
    inv = np.linalg.pinv(Br, rcond=1e-12, hermitian=True)
    result = np.zeros(n, dtype=float)
    result[keep] = np.maximum(np.diag(inv), 0.0)
    return result
