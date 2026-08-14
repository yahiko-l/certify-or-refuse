#!/usr/bin/env python
"""Reviewer-requested robustness diagnostic for the Figure 2 bite slope (Minor 6,
2026-06-24 review): pairs-bootstrap 95% CI on the log-log OLS slope, plus a
residual check, computed from the registered R014 raw points. Deterministic
(seed 20260624). Reproduces the reported OLS slope -2.0021 and writes
R014_bite_bootstrap_ci.json. Does not alter any theorem or the registered fit.
"""
import json, numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "R014_bite_divergence.json"
OUT = HERE / "R014_bite_bootstrap_ci.json"
SEED, B = 20260624, 20000

d = json.load(open(SRC))
sl = np.array(d["slacks"], float)
rn = np.array([d["required_n"][str(s)] for s in sl], float)
x, y = np.log10(sl), np.log10(rn)
A = np.vstack([x, np.ones_like(x)]).T
beta, *_ = np.linalg.lstsq(A, y, rcond=None)
resid = y - A @ beta

rng = np.random.default_rng(SEED)
slopes = []
for _ in range(B):
    idx = rng.integers(0, len(sl), len(sl))
    if np.ptp(x[idx]) == 0:
        continue
    Ab = np.vstack([x[idx], np.ones(len(idx))]).T
    bb, *_ = np.linalg.lstsq(Ab, y[idx], rcond=None)
    slopes.append(bb[0])
slopes = np.array(slopes)
lo, hi = np.percentile(slopes, [2.5, 97.5])

out = {
    "source": SRC.name,
    "n_points": int(len(sl)),
    "ols_slope": float(beta[0]),
    "ols_se_normal_ci95": [float(c) for c in d["fit"]["ci95"]],
    "pairs_bootstrap": {"seed": SEED, "n_resamples_valid": int(len(slopes)),
                        "median": float(np.median(slopes)), "ci95": [float(lo), float(hi)]},
    "log10_residual_absmax": float(np.max(np.abs(resid))),
    "curvature_ci95": [float(c) for c in d["fit"]["curvature_ci"]],
    "note": "required_n at slacks 0.02 and 0.017 coincide (search-cap plateau).",
}
json.dump(out, open(OUT, "w"), indent=2)
print(json.dumps(out, indent=2))
