"""T1b-i — post-hoc real-workload frontier MAP β̂*(α), leg-split (TPAMI revision).

Pure POST-HOC diagnosis on the FROZEN eval artifacts (`_eval_target_frozen.npz`: scores s,
loss L = 1 − [token-F1 ≥ 0.5]). Reads the frozen arrays directly — never re-runs a model and
never opens the eval-label guard: the single registered access (`R033_final_eval`) is already
spent and the frozen .npz is the access-once barrier (data.py:load_eval_golds_guarded). This
is exactly the paper's "post-hoc eval-set frontier diagnosis", just swept over a full α grid
instead of the single α=.10 point — turning §6's single refusal verdict into a frontier-vs-
floor MAP.

β̂*(α) = max over score thresholds t of realized coverage P(score ≥ t) subject to realized
selective risk E[L | score ≥ t] ≤ α (the best achievable selective operating point on the
eval set; an in-sample diagnostic, NOT a certificate — the B′ certificate still refuses
pre-deployment, consuming no eval labels).

Parity: reproduces the paper β̂*(.10) = .050 (leg 1 Llama-3-8B) / .209 (leg 2 DeepSeek-V4-
Flash). Both legs sit far below the floor β = .60 at every defensible α — the honest
frontier-infeasible map.

Usage: python -m experiments.real.frontier_sweep
"""
from __future__ import annotations

import os

import numpy as np

from ..runner_util import save_json

HERE = os.path.dirname(__file__)
LEGS = {1: ("cache_leg1_llama3_8b", "Llama-3-8B"),
        2: ("cache", "DeepSeek-V4-Flash")}
ALPHA_HEAD = [0.05, 0.10, 0.15, 0.20]                 # headline grid (with bootstrap CIs)
ALPHA_DENSE = [round(float(a), 3) for a in np.arange(0.03, 0.2501, 0.01)]
BETA_FLOOR = 0.60                                     # the pre-registered SLA floor
N_BOOT = 500                                          # eval-item bootstrap (paper convention)
SEED = 20260611


def _load(leg):
    d = os.path.join(HERE, LEGS[leg][0])
    z = np.load(os.path.join(d, "_eval_target_frozen.npz"), allow_pickle=False)
    return z["s"].astype(float), z["L"].astype(float)


def betastar(s, L, a, min_acc=1):
    """Best achievable coverage at selective risk ≤ a over all score thresholds."""
    order = np.argsort(-s)                            # descending score = growing acceptance
    Ls = L[order]
    n = len(s)
    k = np.arange(1, n + 1)
    risk = np.cumsum(Ls) / k
    cov = k / n
    ok = (risk <= a) & (k >= min_acc)
    return float(cov[ok].max()) if ok.any() else 0.0


def boot_ci(s, L, a, seed):
    """Percentile CI from N_BOOT paired eval-item resamples (re-optimizing β̂* each time)."""
    rng = np.random.default_rng(seed)
    n = len(s)
    bs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        bs.append(betastar(s[idx], L[idx], a))
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def run_leg(leg):
    s, L = _load(leg)
    head = []
    for a in ALPHA_HEAD:
        b = betastar(s, L, a)
        ci = boot_ci(s, L, a, SEED + int(a * 1000))
        head.append({"alpha": a, "beta_star": round(b, 4), "ci95": [round(c, 4) for c in ci],
                     "crosses_floor": bool(b >= BETA_FLOOR),
                     "floor_in_ci": bool(ci[0] <= BETA_FLOOR <= ci[1])})
    dense = [{"alpha": a, "beta_star": round(betastar(s, L, a), 4)} for a in ALPHA_DENSE]
    out = {"run": f"T1bi_frontier_leg{leg}", "leg": leg, "model": LEGS[leg][1],
           "N_eval": len(s), "mean_loss": round(float(L.mean()), 4), "beta_floor": BETA_FLOOR,
           "n_boot": N_BOOT, "headline": head, "dense_curve": dense,
           "note": "post-hoc eval-set frontier diagnosis (best achievable selective point); "
                   "leg-split, NOT a certificate; frozen-artifact read, no new label access."}
    save_json(f"T1bi_frontier_leg{leg}.json", out)
    return out


def main():
    outs = {leg: run_leg(leg) for leg in LEGS}
    print(f"\n=== T1b-i frontier map vs floor β={BETA_FLOOR} (β̂*(α), 95% CI) ===")
    print(f"{'alpha':>6} | {'leg1 Llama-3-8B':>26} | {'leg2 DeepSeek-V4-Flash':>26}")
    for i, a in enumerate(ALPHA_HEAD):
        h1, h2 = outs[1]["headline"][i], outs[2]["headline"][i]
        print(f"{a:>6.2f} | {h1['beta_star']:>8.4f} [{h1['ci95'][0]:.3f},{h1['ci95'][1]:.3f}]"
              f"{'':>5} | {h2['beta_star']:>8.4f} [{h2['ci95'][0]:.3f},{h2['ci95'][1]:.3f}]")
    cross = any(h["crosses_floor"] or h["floor_in_ci"]
                for leg in LEGS for h in outs[leg]["headline"])
    print(f"\nfloor β={BETA_FLOOR} crossed or within CI at any tested α≤.20? {cross}")
    print("paper parity: leg1 β̂*(.10)=.050 [.035,.130]; leg2 β̂*(.10)=.209 [.160,.281]")


if __name__ == "__main__":
    main()
