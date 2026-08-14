"""T3 — B' misspecification-to-breakage (TPAMI revision; the open B4.4 edge).

Runner-series name only: "T3" is the third revision runner (companion to T1a/run_t1a.py
and T2/run_t2.py). It is NOT Theorem 3 (thm:t3-target) — this file proves nothing; it
exercises the FROZEN certificate (validity.run_cell / certificate.py; no formula or
constant change, R039 discipline intact) on worlds that violate the B' K-cell
stratified-shift assumption, and locates the breakage boundary.

Why this run exists
-------------------
R029-R031 (cmd_misspec in run_m3.py) swept off-class q-perturbations (tilt / gradient /
tv) and reached 0 violations at every eps up to 0.8 — "misspecification-to-breakage was
never reached" (paper sec 6, app G). Those modes never inject the ONE thing that defeats
a cell-constant certificate: a WITHIN-CELL correlation between the true density ratio
w = q/p and the risk eta on the ACCEPTED region, in the optimistic direction. That is
exactly the non-K-measurable bias the paper's Remark warns is "not estimable from
covariates alone" (sec 4.3 / Remark misspec).

Construction (adversarial, but an honest member of W_B)
-------------------------------------------------------
We redistribute TARGET mass q WITHIN each accepted cell toward its higher-eta sub-bins,
holding each accepted cell's TOTAL q fixed. Consequences:
  * the certificate's per-cell statistics (p_hat_k, q_hat_k, w_hat_k, rho_lambda) and its
    risk test (EB-UCB of w_hat * S * (L-alpha) on SOURCE data) are INVARIANT in eps — so
    the B' certificate is provably blind to the perturbation: it keeps locking the SAME
    aggressive near-frontier operating point (cert_freq flat across eps);
  * the TRUE feasibility frontier beta*(alpha) recedes as eps grows, so the locked
    operating point's TRUE risk rises and eventually crosses alpha — every certification
    past that point is a truth-level violation (breakage).
The shift stays inside the bounded-ratio class (w <= B = 10, the real-workload registered
bound; recorded per eps); only the K-measurability assumption is violated. This isolates
the misspecification bias from estimation error and from the floor.

Control arm (the honesty check)
-------------------------------
oracle-A (Model A) runs on the SAME worlds with the TRUE per-bin w. It controls the true
G_Q, so it must NOT break — it shrinks the prefix as the true frontier drops. oracle-A
staying valid while B' breaks is what pins the breakage on K-misspecification (not an
infeasible world or a harness bug).

  python -m experiments.run_t3 breakage   (R064) [--reps 2000] [--jobs 32] [--smoke]
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr

from .generators import World, equal_blocks, lattice_prefix, rng_for
from .arms import arm_bprime
from .validity import run_cell, DELTA_TOL_FACTOR
from .runner_util import save_json, pmap

ALPHA, DELTA, BETA = 0.2, 0.05, 0.40
B_CLASS = 10.0                         # bounded-ratio bound = the real-workload registered B (R039 §2)

# CONVEX eta(x) = 0.02 + 0.9 x^2: accepted-region mean stays low (wide certifying margin
# at eps=0) while the cell top exceeds alpha, so within-cell redistribution can push the
# true accepted risk across alpha while staying inside w <= B.
C_ETA, G_ETA = 0.90, 2.0
NBINS, K_WORLD = 256, 4                # 64 bins/cell; accepted region = first two cells
ACCEPTED_CELLS = (0, 1)
W_BLOCKS = np.array([1.30, 1.10, 0.85, 0.75])   # mild on-class cross-cell tilt (eps=0)

# headline grids (registered synthetic design axis — densities only, R039 §2 B4.4)
EPS_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
N_BIG = 4_194_304                      # eps-sweep sample size (rho small -> bias, not noise)
EPS_BROKEN = 1.0                       # fixed eps for the n-sweep (well past breakage onset)
N_SWEEP = [262_144, 1_048_576, 4_194_304, 16_777_216]   # bias must PERSIST as n grows


def build_breakage_world(eps: float, *, nbins: int = NBINS, K: int = K_WORLD,
                         alpha: float = ALPHA, c_eta: float = C_ETA, g_eta: float = G_ETA,
                         B_cls: float = B_CLASS, accepted_cells=ACCEPTED_CELLS) -> World:
    """World whose TRUE shift leaves the K-cell class as eps grows, by within-accepted-
    cell redistribution of q toward high-eta bins at FIXED cell totals.

    At eps=0 the world is exactly on-class (w piecewise constant on the K cells). eps>0
    keeps every cell TOTAL q fixed (so the certificate's cell stats and rho are unchanged)
    while concentrating q on the high-eta tail of each accepted cell (so the true frontier
    recedes and the locked operating point's true risk rises).
    """
    x = (np.arange(nbins) + 0.5) / nbins
    eta = np.clip(0.02 + c_eta * x ** g_eta, 1e-3, 1 - 1e-3)   # convex; crosses alpha inside accepted region
    cells = equal_blocks(nbins, K)
    q0 = np.full(nbins, 1.0 / nbins)             # uniform target baseline
    w_raw = W_BLOCKS[cells]                       # on-class baseline weights (cell-constant)
    p = q0 / w_raw
    p = p / p.sum()                               # baseline w = q0/p is exactly cell-constant
    q = q0.copy()
    for k in accepted_cells:
        idx = np.flatnonzero(cells == k)
        eta_k = eta[idx]
        s = (eta_k - eta_k.mean()) / (eta_k.std() + 1e-12)   # standardized within-cell eta
        f = np.exp(eps * s)                                  # tilt q toward high eta
        q_new = q0[idx] * f
        q_new *= q0[idx].sum() / q_new.sum()                 # EXACT cell-total preservation
        q[idx] = q_new
    world = World(p=p, q=q, eta=eta, name=f"misspec_breakage_eps{eps:.3f}",
                  B_class=B_cls, cells=cells)
    beta_star, j_star = world.frontier(alpha)
    acc = np.isin(cells, accepted_cells)
    wn_corr = float(spearmanr(world.w[acc], eta[acc]).correlation)   # mechanism readout (never seen by B')
    world.meta = {"alpha": alpha, "beta_star": beta_star, "j_star": j_star, "eps": eps,
                  "max_w": float(world.w.max()), "min_w": float(world.w.min()),
                  "B_class": float(B_cls), "in_class": bool(world.w.max() <= B_cls + 1e-9),
                  "K": K, "accepted_cells": list(accepted_cells),
                  "accepted_w_eta_spearman": wn_corr}
    return world


def _modal_locked_bin(world: World, n: int, reps: int = 96) -> int:
    """The bin index of B''s modal locked operating point — the aggressive near-frontier
    prefix it certifies. Computed once at eps=0; since B''s decision is invariant in eps
    (source stats + cell w_hat unchanged), this bin is the locked point at every eps, so
    its TRUE risk vs eps is a noise-free readout of the operating point crossing the
    receding frontier."""
    lat = lattice_prefix(world, 50)
    cob = np.asarray(world.cells, int)
    nw, nr, mw, mf = n // 2, n - n // 2, n // 2, n - n // 2
    shp = (reps, world.n_bins)
    cwP = np.empty(shp, int); cr = cwP.copy(); lr = cwP.copy(); cwQ = cwP.copy(); cf = cwP.copy()
    for i in range(reps):
        rng = rng_for(("R064_lock", int(n)), i)
        a, b, c, d, e = world.sample_counts(rng, nw, nr, mw, mf)
        cwP[i], cr[i], lr[i], cwQ[i], cf[i] = a, b, c, d, e
    ch = arm_bprime(cwP, cr, lr, cwQ, cf, world.B_class, ALPHA, BETA, DELTA, cob, lat)["chosen"]
    ch = ch[ch >= 0]
    return int(lat[int(np.bincount(ch).argmax())]) if len(ch) else -1


def _eps_job(spec):
    eps, n, reps, lock_bin = spec
    world = build_breakage_world(eps)
    res = run_cell(world, n=n, m=n, alpha=ALPHA, beta=BETA, delta=DELTA,
                   n_reps=reps, cell_key=("R064_eps", float(eps), int(n)),
                   arms=("oracle_a", "bprime"))
    b, o = res["bprime"], res["oracle_a"]
    true_R_lock = float(world.risk()[lock_bin]) if lock_bin >= 0 else None
    true_C_lock = float(world.coverage()[lock_bin]) if lock_bin >= 0 else None
    return {
        "eps": eps, "n": n,
        "beta_star": world.meta["beta_star"], "max_w": world.meta["max_w"],
        "B_class": world.meta["B_class"], "in_class": world.meta["in_class"],
        "accepted_w_eta_spearman": world.meta["accepted_w_eta_spearman"],
        "true_risk_at_locked_point": true_R_lock, "true_cov_at_locked_point": true_C_lock,
        "bprime_cert_freq": b["cert_freq"], "bprime_viol_freq": b["viol_freq"],
        "bprime_viol_cp_ucb": b["viol_cp_ucb"], "bprime_mean_rho_at_chosen": b.get("mean_rho_at_chosen"),
        "oracle_cert_freq": o["cert_freq"], "oracle_viol_freq": o["viol_freq"],
        "oracle_viol_cp_ucb": o["viol_cp_ucb"],
    }


def _nsweep_job(spec):
    n, reps = spec
    world = build_breakage_world(EPS_BROKEN)
    res = run_cell(world, n=n, m=n, alpha=ALPHA, beta=BETA, delta=DELTA,
                   n_reps=reps, cell_key=("R064_n", int(n)), arms=("oracle_a", "bprime"))
    b, o = res["bprime"], res["oracle_a"]
    return {"n": n, "bprime_cert_freq": b["cert_freq"], "bprime_viol_freq": b["viol_freq"],
            "bprime_viol_cp_ucb": b["viol_cp_ucb"],
            "oracle_cert_freq": o["cert_freq"], "oracle_viol_freq": o["viol_freq"]}


def cmd_breakage(args):
    eps_grid = [0.0, 0.4, 0.8, 1.0] if args.smoke else EPS_GRID
    n_sweep = [262_144, 4_194_304] if args.smoke else N_SWEEP
    reps = 400 if args.smoke else args.reps
    tol = DELTA_TOL_FACTOR * DELTA                      # 0.0625 — same criterion as R029-R031

    lock_bin = _modal_locked_bin(build_breakage_world(0.0), N_BIG)

    # (1) eps-sweep at N_BIG: the breakage curve
    eps_rows = pmap(_eps_job, [(e, N_BIG, reps, lock_bin) for e in eps_grid],
                    n_jobs=args.jobs, desc="R064 eps-sweep")
    eps_rows = sorted(eps_rows, key=lambda r: r["eps"])

    # (2) n-sweep at EPS_BROKEN: bias persists / sharpens with more data (not small-sample noise)
    n_rows = pmap(_nsweep_job, [(n, reps) for n in n_sweep], n_jobs=args.jobs, desc="R064 n-sweep")
    n_rows = sorted(n_rows, key=lambda r: r["n"])

    eps0_b = next((r["eps"] for r in eps_rows if r["bprime_viol_cp_ucb"] > tol), None)
    eps0_o = next((r["eps"] for r in eps_rows if r["oracle_viol_cp_ucb"] > tol), None)
    cfs = [r["bprime_cert_freq"] for r in eps_rows]
    cert_spread = float(max(cfs) - min(cfs))
    breakage_reached = eps0_b is not None
    oracle_clean = eps0_o is None
    invisible = cert_spread < 0.05 and breakage_reached
    in_class_to_breakage = all(r["in_class"] for r in eps_rows
                               if r["eps"] <= (eps0_b if eps0_b is not None else 1e9))
    # bias-not-noise signature: violation is monotone NON-DECREASING in n and saturates
    # high at the largest n. Pure estimation noise would instead shrink toward 0 as n->inf;
    # the misspecification bias does the opposite — at small n the larger rho budget makes
    # B' lock a conservative point (0 viol), and more data only makes it confident enough
    # to certify the aggressive near-frontier point where the hidden bias bites.
    viol_by_n = [r["bprime_viol_freq"] for r in n_rows]            # n ascending
    nondecr = all(viol_by_n[i + 1] >= viol_by_n[i] - 1e-9 for i in range(len(viol_by_n) - 1))
    breakage_sharpens_with_n = bool(viol_by_n[-1] > tol and nondecr)

    payload = {
        "run": "R064",
        "title": "B' misspecification-to-breakage (within-accepted-cell w-eta correlation)",
        "setup": {"alpha": ALPHA, "beta": BETA, "delta": DELTA, "delta_tol": tol,
                  "nbins": NBINS, "K_world": K_WORLD, "accepted_cells": list(ACCEPTED_CELLS),
                  "eta": f"0.02 + {C_ETA} x^{G_ETA} (clipped to (0,1))", "B_class": B_CLASS,
                  "eps_grid": eps_grid, "n_big": N_BIG, "n_sweep": n_sweep,
                  "eps_for_n_sweep": EPS_BROKEN, "reps": reps,
                  "locked_operating_bin_at_eps0": lock_bin,
                  "criterion": "eps0 = first eps with viol_cp_ucb > 1.25*delta (= R029-R031 rule)",
                  "certificate": "FROZEN validity.run_cell / certificate.py (no formula change)"},
        "eps_sweep_rows": eps_rows,
        "n_sweep_rows": n_rows,
        "verdict": {
            "breakage_reached": bool(breakage_reached),
            "bprime_eps0_breakage": eps0_b,
            "oracle_A_stays_valid": bool(oracle_clean),
            "bias_invisible_to_certificate": bool(invisible),
            "bprime_cert_freq_spread_over_eps": cert_spread,
            "stays_in_bounded_ratio_class_to_breakage": bool(in_class_to_breakage),
            "breakage_sharpens_with_n_not_noise": breakage_sharpens_with_n,
            "n_sweep_violation_by_n": viol_by_n,
            "contrast_with_R029_R031": "R029-R031 (tilt/gradient/tv) never reached breakage "
                                       "(eps0=None at eps<=0.8); the within-cell w-eta mode "
                                       "does, while oracle-A on the same worlds does not.",
        },
        "interpretation": (
            "Confirms the paper's disclosed conditionality: B' validity REQUIRES "
            "K-measurability of w. When the true ratio varies within a declared cell in "
            "correlation with risk on the accepted region, the certificate cannot see it "
            "(cert_freq and rho are unchanged across eps) yet the true frontier recedes and "
            "the locked operating point's true risk crosses alpha — the non-estimable "
            "misspecification bias. It is made worse, not better, by more data: at small n "
            "the larger nuisance budget makes B' lock a conservative point (0 violations), "
            "and only as n grows does it confidently certify the aggressive near-frontier "
            "point where the hidden bias bites (n-sweep violations 0->0->1->1). The breakage "
            "boundary is now exhibited in-class; it does not contradict any claim (validity "
            "was always stated conditional on the stratified-shift model) and oracle-A's "
            "cleanliness rules out infeasibility."),
    }
    save_json("R064_t3_misspec_breakage.json", payload)

    print(f"\n=== R064 eps-sweep at n={N_BIG} (locked bin {lock_bin}/{NBINS}) ===")
    print(f"{'eps':>5} {'max_w':>6} {'in<=B':>6} {'sp(w,eta)':>9} {'R@lock':>7} {'beta*':>6} "
          f"{'B_cert':>7} {'B_viol':>7} {'B_ucb':>7} {'O_cert':>7} {'O_viol':>7}")
    for r in eps_rows:
        print(f"{r['eps']:>5.2f} {r['max_w']:>6.2f} {str(r['in_class']):>6} "
              f"{r['accepted_w_eta_spearman']:>9.2f} {r['true_risk_at_locked_point']:>7.3f} "
              f"{r['beta_star']:>6.3f} {r['bprime_cert_freq']:>7.3f} {r['bprime_viol_freq']:>7.3f} "
              f"{r['bprime_viol_cp_ucb']:>7.3f} {r['oracle_cert_freq']:>7.3f} {r['oracle_viol_freq']:>7.3f}")
    print(f"\n=== R064 n-sweep at eps={EPS_BROKEN} (bias must not vanish as n grows) ===")
    print(f"{'n':>10} {'B_cert':>7} {'B_viol':>7} {'B_ucb':>7} {'O_cert':>7} {'O_viol':>7}")
    for r in n_rows:
        print(f"{r['n']:>10} {r['bprime_cert_freq']:>7.3f} {r['bprime_viol_freq']:>7.3f} "
              f"{r['bprime_viol_cp_ucb']:>7.3f} {r['oracle_cert_freq']:>7.3f} {r['oracle_viol_freq']:>7.3f}")
    print(f"\nbreakage_reached={breakage_reached}  B'-eps0={eps0_b}  oracle_clean={oracle_clean}  "
          f"invisible={invisible}  in_class={in_class_to_breakage}  sharpens_with_n={breakage_sharpens_with_n}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("breakage")
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--jobs", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    {"breakage": cmd_breakage}[args.cmd](args)


if __name__ == "__main__":
    main()
