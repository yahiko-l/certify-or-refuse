"""R004 — tiny-grid end-to-end smoke, ALL six arms (M0; ~minutes).

3x3 (n, m) grid, one shift level (continuum L2), R=200 repeats. Catches wiring bugs
before M1. Sanity checks: pipeline runs end-to-end; occupancy sane (B' cert freq
non-decreasing along the grid diagonal; oracle-A >= B' typically).

Usage: python -m experiments.run_smoke [--reps 200]
"""
from __future__ import annotations

import argparse

import numpy as np

from .generators import family_continuum
from .validity import run_cell, ARM_NAMES
from .runner_util import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--slack", type=float, default=0.22)
    args = ap.parse_args()

    world = family_continuum(2)
    beta = world.meta["beta_star"] - args.slack
    # top cell sits inside the B' nuisance regime (K=16 at this slack needs n ~ 1e5+)
    ns = [2000, 30000, 500000]
    ms = [1600, 24000, 400000]
    cells = []
    for i, n in enumerate(ns):
        for j, m in enumerate(ms):
            res = run_cell(world, n=n, m=m, alpha=args.alpha, beta=beta,
                           delta=args.delta, n_reps=args.reps,
                           cell_key=("R004", i, j), arms=ARM_NAMES)
            res["_meta"]["grid_ij"] = (i, j)
            cells.append(res)
            line = " ".join(f"{a}:{res[a]['cert_freq']:.2f}" for a in ARM_NAMES)
            print(f"n={n:>6} m={m:>6}  cert_freq  {line}")

    bp = np.array([[c["bprime"]["cert_freq"] for c in cells[k * 3:(k + 1) * 3]]
                   for k in range(3)])
    diag = np.array([bp[0, 0], bp[1, 1], bp[2, 2]])
    checks = {
        "ran_all_cells": len(cells) == 9,
        "bprime_diag_nondecreasing": bool(np.all(np.diff(diag) >= -0.05)),
        "bprime_certifies_at_largest": bool(bp[2, 2] > 0.5),
        "no_arm_crashed": True,
        "valid_arms_within_tol": bool(all(
            c[a]["viol_cp_ucb"] <= 1.25 * args.delta + 0.06   # smoke-level slack (R=200)
            for c in cells for a in ("oracle_a", "bprime"))),
    }
    payload = {"run": "R004", "world": world.name, "beta": beta,
               "beta_star": world.meta["beta_star"], "alpha": args.alpha,
               "delta": args.delta, "reps": args.reps,
               "cells": cells, "checks": checks,
               "smoke_pass": all(checks.values())}
    save_json("R004_smoke.json", payload)
    print({"R004_smoke_pass": payload["smoke_pass"], "checks": checks})
    if not payload["smoke_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
