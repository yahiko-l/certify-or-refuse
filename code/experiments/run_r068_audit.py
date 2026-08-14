"""R068 — floor-augmented SCoRE head-to-head audit (NON-DESTRUCTIVE).

Reuses the EXACT R042 audit-cell subset (B'-certified cells: ALL at headline intensity
L2 + stratified 10% elsewhere, read from the frozen B1_grid_L*.json) and the SAME rep
streams (cell_key, rep_offset=1e6) as cmd_audit, then runs oracle_a + bprime +
score_floor at >=4000 reps. Reports the B'-vs-SCoRE head-to-head (per-cell cert_freq +
pooled Clopper-Pearson violation bounds). Does NOT touch the frozen R042 / B1_grid_L*.json
/ ARMS_GRID; writes a new R068_score_floor_audit.json. Reproducing bprime/oracle here is
also a determinism check against R042 (bprime/oracle pooled viol must be 0).
"""
from __future__ import annotations

import argparse
import json
import os

from .generators import family_continuum
from .validity import run_cell, cp_ucb
from .runner_util import save_json, pmap, RESULTS_DIR
from .run_m2_b1 import ALPHA, DELTA, HEADLINE_INTENSITY, N_LATTICE

ARMS = ("oracle_a", "bprime", "score_floor")


def _audit_cells():
    """Replicate cmd_audit's cell selection EXACTLY from the frozen grids."""
    cells = []
    for intensity in (1, 2, 3, 4):
        p = os.path.join(RESULTS_DIR, f"B1_grid_L{intensity}.json")
        if not os.path.exists(p):
            raise SystemExit(f"missing frozen grid {p} (run R042 prerequisite grids first)")
        data = json.load(open(p))
        certified = [c for c in data["cells"] if c["bprime"]["cert"] > 0]
        if intensity == HEADLINE_INTENSITY:
            chosen = certified
        else:
            stride = max(len(certified) // max(len(certified) // 10, 1), 1)
            chosen = certified[::stride][: max(len(certified) // 10, 1)]
        for c in chosen:
            cells.append((intensity, c["_meta"]["n"], c["_meta"]["m"], data["beta"]))
    return cells


def _job(spec):
    intensity, n, m, beta, reps = spec
    world = family_continuum(intensity)
    return run_cell(world, n=int(n), m=int(m), alpha=ALPHA, beta=beta, delta=DELTA,
                    n_reps=reps, cell_key=("B1", intensity, int(n), int(m)),
                    arms=ARMS, n_lattice=N_LATTICE, rep_offset=10 ** 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--jobs", type=int, default=48)
    args = ap.parse_args()
    cells = _audit_cells()
    specs = [(i, n, m, beta, args.reps) for (i, n, m, beta) in cells]
    rows = pmap(_job, specs, n_jobs=args.jobs,
                desc=f"R068 SCoRE audit ({len(specs)} cells x {args.reps})")

    recs = []
    pooled = {a: [0, 0] for a in ARMS}          # arm -> [viol, n_reps]
    per_cell = []
    conf = {"both": 0, "only_bprime": 0, "only_score": 0, "neither": 0}
    score_ge_bprime = score_gt_bprime = 0
    for (i, n, m, beta), r in zip(cells, rows):
        for a in ARMS:
            recs.append({"intensity": i, "n": n, "m": m, "arm": a,
                         "cert_freq": r[a]["cert_freq"], "cert": r[a]["cert"],
                         "viol": r[a]["viol"], "reps": r[a]["n_reps"],
                         "viol_cp_ucb": r[a]["viol_cp_ucb"]})
            pooled[a][0] += r[a]["viol"]
            pooled[a][1] += r[a]["n_reps"]
        bp, sc = r["bprime"]["cert_freq"], r["score_floor"]["cert_freq"]
        per_cell.append({"intensity": i, "n": n, "m": m, "beta": beta,
                         "oracle_cert": r["oracle_a"]["cert_freq"],
                         "bprime_cert": bp, "score_cert": sc,
                         "bprime_viol_cp": r["bprime"]["viol_cp_ucb"],
                         "score_viol_cp": r["score_floor"]["viol_cp_ucb"]})
        # "powered" convention: cert_freq >= 1/2
        bpc, scc = bp >= 0.5, sc >= 0.5
        conf["both" if (bpc and scc) else "only_bprime" if bpc else
             "only_score" if scc else "neither"] += 1
        score_ge_bprime += int(sc >= bp - 1e-12)
        score_gt_bprime += int(sc > bp + 1e-12)

    summary = {a: {"pooled_viol": pooled[a][0], "pooled_n": pooled[a][1],
                   "pooled_cp_ucb": cp_ucb(pooled[a][0], pooled[a][1]),
                   "pooled_pass_delta": bool(cp_ucb(pooled[a][0], pooled[a][1]) <= DELTA)}
               for a in ARMS}
    payload = {
        "run": "R068", "reps_per_cell": args.reps, "n_audit_cells": len(cells),
        "arms": list(ARMS), "alpha": ALPHA, "delta": DELTA,
        "per_arm_pooled": summary,
        "head_to_head": {
            "confusion_powered_geq_half": conf,
            "cells_score_ge_bprime": score_ge_bprime,
            "cells_score_gt_bprime": score_gt_bprime,
            "n_cells": len(cells)},
        "per_cell": per_cell, "records": recs,
        "determinism_check_vs_R042": {
            "bprime_pooled_viol_is_zero": bool(pooled["bprime"][0] == 0),
            "oracle_pooled_viol_is_zero": bool(pooled["oracle_a"][0] == 0)},
        "score_floor_valid_on_grid": bool(summary["score_floor"]["pooled_pass_delta"]),
    }
    save_json("R068_score_floor_audit.json", payload)
    print("=== R068 SCoRE head-to-head audit ===")
    print(f"cells={len(cells)} reps={args.reps}")
    for a in ARMS:
        s = summary[a]
        print(f"  {a:13s} pooled_viol={s['pooled_viol']}/{s['pooled_n']} "
              f"cp_ucb={s['pooled_cp_ucb']:.2e} pass_delta={s['pooled_pass_delta']}")
    print(f"  confusion(powered>=1/2): {conf}")
    print(f"  score>=bprime in {score_ge_bprime}/{len(cells)} cells; "
          f"score>bprime in {score_gt_bprime}/{len(cells)}")
    print(f"  determinism (bprime/oracle viol==0): "
          f"{payload['determinism_check_vs_R042']}")


if __name__ == "__main__":
    main()
