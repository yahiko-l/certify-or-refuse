"""R068 — SCoRE vs B' certification-count head-to-head on the full 1024-cell grid
(the 4 intensity grids combined; NON-DESTRUCTIVE — reads frozen B1_grid_L*.json for cell
configs, writes a new file). Runs oracle_a + bprime + score_floor at a fixed matched reps
on ALL cells and tallies certification counts under both conventions (any-cert cert>0 and
powered cert_freq>=1/2). bprime/oracle counts here are a matched-reps reproduction of the
frozen headline (583 / 8); the score_floor count is the new datum. Does NOT overwrite the
frozen grids or their escalated 583/8."""
from __future__ import annotations

import argparse
import json
import os

from .generators import family_continuum
from .validity import run_cell
from .runner_util import save_json, pmap, RESULTS_DIR
from .run_m2_b1 import ALPHA, DELTA, N_LATTICE

ARMS = ("oracle_a", "bprime", "score_floor")


def _all_cells():
    cells = []
    for intensity in (1, 2, 3, 4):
        p = os.path.join(RESULTS_DIR, f"B1_grid_L{intensity}.json")
        data = json.load(open(p))
        beta = data["beta"]
        for c in data["cells"]:
            cells.append((intensity, c["_meta"]["n"], c["_meta"]["m"], beta))
    return cells


def _job(spec):
    intensity, n, m, beta, reps = spec
    world = family_continuum(intensity)
    r = run_cell(world, n=int(n), m=int(m), alpha=ALPHA, beta=beta, delta=DELTA,
                 n_reps=reps, cell_key=("B1", intensity, int(n), int(m)),
                 arms=ARMS, n_lattice=N_LATTICE, rep_offset=0)
    return {a: {"cert": r[a]["cert"], "cert_freq": r[a]["cert_freq"],
                "viol": r[a]["viol"]} for a in ARMS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=48)
    args = ap.parse_args()
    cells = _all_cells()
    specs = [(i, n, m, beta, args.reps) for (i, n, m, beta) in cells]
    rows = pmap(_job, specs, n_jobs=args.jobs,
                desc=f"R068 full-grid count ({len(specs)} cells x {args.reps})")
    cnt_any = {a: 0 for a in ARMS}          # cert > 0
    cnt_pow = {a: 0 for a in ARMS}          # cert_freq >= 1/2
    viol_tot = {a: 0 for a in ARMS}
    score_ge_bp = score_gt_bp = 0
    for r in rows:
        for a in ARMS:
            cnt_any[a] += int(r[a]["cert"] > 0)
            cnt_pow[a] += int(r[a]["cert_freq"] >= 0.5)
            viol_tot[a] += r[a]["viol"]
        score_ge_bp += int(r["score_floor"]["cert_freq"] >= r["bprime"]["cert_freq"] - 1e-12)
        score_gt_bp += int(r["score_floor"]["cert_freq"] > r["bprime"]["cert_freq"] + 1e-12)
    payload = {"run": "R068_fullgrid", "reps_per_cell": args.reps, "n_cells": len(cells),
               "arms": list(ARMS), "count_any_cert": cnt_any, "count_powered": cnt_pow,
               "total_violations": viol_tot,
               "score_ge_bprime_cells": score_ge_bp, "score_gt_bprime_cells": score_gt_bp,
               "note": ("matched-reps reproduction; frozen headline 583/8 used 1000+"
                        "escalation. Comparison is SCoRE vs B' at identical reps/cells.")}
    save_json("R068_fullgrid_count.json", payload)
    print("=== R068 full-grid certification-count head-to-head ===")
    print(f"cells={len(cells)} reps={args.reps}")
    print(f"  count (cert>0):       {cnt_any}")
    print(f"  count (powered>=1/2): {cnt_pow}")
    print(f"  total violations:     {viol_tot}")
    print(f"  score_floor cert_freq >= bprime in {score_ge_bp}/{len(cells)} cells; "
          f"> in {score_gt_bp}/{len(cells)}")


if __name__ == "__main__":
    main()
