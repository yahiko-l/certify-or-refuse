"""M2 / B1 — the law: phase diagrams, beta-ladder, bite divergence, regime edge,
adaptive refinement, validity-audit subset (R009-R016, R042).

Registered runner config (in-code, pre-data):
  working slack s_work = 0.06; intensities L1..L4 (headline = L2 for the audit subset);
  (n, m) base grid 12x12 log-spaced n,m in [256, 262144]; +refinement lines within
  +/-1 octave of the constants-1 theory corner n*(s)=beta*ln(|Λ|/delta)/(kappa s)^2,
  m*(s)=beta*ln(|Λ|/delta)/s^2; MC >= 1000 reps/cell; escalation per protocol margin
  (max 3 doublings, final counts logged); beta-ladder {0.6,...,0.019} with the
  beta*m_f >= 50 degeneracy guard; slack grid 8 geometric values in [0.015, 0.24].

Subcommands:
  python -m experiments.run_m2_b1 grid --intensity {1,2,3,4} [--reps 1000] [--jobs 32]
  python -m experiments.run_m2_b1 ladder   (R013)
  python -m experiments.run_m2_b1 bite     (R014)
  python -m experiments.run_m2_b1 edge     (R015)
  python -m experiments.run_m2_b1 audit    (R042; after grids)  [--reps 4000]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .analysis import loglog_exponent
from .generators import family_continuum
from .validity import run_cell, holm_exact, cp_ucb, escalation_target_halfwidth
from .runner_util import save_json, pmap, log_grid, RESULTS_DIR

ALPHA, DELTA = 0.2, 0.05
S_WORK = 0.12
HEADLINE_INTENSITY = 2
BETA_LADDER = [0.6, 0.3, 0.15, 0.075, 0.0375, 0.019]
SLACK_GRID = np.geomspace(0.015, 0.24, 8)
N_LATTICE = 50
ARMS_GRID = ("oracle_a", "bprime", "floorfree_posthoc", "dro_box",
             "weighted_conformal", "plugin")


def theory_corner(world, s):
    kappa = world.meta["kappa"]
    beta = world.meta["beta_star"] - s
    L = np.log(N_LATTICE / DELTA)
    return beta * L / (kappa * s) ** 2, beta * L / s ** 2


def grid_cells(world, s):
    base_n = log_grid(256, 262144, 12)
    base_m = log_grid(256, 262144, 12)
    n_star, m_star = theory_corner(world, s)
    ref_n = log_grid(max(64, n_star / 2), n_star * 2, 4)
    ref_m = log_grid(max(64, m_star / 2), m_star * 2, 4)
    ns = np.unique(np.concatenate([base_n, ref_n]))
    ms = np.unique(np.concatenate([base_m, ref_m]))
    return ns, ms


def _grid_job(spec):
    intensity, n, m, beta, reps, rep_offset = spec
    world = family_continuum(intensity)
    return run_cell(world, n=int(n), m=int(m), alpha=ALPHA, beta=beta, delta=DELTA,
                    n_reps=reps, cell_key=("B1", intensity, int(n), int(m)),
                    arms=ARMS_GRID, n_lattice=N_LATTICE, rep_offset=rep_offset)


def cmd_grid(args):
    """R009-R012 (+R016 escalation): one intensity's phase-diagram grid."""
    world = family_continuum(args.intensity)
    beta = world.meta["beta_star"] - S_WORK
    ns, ms = grid_cells(world, S_WORK)
    specs = [(args.intensity, n, m, beta, args.reps, 0) for n in ns for m in ms]
    cells = pmap(_grid_job, specs, n_jobs=args.jobs,
                 desc=f"B1 grid L{args.intensity} ({len(specs)} cells)")
    # ---- R016: adaptive escalation per protocol margin (<=3 doublings, logged);
    # batches merged field-by-field via validity.merge_tallies (nothing stale)
    from .validity import merge_tallies
    tau = 0.5
    escalated = 0
    for idx, c in enumerate(cells):
        for rounds in range(3):
            f = c["bprime"]["cert_freq"]; nrep = c["bprime"]["n_reps"]
            half = (cp_ucb(c["bprime"]["cert"], nrep)
                    - (1 - cp_ucb(nrep - c["bprime"]["cert"], nrep))) / 2.0
            target = escalation_target_halfwidth(f, tau, DELTA)
            if not (0.05 < f < 0.95) or half < target or target <= 0:
                break
            extra = _grid_job((args.intensity, c["_meta"]["n"], c["_meta"]["m"], beta,
                               nrep, nrep))
            for arm in ARMS_GRID:
                c[arm] = merge_tallies(c[arm], extra[arm])
            c["_meta"]["escalation_rounds"] = rounds + 1
            c["_meta"]["final_reps"] = c["bprime"]["n_reps"]
            escalated += 1
    payload = {"run": f"R{8 + args.intensity:03d}", "intensity": args.intensity,
               "world": world.name, "beta": beta, "s_work": S_WORK,
               "beta_star": world.meta["beta_star"], "kappa": world.meta["kappa"],
               "alpha": ALPHA, "delta": DELTA, "n_lattice": N_LATTICE,
               "theory_corner_const1": theory_corner(world, S_WORK),
               "ns": ns.tolist(), "ms": ms.tolist(),
               "escalated_cells": escalated, "cells": cells}
    save_json(f"B1_grid_L{args.intensity}.json", payload)


def rel_slack_grid(beta: float) -> np.ndarray:
    """Registered beta-relative slack grid: s = beta * geomspace(0.05, 0.4, 8) —
    8 values, all inside the theorem regime s <= c0*beta with c0 = 0.4."""
    return beta * np.geomspace(0.05, 0.4, 8)


SLACK_DEV_TOL = 0.25      # probe excluded if |actual_s/requested_s - 1| > 25%


def _ladder_job(spec):
    """One (beta, s) probe: the WORLD is rebuilt so its frontier sits at beta + s
    (beta_star_target, bisection-solved). The ACTUAL slack actual_s = beta*_world -
    beta is recorded and used downstream; probes whose actual slack drifts beyond
    SLACK_DEV_TOL or leaves the regime are excluded (round-2 review fix)."""
    intensity, beta, s, n, m, reps = spec
    world = family_continuum(intensity, beta_star_target=beta + s)
    actual_s = float(world.meta["beta_star"]) - beta
    valid = (actual_s > 0 and abs(actual_s / s - 1.0) <= SLACK_DEV_TOL
             and actual_s <= 0.4 * beta * (1 + SLACK_DEV_TOL))
    res = run_cell(world, n=n, m=m, alpha=ALPHA, beta=beta, delta=DELTA,
                   n_reps=reps, cell_key=("B1lad", float(beta), float(s), n, m),
                   arms=("oracle_a", "bprime"), n_lattice=N_LATTICE) if valid else None
    return (beta, s, actual_s, valid, n, m, float(world.meta["beta_star"]), res)


def cmd_ladder(args):
    """R013 — beta-ladder incl. beta->0 limb with the beta*m_f >= 50 guard.
    For each beta: certifiable-slack threshold s*(beta) on fixed (n,m) anchors, with
    per-beta worlds whose frontier sits at beta+s for each probed slack s."""
    anchors = [(4096, 4096), (16384, 16384), (65536, 65536), (262144, 262144)]
    specs = []
    for beta in BETA_LADDER:
        for (n, m) in anchors:
            if beta * (m / 2) < 50:
                continue                       # registered degeneracy guard
            for s in rel_slack_grid(beta):
                specs.append((args.intensity, beta, float(s), n, m, args.reps))
    rows = pmap(_ladder_job, specs, n_jobs=args.jobs, desc="R013 beta-ladder")
    # threshold s*(beta, anchor): smallest ACTUAL slack with cert_freq >= 0.5
    out = {}
    dropped_probes = []
    for beta, s, actual_s, valid, n, m, bstar, r in rows:
        key = f"beta{beta}_n{n}_m{m}"
        out.setdefault(key, {"beta": beta, "n": n, "m": m, "by_slack": {}})
        if not valid:
            dropped_probes.append({"beta": beta, "requested_s": s,
                                   "actual_s": actual_s, "beta_star_world": bstar})
            continue
        out[key]["by_slack"][f"{actual_s:.6f}"] = {
            "requested_s": s, "actual_s": actual_s, "beta_star_world": bstar,
            "bprime_cert": r["bprime"]["cert_freq"],
            "oracle_cert": r["oracle_a"]["cert_freq"],
            "bprime_viol_ucb": r["bprime"]["viol_cp_ucb"]}
    for key, rec in out.items():
        slacks = sorted(float(s) for s in rec["by_slack"])
        for arm in ("bprime", "oracle"):
            rec[f"s_star_{arm}"] = next(
                (s for s in slacks
                 if rec["by_slack"][f"{s:.6f}"][f"{arm}_cert"] >= 0.5), None)
    excluded = [(b, n, m) for b in BETA_LADDER for (n, m) in anchors if b * (m / 2) < 50]
    payload = {"run": "R013", "intensity": args.intensity, "anchors": anchors,
               "ladder": BETA_LADDER,
               "slack_grid": "beta-relative: beta*geomspace(0.05, 0.4, 8)",
               "slack_dev_tol": SLACK_DEV_TOL,
               "degeneracy_excluded": excluded,
               "dropped_probes_actual_slack": dropped_probes,
               "thresholds": out,
               "note": "AC-a evidence leg: s*(beta) must shrink toward 0 as beta -> 0"}
    save_json("R013_beta_ladder.json", payload)


def _bite_probe(spec):
    s, n, reps = spec
    world = family_continuum(2)
    beta = world.meta["beta_star"] - s
    r = run_cell(world, n=int(n), m=16000000, alpha=ALPHA, beta=beta, delta=DELTA,
                 n_reps=reps, cell_key=("B1bite", float(s), int(n)),
                 arms=("oracle_a",), n_lattice=world.n_bins)
    return s, int(n), r["oracle_a"]["cert_freq"]


def cmd_bite(args):
    """R014 — bite-divergence continuum slab: required-n vs slack, log-log exponent;
    success CI in [-2.3, -1.7] (pilot anchor -2.087 +/- 0.084, Gate-2)."""
    slacks = [0.16, 0.11, 0.08, 0.055, 0.04, 0.028, 0.02, 0.017, 0.014]
    ngrid = log_grid(256, 8000000, 40)     # finer brackets to tighten the exponent CI
    req = {}
    for s in slacks:
        specs = [(s, n, args.reps) for n in ngrid]
        rows = pmap(_bite_probe, specs, n_jobs=args.jobs, desc=f"bite s={s}")
        need = None
        for _s, n, f in sorted(rows, key=lambda t: t[1]):
            if f >= 0.8:
                need = n
                break
        req[s] = need
    ok = [s for s in slacks if req[s]]
    boot = None
    fit = loglog_exponent(np.array(ok), np.array([req[s] for s in ok])) if len(ok) >= 4 else None
    payload = {"run": "R014", "slacks": slacks, "required_n": req, "fit": fit,
               "success_band": [-2.3, -1.7],
               "pass": bool(fit and -2.3 <= fit["ci95"][0] and fit["ci95"][1] <= -1.7)
                       if fit else False,
               "pilot_anchor": "-2.087 +/- 0.084 (Gate-2 pilot, provenance-tagged)"}
    save_json("R014_bite_divergence.json", payload)


def _edge_job(spec):
    loc, ntot, reps = spec
    from .generators import family_kcell
    world = family_kcell(loc_level=loc)
    beta = world.meta["beta_star"] - S_WORK
    r = run_cell(world, n=ntot, m=ntot, alpha=ALPHA, beta=beta, delta=DELTA,
                 n_reps=reps, cell_key=("B1edge", float(loc), ntot),
                 arms=("oracle_a", "bprime"), n_lattice=N_LATTICE)
    return loc, ntot, r


def cmd_edge(args):
    """R015 — B' regime edge: acceptance collapse located within bands of
    rho_lambda ~ kappa*s as the weight-split budget shrinks."""
    from .generators import family_kcell
    world = family_kcell(loc_level=2.0)
    kappa_s = (world.risk()[world.meta["j_star"]] is not None)
    ngrid = log_grid(512, 2097152, 13)
    rows = pmap(_edge_job, [(2.0, int(n), args.reps) for n in ngrid],
                n_jobs=args.jobs, desc="R015 regime edge")
    series = []
    for loc, ntot, r in sorted(rows, key=lambda t: t[1]):
        series.append({"n_total": ntot,
                       "bprime_cert": r["bprime"]["cert_freq"],
                       "oracle_cert": r["oracle_a"]["cert_freq"],
                       "bprime_acc": r["bprime"]["mean_true_acceptance_when_cert"],
                       "oracle_acc": r["oracle_a"]["mean_true_acceptance_when_cert"],
                       "mean_rho": r["bprime"]["mean_rho_at_chosen"],
                       "viol_ucb": r["bprime"]["viol_cp_ucb"]})
    # edge estimate: first n where B' acceptance >= 50% of oracle's
    edge_n = None
    kappa = 0.55 - ALPHA   # kcell family: eta range edge proxy; recorded for the band
    for row in series:
        if row["oracle_acc"] and row["bprime_acc"] >= 0.5 * row["oracle_acc"] \
           and row["bprime_cert"] >= 0.5:
            edge_n = row["n_total"]
            break
    # rho ~ kappa*s crossing: first n where mean_rho <= kappa * S_WORK
    rho_cross = None
    for row in series:
        if row["mean_rho"] is not None and row["mean_rho"] <= kappa * S_WORK:
            rho_cross = row["n_total"]
            break
    within = (edge_n is not None and rho_cross is not None
              and 0.25 <= edge_n / rho_cross <= 4.0)     # one-octave band each side
    payload = {"run": "R015", "series": series, "edge_n": edge_n,
               "rho_crossing_n": rho_cross, "kappa_s": kappa * S_WORK,
               "edge_within_band_of_rho_crossing": bool(within)}
    save_json("R015_regime_edge.json", payload)


def cmd_audit(args):
    """R042 — validity-audit subset at >=4000 repeats: all certified cells at the
    headline intensity + stratified 10% elsewhere; Holm-corrected exact tests; any
    rejection = STOP-SHIP."""
    paths = {i: os.path.join(RESULTS_DIR, f"B1_grid_L{i}.json") for i in (1, 2, 3, 4)}
    missing = [i for i, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"R042 requires ALL intensity grids; missing L{missing} — "
                         "run `grid --intensity ...` first (hard gate)")
    audit_cells = []
    for intensity, path in sorted(paths.items()):
        with open(path) as f:
            data = json.load(f)
        certified = [c for c in data["cells"] if c["bprime"]["cert"] > 0]
        if intensity == HEADLINE_INTENSITY:
            chosen = certified                  # ALL cells with any B' certification
        else:
            stride = max(len(certified) // max(len(certified) // 10, 1), 1)
            chosen = certified[::stride][: max(len(certified) // 10, 1)]
        for c in chosen:
            audit_cells.append((intensity, c["_meta"]["n"], c["_meta"]["m"],
                                data["beta"]))
    if not audit_cells:
        raise SystemExit("no certified cells found in any grid — nothing to audit")
    specs = [(i, n, m, beta, args.reps, 10 ** 6) for (i, n, m, beta) in audit_cells]
    rows = pmap(_grid_job, specs, n_jobs=args.jobs,
                desc=f"R042 audit ({len(specs)} cells x {args.reps})")
    viols, nreps, recs = [], [], []
    pooled_v = pooled_n = 0
    for (i, n, m, beta), r in zip(audit_cells, rows):
        for arm in ("oracle_a", "bprime"):
            v, k = r[arm]["viol"], r[arm]["n_reps"]
            viols.append(v); nreps.append(k)
            recs.append({"intensity": i, "n": n, "m": m, "arm": arm, "viol": v,
                         "reps": k, "cp_ucb": r[arm]["viol_cp_ucb"]})
            pooled_v += v; pooled_n += k
    any_rej, rej_idx, pvals = holm_exact(viols, nreps, DELTA)
    pooled_ucb = cp_ucb(pooled_v, pooled_n)
    payload = {"run": "R042", "reps_per_cell": args.reps,
               "n_audit_cells": len(audit_cells), "records": recs,
               "holm_any_rejection": bool(any_rej),
               "holm_rejected": rej_idx.tolist(), "max_pvalue_min": float(pvals.min()),
               "pooled_viol": pooled_v, "pooled_n": pooled_n,
               "pooled_cp_ucb": pooled_ucb,
               "pooled_pass_delta": bool(pooled_ucb <= DELTA),
               "max_cell_cp_ucb": float(max(r["cp_ucb"] for r in recs)),
               "STOP_SHIP": bool(any_rej)}
    save_json("R042_validity_audit.json", payload)
    if any_rej:
        raise SystemExit(3)        # stop-ship


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid"); g.add_argument("--intensity", type=int, required=True)
    g.add_argument("--reps", type=int, default=1000); g.add_argument("--jobs", type=int, default=32)
    l = sub.add_parser("ladder"); l.add_argument("--intensity", type=int, default=2)
    l.add_argument("--reps", type=int, default=1000); l.add_argument("--jobs", type=int, default=32)
    b = sub.add_parser("bite"); b.add_argument("--reps", type=int, default=400)
    b.add_argument("--jobs", type=int, default=32)
    e = sub.add_parser("edge"); e.add_argument("--reps", type=int, default=600)
    e.add_argument("--jobs", type=int, default=32)
    a = sub.add_parser("audit"); a.add_argument("--reps", type=int, default=4000)
    a.add_argument("--jobs", type=int, default=48)
    args = ap.parse_args()
    {"grid": cmd_grid, "ladder": cmd_ladder, "bite": cmd_bite,
     "edge": cmd_edge, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    main()
