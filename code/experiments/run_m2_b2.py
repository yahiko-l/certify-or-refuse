"""M2 / B2 — the functional: matched-pair construction + audit (R017), required-n
sweeps oracle/B' (R018/R019), rank-correlation analysis (R020), second family (R043).

Registered: ratios {1.15,1.5,1.9,2.3,2.7,3.0,3.25,3.5} (8 pts), >=40 audited world
draws per ratio per family; required-n decision rule per R039 §6.4 (cert-prob CP-LCB
> tau=0.5 AND violation CP-UCB <= delta_tol; bracket ratio <= 1.1; MC >= 500/probe);
slack fixed 0.05; m fixed 1e5 (floor side saturated — isolates the n requirement).

Usage:
  python -m experiments.run_m2_b2 build --fam {1,2}     (R017 / R043 worlds + audit)
  python -m experiments.run_m2_b2 sweep --fam {1,2} [--arm bprime|oracle_a] [--jobs 48]
  python -m experiments.run_m2_b2 analyze               (R020)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .analysis import rank_correlations
from .generators import family_matched_pair, audit_pair
from .validity import run_cell, cp_ucb, cp_lcb, DELTA_TOL_FACTOR
from .runner_util import save_json, pmap, RESULTS_DIR

ALPHA, DELTA = 0.2, 0.05
RATIOS = [1.15, 1.5, 1.9, 2.3, 2.7, 3.0, 3.25, 3.5]
N_DRAWS = 40
SLACK = 0.05
M_FIXED = 100000
REPS_PROBE = 500


def cmd_build(args):
    """R017/R043: construct >= N_DRAWS AUDITED pairs per ratio; failing pairs are
    regenerated at the next draw index (never silently kept); audit table published."""
    table, audits = {}, []
    for ratio in RATIOS:
        kept, draw, attempts = [], 0, 0
        while len(kept) < N_DRAWS and attempts < N_DRAWS * 30:
            got = family_matched_pair(ratio, draw, fam=args.fam)
            attempts += 1
            if got is not None:
                lo, hi, audit = got
                audit["draw"] = draw
                audit["ratio_target"] = ratio
                ok = audit_pair(audit, ratio)
                audit["audit_pass"] = ok
                audits.append(audit)
                if ok:
                    kept.append(draw)
            draw += 1
        table[str(ratio)] = {"kept_draws": kept, "n_attempts": attempts,
                             "n_regenerated": attempts - len(kept)}
        print(f"fam{args.fam} ratio {ratio}: kept {len(kept)}/{attempts} attempts")
    payload = {"run": "R017" if args.fam == 1 else "R043", "fam": args.fam,
               "ratios": RATIOS, "n_draws": N_DRAWS, "audit_tolerances":
               {"ess_rel_dev": 0.01, "frontier_dev": 0.005, "ratio_rel_dev": 0.10},
               "table": table, "audits": audits,
               "all_ratios_filled": all(len(v["kept_draws"]) >= N_DRAWS
                                        for v in table.values())}
    save_json(f"B2_pairs_fam{args.fam}.json", payload)
    if not payload["all_ratios_filled"]:
        raise SystemExit(f"B2 fam{args.fam}: some ratio has < {N_DRAWS} audited pairs "
                         "— construction must be widened, never silently thinned "
                         "(hard gate, code-review round-1 fix)")


def _required_n(world, arm, key):
    """Registered bisection (R039 §6.4): probe passes iff cert-count CP-LCB > 0.5 AND
    violation CP-UCB <= delta_tol; geometric bracket then bisection to ratio <= 1.1."""
    beta = world.meta["beta_star"] - SLACK

    def probe(n, tag):
        r = run_cell(world, n=int(n), m=M_FIXED, alpha=ALPHA, beta=beta, delta=DELTA,
                     n_reps=REPS_PROBE, cell_key=("B2", key, arm, tag, int(n)),
                     arms=(arm,))
        t = r[arm]
        ok = (cp_lcb(t["cert"], t["n_reps"]) > 0.5
              and t["viol_cp_ucb"] <= DELTA_TOL_FACTOR * DELTA)
        return ok

    lo, hi = 125, None
    n = 250
    while n <= 536_870_912:
        if probe(n, "bracket"):
            hi = n
            break
        lo = n
        n *= 2
    if hi is None:
        return np.inf
    while hi / lo > 1.1:
        mid = int(np.sqrt(lo * hi))
        if probe(mid, "bisect"):
            hi = mid
        else:
            lo = mid
    return hi


def _sweep_job(spec):
    fam, ratio, draw, arm = spec
    lo, hi, audit = family_matched_pair(ratio, draw, fam=fam)
    jacc = lo.meta["accepted_prefix"]
    rows = []
    for tag, wld in (("LO", lo), ("HI", hi)):
        rn = _required_n(wld, arm, (fam, ratio, draw, tag))
        rows.append({"fam": fam, "ratio": ratio, "draw": draw, "member": tag,
                     "arm": arm, "required_n": rn,
                     "localized": wld.localized(jacc),
                     "global_ess": wld.global_ess_frac(),
                     "beta_star": wld.meta["beta_star"]})
    return rows


def cmd_sweep(args):
    with open(os.path.join(RESULTS_DIR, f"B2_pairs_fam{args.fam}.json")) as f:
        pairs = json.load(f)
    specs = []
    for ratio in RATIOS:
        for draw in pairs["table"][str(ratio)]["kept_draws"]:
            specs.append((args.fam, ratio, draw, args.arm))
    rows = pmap(_sweep_job, specs, n_jobs=args.jobs,
                desc=f"B2 sweep fam{args.fam} {args.arm} ({len(specs)} pairs)")
    flat = [r for pair in rows for r in pair]
    run_id = {("bprime", 1): "R019", ("oracle_a", 1): "R018"}.get(
        (args.arm, args.fam), f"R0{18 if args.arm == 'oracle_a' else 19}_fam{args.fam}")
    save_json(f"B2_required_n_fam{args.fam}_{args.arm}.json",
              {"run": run_id, "fam": args.fam, "arm": args.arm, "slack": SLACK,
               "m_fixed": M_FIXED, "rows": flat})


def cmd_analyze(args):
    """R020 — Spearman (Kendall secondary) of required-n with localized vs global ESS,
    bootstrap CIs, per family x arm; plus monotonicity of the pair required-n ratio."""
    out = {}
    for fam in (1, 2):
        for arm in ("oracle_a", "bprime"):
            path = os.path.join(RESULTS_DIR, f"B2_required_n_fam{fam}_{arm}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                rows = json.load(f)["rows"]
            rn = np.array([r["required_n"] for r in rows], float)
            loc = np.array([r["localized"] for r in rows], float)
            ges = np.array([r["global_ess"] for r in rows], float)
            res = rank_correlations(rn, loc, ges)
            # within-pair required-n ratio vs localized ratio (monotonicity)
            by_pair = {}
            for r in rows:
                by_pair.setdefault((r["ratio"], r["draw"]), {})[r["member"]] = r
            ratios, rnr = [], []
            for (ratio, draw), d in sorted(by_pair.items()):
                if "LO" in d and "HI" in d and np.isfinite(d["LO"]["required_n"]) \
                        and np.isfinite(d["HI"]["required_n"]):
                    ratios.append(d["HI"]["localized"] / d["LO"]["localized"])
                    rnr.append(d["HI"]["required_n"] / d["LO"]["required_n"])
            mono = (np.corrcoef(ratios, rnr)[0, 1] if len(ratios) >= 3 else None)
            out[f"fam{fam}_{arm}"] = {"rank_corr": res,
                                      "pair_ratio_pearson": float(mono) if mono else None,
                                      "n_pairs": len(ratios)}
    required = [f"fam{f}_{a}" for f in (1, 2) for a in ("oracle_a", "bprime")]
    missing = [k for k in required if k not in out]
    both_pass = (not missing) and all(out[k]["rank_corr"]["success"] for k in out
                                      if k.endswith("bprime"))
    save_json("R020_rank_correlations.json",
              {"run": "R020", "per_family_arm": out, "missing_inputs": missing,
               "success_both_families_bprime": both_pass,
               "pilot_anchor": "3.15x localized -> 3.37x required-n (Gate-2)"})
    if missing:
        raise SystemExit(f"R020 incomplete — missing sweeps: {missing} (hard gate; "
                         "run all family x arm sweeps first)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--fam", type=int, required=True)
    s = sub.add_parser("sweep"); s.add_argument("--fam", type=int, required=True)
    s.add_argument("--arm", default="bprime"); s.add_argument("--jobs", type=int, default=48)
    sub.add_parser("analyze")
    args = ap.parse_args()
    {"build": cmd_build, "sweep": cmd_sweep, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
