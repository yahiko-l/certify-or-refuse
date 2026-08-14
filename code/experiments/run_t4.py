"""T4 — two-axis completion: the floor (m) bite + B′ positive-certification grid.

STATUS (2026-06-17, user decision — keep only clearly-supporting experiments active):
  * bprime_positive (R066) = KEPT, clear support → experiments/results/R066_t4_bprime_positive.json
  * floorbite       (R065) = PAUSED FOR EVALUATION (qualified support: slope −1.835, CI contains
                    −2 but not fully inside the [−2.3,−1.7] band, pass_oracle=false). Result moved
                    to the authors' archive (R065; excluded from the release, backs no paper claim). Code kept here, runnable for
                    the user's evaluation; NOT a paper-supporting result unless re-adjudicated.

NEW sweeps over the VALIDATED kernel (validity.run_cell / certificate.py); no formula or
constant changes (R039 discipline intact) — this module only chooses (n, m, slack) inputs
and aggregates onsets/margins, never touching the registered statistics. Companion to
run_t1a (B′ envelope), run_t2 (labeled bite breadth), run_t3 (misspecification breakage).

Closes two gaps flagged by the 2026-06-17 GPT-5.5 experiment-sufficiency review:

  floorbite (R065): the FLOOR-axis analogue of the labeled bite (R014/R060). R014 probes
    the labeled-risk axis (required-n ∝ s^-2, floor freed with big m). Here we FREE THE RISK
    AXIS (big n) and sweep the floor: required-m vs slack near the certifiable boundary. The
    map's floor axis (Claim 1 lower √(β/m); Claim 2 upper m ≳ β log/s²) predicts the SAME
    s^-2 bite. The oracle-A floor test is weight-free (Bernstein LCB on coverage), so this is
    the cleanest possible isolation of the unlabeled-target axis. Same continuum family as
    R014 for direct comparability. B′ is EXCLUDED here on purpose: its target split m_w = m/2
    means lowering m also starves weight estimation, coupling the risk/nuisance axes into a
    floor sweep — so only the weight-free oracle-A arm yields a clean floor-axis measurement.

  bprime_positive (R066): a TARGETED B′ positive-certification grid. The full validity grid
    (R054) shows B′ certifies only 8/1024 cells (harsh-constants regime) — a thin positive
    existence proof. Here we zoom into a regime where B′ DOES certify, run it to high reps,
    and show (i) B′ reaches cert_freq → 1 as (n,m) grow past its onset, (ii) 0 violations
    across thousands of certifying replications (validity at the issued point), and (iii) the
    issued policy is NON-TRIVIAL: true risk ≤ α with margin AND true coverage ≥ β with margin
    (not a near-zero-acceptance pseudo-win). On-class K-cell world, K=4 (cheapest partition),
    same operating geometry as the paper's §5 constant.

Subcommands:
  python -m experiments.run_t4 floorbite          (R065: floor-axis bite)
  python -m experiments.run_t4 bprime_positive     (R066: B′ positive-cert grid)
  python -m experiments.run_t4 all
  [--reps N] [--jobs J] [--smoke]
"""
from __future__ import annotations

import argparse

import numpy as np

from .analysis import loglog_exponent
from .generators import family_continuum, family_kcell
from .validity import run_cell, cp_ucb
from .runner_util import save_json, pmap, log_grid

ALPHA, DELTA = 0.2, 0.05

# ---- floor bite (R065): RISK axis freed, FLOOR axis bites
BIG_N = 8_000_000                       # risk + nuisance freed → only the floor (m) axis binds
FLOOR_SLACKS = [0.16, 0.11, 0.08, 0.055, 0.04, 0.028, 0.02]   # mirror R052/R014 slack ladder
FLOOR_MGRID = log_grid(32, 4_000_000, 44)     # brackets required-m across the slack ladder
FLOOR_BAND = [-2.3, -1.7]               # same pre-registered acceptance band as the labeled bite

# ---- B′ positive grid (R066)
POS_SLACKS = [0.20, 0.12]               # an easy point + the paper's §5 operating slack
POS_NGRID = log_grid(32768, 4194304, 8)         # 2^15 .. 2^22 (n = m), brackets the B′ onset
POS_K = 4                               # cheapest on-class partition
N_LATTICE_POS = 50                      # matches run_t1a / validity B′ runs


# ============================================================ floor bite (R065)

def _floor_job(spec):
    """One (slack, m) probe: free the risk axis (n = BIG_N), measure floor cert_freq."""
    s, m, reps, arm = spec
    world = family_continuum(2)
    beta = float(world.meta["beta_star"]) - s
    if beta <= 0.02:
        return (s, int(m), None)
    r = run_cell(world, n=BIG_N, m=int(m), alpha=ALPHA, beta=beta, delta=DELTA,
                 n_reps=reps, arms=(arm,), n_lattice=world.n_bins,
                 cell_key=("T4_floorbite", arm, float(s), int(m)))
    return (s, int(m), r[arm]["cert_freq"])


def _required_m(rows):
    """Smallest m whose cert_freq ≥ 0.8 (R014 onset convention); None if never (censored)."""
    return next((m for _, m, f in sorted(rows, key=lambda t: t[1])
                 if f is not None and f >= 0.8), None)


def _floor_axis(arm, mgrid, slacks, reps, jobs):
    bstar = float(family_continuum(2).meta["beta_star"])
    use_s = [s for s in slacks if bstar - s > 0.05]
    req, series = {}, {}
    for s in use_s:
        rows = pmap(_floor_job, [(s, m, reps, arm) for m in mgrid],
                    n_jobs=jobs, desc=f"T4 floorbite[{arm}] s={s}")
        req[str(s)] = _required_m(rows)
        series[str(s)] = [{"m": m, "cert_freq": f} for _, m, f in sorted(rows, key=lambda t: t[1])]
    ok = [s for s in use_s if req[str(s)]]
    fit = (loglog_exponent(np.array(ok, float), np.array([req[str(s)] for s in ok], float))
           if len(ok) >= 4 else None)
    in_band = (bool(FLOOR_BAND[0] <= fit["ci95"][0] and fit["ci95"][1] <= FLOOR_BAND[1])
               if fit else None)
    return {"beta_star": bstar, "required_m": req, "n_points": len(ok),
            "slope": (fit["slope"] if fit else None),
            "ci95": (fit["ci95"] if fit else None),
            "curvature_ci": (fit.get("curvature_ci") if fit else None),
            "curvature_contains_0": (fit.get("curvature_contains_0") if fit else None),
            "ci_in_band": in_band, "series_by_slack": series}


def cmd_floorbite(args):
    mgrid = log_grid(32, 4_000_000, 12) if args.smoke else FLOOR_MGRID
    slacks = [0.11, 0.055, 0.028, 0.016] if args.smoke else FLOOR_SLACKS
    # Oracle-A ONLY: the floor test is weight-free (Bernstein LCB on coverage), so with the
    # risk axis freed this isolates the unlabeled-target axis cleanly — the exact mirror of the
    # oracle-arm labeled bite (R014). B′ is deliberately NOT used here: lowering m also starves
    # its target weight split (m_w = m/2), coupling the risk/nuisance axes into the probe, so
    # B′ required-m would not be a clean floor-axis measurement.
    oracle = _floor_axis("oracle_a", mgrid, slacks, args.reps, args.jobs)
    payload = {
        "run": "R065", "family": "continuum", "intensity": 2, "alpha": ALPHA, "delta": DELTA,
        "reps": args.reps, "big_n": BIG_N, "axis": "floor (unlabeled target m)",
        "pred_slope": -2.0, "success_band": FLOOR_BAND, "arm": "oracle_a", "oracle_a": oracle,
        "pass_oracle": (oracle["ci_in_band"] is True),
        "claim": "compatible with the predicted s^-2 floor-axis bite (slope -1.835, 95% CI "
                 "contains -2.0); NOT a passed-band / exact-(-2) result (CI upper endpoint "
                 "pokes just above the -1.7 band edge — pass_oracle is false, reported honestly)",
        "note": "Floor-axis bite = the symmetric mirror of the R014 labeled bite. R014 freed "
                "the FLOOR (big m) and bit the labeled-risk axis (required-n ∝ s^-2, oracle arm); "
                "here we free the RISK axis (n=8e6) and bite the FLOOR axis (required-m vs slack). "
                "The map's floor axis (Claim 1 √(β/m); Claim 2 m ≳ β log/s²) predicts the same "
                "s^-2. Oracle-A floor test is weight-free (Bernstein LCB on coverage) — the "
                "cleanest isolation of the unlabeled-target axis; B′ is excluded because its "
                "m_w=m/2 weight split couples the risk axis into a floor sweep. Onset at "
                "cert_freq ≥ 0.8 (R014 convention).",
    }
    save_json("R065_t4_floor_bite.json", payload)
    print({"oracle": {"slope": _r(oracle["slope"]), "ci95": oracle["ci95"],
                      "in_band": oracle["ci_in_band"], "pts": oracle["n_points"],
                      "pred_slope": -2.0, "required_m": oracle["required_m"]}})
    return payload


# ===================================================== B′ positive grid (R066)

def _pos_job(spec):
    """One (slack, n=m) B′/oracle positive-certification cell with truth margins."""
    s, n, reps = spec
    world = family_kcell(K_world=4, loc_level=2.0, B_cls=5.0, alpha=ALPHA)
    bstar = float(world.meta["beta_star"])
    beta = bstar - s
    r = run_cell(world, n=int(n), m=int(n), alpha=ALPHA, beta=beta, delta=DELTA,
                 n_reps=reps, arms=("oracle_a", "bprime"), K_cells=POS_K,
                 n_lattice=N_LATTICE_POS, collect_truth_margins=True,
                 cell_key=("T4_pos", float(s), int(n)))
    out = {"slack": s, "n": int(n), "beta": beta, "beta_star": bstar}
    for arm in ("oracle_a", "bprime"):
        a = r[arm]
        out[arm] = {
            "cert_freq": a["cert_freq"], "cert": a["cert"], "viol": a["viol"],
            "viol_cp_ucb": a["viol_cp_ucb"], "n_reps": a["n_reps"],
            "mean_true_cov": a.get("mean_true_acceptance_when_cert"),
            "min_true_cov": a.get("min_true_cov_when_cert"),
            "mean_true_risk": a.get("mean_true_risk_when_cert"),
            "max_true_risk": a.get("max_true_risk_when_cert"),
            "risk_margin_mean": a.get("risk_margin_mean"),
            "cov_margin_min": a.get("cov_margin_min"),
        }
    out["bprime"]["mean_rho"] = r["bprime"].get("mean_rho_at_chosen")
    return out


def _summarize_arm(cells, arm, cert_thr=0.9):
    """Pool validity over all certifying reps; find the positive-cert onset + its margins."""
    tot_cert = sum(c[arm]["cert"] for c in cells)
    tot_viol = sum(c[arm]["viol"] for c in cells)
    # per-slack onset (smallest n with cert_freq ≥ thr) + the margins at that point
    per_slack = {}
    for s in sorted({c["slack"] for c in cells}):
        sc = sorted([c for c in cells if c["slack"] == s], key=lambda c: c["n"])
        on = next((c for c in sc if c[arm]["cert_freq"] >= cert_thr), None)
        top = sc[-1]
        per_slack[str(s)] = {
            "onset_n": (on["n"] if on else None),
            "onset_cert_freq": (on[arm]["cert_freq"] if on else None),
            "onset_min_true_cov": (on[arm]["min_true_cov"] if on else None),
            "onset_max_true_risk": (on[arm]["max_true_risk"] if on else None),
            "onset_cov_margin_min": (on[arm]["cov_margin_min"] if on else None),
            "onset_risk_margin_mean": (on[arm]["risk_margin_mean"] if on else None),
            "top_n": top["n"], "top_cert_freq": top[arm]["cert_freq"],
            "top_viol": top[arm]["viol"],
        }
    return {
        "total_cert_reps": int(tot_cert), "total_viol": int(tot_viol),
        "pooled_viol_cp_ucb": cp_ucb(int(tot_viol), max(int(tot_cert), 1)),
        "per_slack": per_slack,
    }


def cmd_bprime_positive(args):
    ngrid = log_grid(32768, 4194304, 4) if args.smoke else POS_NGRID
    slacks = [0.20] if args.smoke else POS_SLACKS
    reps = 300 if args.smoke else args.reps
    specs = [(s, n, reps) for s in slacks for n in ngrid]
    cells = pmap(_pos_job, specs, n_jobs=args.jobs, desc=f"T4 bprime_positive ({len(specs)} cells)")
    cells = sorted(cells, key=lambda c: (c["slack"], c["n"]))
    bp = _summarize_arm(cells, "bprime")
    orc = _summarize_arm(cells, "oracle_a")
    # headline: does B′ achieve a NON-TRIVIAL valid positive certification?
    bp_pos = [c for c in cells if c["bprime"]["cert_freq"] >= 0.9]
    # "valid" predicate uses the WORST-case truth at the issued λ (max true risk ≤ α and min
    # true coverage ≥ β over all certifying reps), matching the flag name — not a mean.
    nontrivial = [c for c in bp_pos
                  if c["bprime"]["min_true_cov"] is not None
                  and c["bprime"]["cov_margin_min"] >= 0.0
                  and c["bprime"]["max_true_risk"] is not None
                  and c["bprime"]["max_true_risk"] <= ALPHA
                  and c["bprime"]["mean_true_cov"] >= 0.5]
    payload = {
        "run": "R066", "family": "kcell", "K": POS_K, "loc_level": 2.0, "B_class": 5.0,
        "alpha": ALPHA, "delta": DELTA, "n_lattice": N_LATTICE_POS, "reps": reps,
        "slacks": slacks, "ngrid": [int(n) for n in ngrid],
        "bprime_summary": bp, "oracle_a_summary": orc,
        "bprime_positive_exists": bool(bp_pos),
        "bprime_nontrivial_valid_exists": bool(nontrivial),
        "bprime_total_viol": bp["total_viol"],
        "bprime_pooled_viol_cp_ucb": bp["pooled_viol_cp_ucb"],
        "cells": cells,
        "note": "Targeted B′ positive-certification grid. The full validity grid (R054) shows "
                "B′ certifying only 8/1024 cells; this zooms into a regime where B′ DOES certify "
                "and shows the full positive ladder: cert_freq → 1 as (n,m) pass the onset, 0 "
                "violations across all certifying reps (validity at the issued λ), and the issued "
                "policy is non-trivial — true coverage ≥ β with margin AND true risk ≤ α with "
                "margin (mean_true_cov ≥ 0.5, not a near-zero-acceptance pseudo-win). Oracle-A "
                "shown alongside as the cheaper rate-reference arm: B′ 0 violations in all "
                "certifying reps; oracle-A 1 violation (a single near-boundary risk-side event, "
                "max true risk ≈ 0.20004 just above α), both pooled CP-UCB ≪ δ = 0.05.",
    }
    save_json("R066_t4_bprime_positive.json", payload)
    print({"bprime_positive_exists": payload["bprime_positive_exists"],
           "bprime_nontrivial_valid_exists": payload["bprime_nontrivial_valid_exists"],
           "bprime_total_cert_reps": bp["total_cert_reps"],
           "bprime_total_viol": bp["total_viol"],
           "bprime_pooled_viol_cp_ucb": _r(bp["pooled_viol_cp_ucb"], 6),
           "bprime_onsets": {s: d["onset_n"] for s, d in bp["per_slack"].items()},
           "bprime_margins_at_onset": {
               s: {"min_cov": _r(d["onset_min_true_cov"]), "max_risk": _r(d["onset_max_true_risk"]),
                   "beta": _r(next(c["beta"] for c in cells if c["slack"] == float(s)))}
               for s, d in bp["per_slack"].items()}})
    return payload


def _r(x, k=3):
    return round(x, k) if isinstance(x, (int, float)) else x


def cmd_all(args):
    fb = cmd_floorbite(args)
    bp = cmd_bprime_positive(args)
    summary = {
        "run": "R065_R066_T4_SUMMARY",
        "floor_bite_oracle_slope": fb["oracle_a"]["slope"],
        "floor_bite_oracle_ci95": fb["oracle_a"]["ci95"],
        "floor_bite_oracle_in_band": fb["oracle_a"]["ci_in_band"],
        "floor_bite_pred_slope": -2.0,
        "bprime_positive_exists": bp["bprime_positive_exists"],
        "bprime_nontrivial_valid_exists": bp["bprime_nontrivial_valid_exists"],
        "bprime_total_viol": bp["bprime_total_viol"],
        "bprime_pooled_viol_cp_ucb": bp["bprime_pooled_viol_cp_ucb"],
    }
    save_json("R065_R066_t4_summary.json", summary)
    print("\n=== T4 SUMMARY ===")
    print(summary)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("floorbite", "bprime_positive", "all"):
        p = sub.add_parser(name)
        p.add_argument("--reps", type=int, default=500)
        p.add_argument("--jobs", type=int, default=48)
        p.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    {"floorbite": cmd_floorbite, "bprime_positive": cmd_bprime_positive,
     "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
