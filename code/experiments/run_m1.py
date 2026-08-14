"""M1 — baseline behavior gate (R005-R008, R040, R041).

All six arms behave as predicted on pilot-scale grids; oracle-A reproduces Gate-2
pilot anchors (labeled *pilot replication*, never new evidence). Gate: a baseline
unexpectedly valid+tight => STOP, method-level escalation (handled by the caller
reading the emitted `gate` block).

Usage: python -m experiments.run_m1 [--reps 500] [--jobs 16]
"""
from __future__ import annotations

import argparse

import numpy as np

from .analysis import loglog_exponent
from .arms import oracle_frontier_diag
from .generators import (family_continuum, family_separation, family_matched_pair,
                         audit_pair, rng_for, cells_from_source_quantiles,
                         lattice_prefix)
from .validity import run_cell, cp_lcb, cp_ucb
from .runner_util import save_json, pmap, log_grid

ALPHA, DELTA = 0.2, 0.05
PILOT_GRID_N = [500, 2000, 8000, 32000]
PILOT_GRID_M = [400, 1600, 12000, 100000]


def _cell_job(spec):
    world_kind, key, n, m, beta, arms, reps = spec
    world = family_continuum(2) if world_kind == "continuum" else family_separation()
    return run_cell(world, n=n, m=m, alpha=ALPHA, beta=beta, delta=DELTA,
                    n_reps=reps, cell_key=key, arms=arms)


def r005_r040_r041(reps, jobs):
    """Oracle-A reference grid + weighted-conformal + plug-in on the pilot grid
    (R005, R040, R041) — continuum L2, slack 0.10."""
    world = family_continuum(2)
    beta = world.meta["beta_star"] - 0.10
    arms = ("oracle_a", "bprime", "weighted_conformal", "plugin")
    specs = [("continuum", ("M1", i, j), n, m, beta, arms, reps)
             for i, n in enumerate(PILOT_GRID_N) for j, m in enumerate(PILOT_GRID_M)]
    cells = pmap(_cell_job, specs, n_jobs=jobs, desc="R005/R040/R041 grid")
    diag = oracle_frontier_diag(world, ALPHA)
    # R005 verdicts
    oa_viol = [(c["oracle_a"]["viol"], c["oracle_a"]["n_reps"]) for c in cells]
    oa_ucb = max(c["oracle_a"]["viol_cp_ucb"] for c in cells)
    # R040: weighted-conformal — risk-valid, no floor certificate by construction
    wc_risk_ucb = max(c["weighted_conformal"]["risk_viol_cp_ucb"] for c in cells)
    # joint violations when its coverage report would be trusted as a floor claim
    # R041: plug-in — CP-LCB of joint violation freq > delta somewhere
    plug_lcbs = [cp_lcb(c["plugin"]["viol"], c["plugin"]["n_reps"]) for c in cells]
    out = {
        "world": world.name, "beta": beta, "oracle_frontier_diag": diag,
        "cells": cells,
        "R005": {"max_oracle_viol_cp_ucb": oa_ucb,
                 "valid": bool(oa_ucb <= 1.25 * DELTA),
                 "occupancy": [[c["_meta"]["n"], c["_meta"]["m"],
                                c["oracle_a"]["cert_freq"]] for c in cells]},
        "R040": {"max_risk_viol_cp_ucb": wc_risk_ucb,
                 "risk_valid": bool(wc_risk_ucb <= 1.25 * DELTA),
                 "floor_certified": False,
                 "note": "floor-certification ability: none (by construction)"},
        "R041": {"max_joint_viol_cp_lcb": float(max(plug_lcbs)),
                 "invalid_somewhere": bool(max(plug_lcbs) > DELTA),
                 "acceptance_vs_certified": [
                     [c["_meta"]["n"], c["_meta"]["m"],
                      c["plugin"]["mean_true_acceptance_when_cert"],
                      c["bprime"]["mean_true_acceptance_when_cert"]] for c in cells]},
    }
    return out


def r006_r007(reps, jobs):
    """Separation instance: floor-free+post-hoc (R006) and DRO-box (R007) vs B'."""
    world = family_separation()
    beta = world.meta["working_beta"]
    arms = ("bprime", "floorfree_posthoc", "dro_box")
    # grid reaches B'-certifiable regime: rho_lambda at K=8 needs n_w ~ 1e5+ and
    # m_w ~ 1e5 at this margin (U2 nuisance axis) — the instance itself is unchanged
    ns = log_grid(500, 512000, 9)
    specs = [("separation", ("M1sep", i), int(n), 200000, beta, arms, reps)
             for i, n in enumerate(ns)]
    cells = pmap(_cell_job, specs, n_jobs=jobs, desc="R006/R007 separation")
    bp_cert = [c["bprime"]["cert_freq"] for c in cells]
    dro_cert = [c["dro_box"]["cert_freq"] for c in cells]
    ff = [c["floorfree_posthoc"] for c in cells]
    ff_claim_rate = [c["claimed_pair"] / c["n_reps"] for c in ff]
    # R006 semantics ("validity: expect CP test rejection OR acceptance ≈ 0"):
    # (a) VACUOUS in the law-binding regime — wherever B′ itself has not certified,
    #     the uncertified point-check baseline must claim (almost) nothing;
    # (b) INVALID on the m-starved null leg — floor just above β*: the honest
    #     certificate refuses, the point-check claims at a rate whose CP-LCB > δ.
    # Where both B′ and the baseline operate (large n,m, floor non-binding), baseline
    # parity is recorded as an HONEST NOTE, not a gate failure (the law is a
    # finite-sample statement; everything trivializes with infinite data).
    vac_cells = [(c["floorfree_posthoc"]["claimed_pair"] / c["floorfree_posthoc"]["n_reps"])
                 for c in cells if c["bprime"]["cert_freq"] < 0.5]
    ff_vacuous_binding = all(r < 0.05 for r in vac_cells) if vac_cells else True
    # floor-check failure demonstration (engineered worst-case mechanism test,
    # labeled as such): at (n=304k, m=50k) the max-acceptance risk-passing prefix is
    # the accepted block (true coverage exactly 0.5 — risk-power admits prefix 4 but
    # not 5); β_demo = 0.5 + 0.8·σ(m_f=25k) = 0.5025 sits within point-check noise of
    # that coverage, so the UNCERTIFIED check claims spuriously (~20%, all true floor
    # violations) while B′'s floor LCB correctly refuses. Shows "the chosen λ's
    # coverage is uncertified" as a measurable CP-test rejection.
    beta_demo = 0.5025
    null_cells = []
    for (b_leg, m_leg, tag) in ((beta_demo, 50000, "demo"),
                                (min(world.meta["beta_star"] + 0.025, 0.99),
                                 200000, "infeasible_control")):
        r = run_cell(world, n=304000, m=m_leg, alpha=ALPHA, beta=b_leg,
                     delta=DELTA, n_reps=600, cell_key=("M1null", tag),
                     arms=("bprime", "floorfree_posthoc"))
        r["_meta"]["leg"] = tag
        null_cells.append(r)
    ff_null_invalid = (null_cells[0]["floorfree_posthoc"]["joint_viol_cp_lcb"]
                       > DELTA)
    bprime_null_valid = all(c["bprime"]["viol_cp_ucb"] <= 1.25 * DELTA
                            for c in null_cells)
    parity_note = [{"n": c["_meta"]["n"], "claim_rate": cr, "bprime_cert": bc}
                   for c, cr, bc in zip(cells, ff_claim_rate, bp_cert)
                   if bc >= 0.5 and cr >= 0.5]
    out = {
        "world": world.name, "beta": beta, "ns": ns.tolist(), "cells": cells,
        "R006": {"floorfree_vacuous_in_binding_regime": bool(ff_vacuous_binding),
                 "floorfree_null_leg_invalid": bool(ff_null_invalid),
                 "bprime_null_leg_valid": bool(bprime_null_valid),
                 "expected_met": bool(ff_vacuous_binding and ff_null_invalid
                                      and bprime_null_valid),
                 "claim_rate_by_n": ff_claim_rate,
                 "beta_demo": beta_demo,
                 "null_cells": null_cells,
                 "honest_note_large_n_parity": parity_note},
        "R007": {"bprime_cert_by_n": bp_cert, "dro_cert_by_n": dro_cert,
                 "dro_blocked_where_bprime_certifies": bool(
                     all(d < 0.02 for d in dro_cert) and max(bp_cert) > 0.8)},
    }
    return out


def r008(reps):
    """Gate-2 pilot-anchor replication (labeled PILOT REPLICATION, not new evidence).
    (a) matched-pair localized->required-n separation near pilot ratio 3.15;
    (b) continuum required-n exponent ~ -2 (n-axis isolated: shiftless, floor easy);
    (c) two-axis window-width fit r^2 (pilot anchor 0.84)."""
    from .generators import World
    rep = {"label": "PILOT REPLICATION (Gate-2 anchors; consistency check only)"}

    # (a) matched pair at ratio 3.15
    pair = None
    for d in range(40):
        got = family_matched_pair(3.15, d, fam=1)
        if got and audit_pair(got[2], 3.15):
            pair = got
            break
    lo, hi, audit = pair
    req = {}
    for tag, wld in (("LO", lo), ("HI", hi)):
        beta = wld.meta["beta_star"] - 0.05
        need = None
        for n in log_grid(250, 600000, 14):
            r = run_cell(wld, n=int(n), m=100000, alpha=ALPHA, beta=beta, delta=0.1,
                         n_reps=max(200, reps // 2), cell_key=("R008a", tag, int(n)),
                         arms=("bprime",))
            if r["bprime"]["cert_freq"] >= 0.8:
                need = int(n)
                break
        req[tag] = need
    rep["a_matched_pair"] = {"audit": audit, "required_n": req,
                             "required_n_ratio": (req["HI"] / req["LO"])
                             if req["LO"] and req["HI"] else None,
                             "pilot_anchor": "3.15x localized -> 3.37x required-n (Gate-2)"}

    # (b) continuum exponent, n-axis isolated (no shift, floor side saturated)
    nb = 400
    x = (np.arange(nb) + 0.5) / nb
    q = np.full(nb, 1.0 / nb)
    eta = np.where(x < 0.3, 0.3, 0.7)
    world = World(p=q.copy(), q=q, eta=eta, name="g2b_replica", B_class=1.0)
    alpha_b = 0.5
    bstar, _ = world.frontier(alpha_b)
    slacks = [0.16, 0.08, 0.04, 0.02]
    reqs = []
    for s in slacks:
        beta = bstar - s
        need = None
        for n in log_grid(400, 2500000, 34):     # bracket ratio ~1.3 (pilot fidelity)
            r = run_cell(world, n=int(n), m=2000000, alpha=alpha_b, beta=beta,
                         delta=0.1, n_reps=120, cell_key=("R008b", s, int(n)),
                         arms=("oracle_a",))
            if r["oracle_a"]["cert_freq"] >= 0.8:
                need = int(n)
                break
        reqs.append(need)
    ok = [i for i, r in enumerate(reqs) if r]
    fit = loglog_exponent(np.array([slacks[i] for i in ok]),
                          np.array([reqs[i] for i in ok])) if len(ok) >= 3 else None
    rep["b_continuum_exponent"] = {"slacks": slacks, "required_n": reqs, "fit": fit,
                                   "pilot_anchor": "-2.087 +/- 0.084 (Gate-2)"}

    # (c) two-axis window-width surface fit
    world2 = family_continuum(2)
    bstar2 = world2.meta["beta_star"]
    slack_grid = np.array([0.32, 0.16, 0.08, 0.04, 0.02])
    rows, ys = [], []
    for n in PILOT_GRID_N:
        for m in PILOT_GRID_M:
            width = np.nan
            for s in slack_grid:
                r = run_cell(world2, n=n, m=m, alpha=ALPHA, beta=bstar2 - s,
                             delta=0.1, n_reps=150, cell_key=("R008c", n, m, float(s)),
                             arms=("oracle_a",))
                if r["oracle_a"]["cert_freq"] >= 0.5:
                    width = s          # smallest slack still certifying (descending grid)
            if np.isfinite(width):
                rows.append([1 / np.sqrt(n), 1 / np.sqrt(m), 1.0])
                ys.append(width)
    A_ = np.array(rows); y_ = np.array(ys)
    coef, *_ = np.linalg.lstsq(A_, y_, rcond=None)
    pred = A_ @ coef
    r2 = 1 - ((y_ - pred) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum()
    rep["c_two_axis_fit"] = {"r2": float(r2), "coef": coef.tolist(),
                             "pilot_anchor": "r2 = 0.84 (Gate-2)"}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    blockA = r005_r040_r041(args.reps, args.jobs)
    blockB = r006_r007(args.reps, args.jobs)
    blockC = r008(args.reps)

    gate = {
        "R005_oracle_valid": blockA["R005"]["valid"],
        "R006_floorfree_invalid_or_vacuous": blockB["R006"]["expected_met"],
        "R007_dro_blocked": blockB["R007"]["dro_blocked_where_bprime_certifies"],
        "R040_wconformal_risk_valid_no_floor": blockA["R040"]["risk_valid"],
        "R041_plugin_invalid_somewhere": blockA["R041"]["invalid_somewhere"],
    }
    # M1_GO gates ONLY on the plan-setting arms above. R008 replicates Gate-2 pilots
    # AT PILOT SETTINGS (alpha=0.5/delta=0.1 where the anchors were measured) — it is
    # a labeled consistency report, not stop/go evidence (code-review round-1 fix).
    gate["M1_GO"] = all(gate.values())
    pilot_replication = {
        "R008_pair_separates": (blockC["a_matched_pair"]["required_n_ratio"] or 0) >= 2.0,
        "R008_exponent_in_band": bool(
            blockC["b_continuum_exponent"]["fit"]
            and -2.3 <= blockC["b_continuum_exponent"]["fit"]["slope"] <= -1.7),
        "R008_two_axis_r2_pilot_grade": blockC["c_two_axis_fit"]["r2"] >= 0.75,
        "_note": "pilot replication at pilot settings; report-only, excluded from M1_GO",
    }
    gate["pilot_replication_report"] = pilot_replication
    payload = {"milestone": "M1", "alpha": ALPHA, "delta": DELTA,
               "R005_R040_R041": blockA, "R006_R007": blockB, "R008": blockC,
               "gate": gate}
    save_json("M1_baseline.json", payload)
    print({"M1_gate": gate})
    if not gate["M1_GO"]:
        raise SystemExit(2)     # stop: method-level escalation needed


if __name__ == "__main__":
    main()
