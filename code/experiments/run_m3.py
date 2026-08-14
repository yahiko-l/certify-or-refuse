"""M3 — decision ablations (R021-R031, R044): each kills a named anti-claim.

Subcommands:
  axes       R021/R022 — n-only & m-only scaling >= 2 decades; exponent CI in [-0.6,-0.4]
  modelcmp   R023 — additive vs n-only/m-only/max/floor-free, 70/30 CV-RMSE + AIC/BIC
  beta0      R024 — beta->0 axis-vanishing (theorem-aligned families)
  ksweep     R025/R026 — K in {4,8,16,32,64}: paired tests B'<=A + deficit~rho regression
  lattice    R027/R028 — coarse->fine lattice; zero-margin required-n -> inf signature
  misspec    R029/R030/R031 — off-class eps grids (tilt / gradient / tv)
  nontheorem R044 — non-theorem-aligned family beta->0 + scaling leg
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .analysis import loglog_exponent, model_comparison, surface_r2_heldout
from .generators import (family_continuum, family_kcell, family_nontheorem,
                         perturb_offclass, cells_from_source_quantiles, lattice_prefix)
from .validity import run_cell, cp_ucb, holm_exact, DELTA_TOL_FACTOR
from .runner_util import save_json, pmap, log_grid, RESULTS_DIR

ALPHA, DELTA = 0.2, 0.05
SLACK_FINE = np.geomspace(0.012, 0.30, 16)


def offset_power_fit(x, y, n_boot: int = 400, seed: int = 20260611):
    """Fit s* = A·x^p + c (offset c = the fixed axis's additive-law term).
    Returns p with percentile-bootstrap CI (residual resampling, refit) and a
    curvature check on log(y - c_hat) vs log x."""
    from scipy.optimize import least_squares
    lx = np.log(x)

    def fit_once(yv):
        def resid(prm):
            A, p, c = prm
            return np.log(np.maximum(A * x ** p + c, 1e-12)) - np.log(yv)
        sol = least_squares(resid, x0=[float(yv.max()), -0.5,
                                       max(float(yv.min()) * 0.5, 1e-6)],
                            bounds=([1e-12, -2.0, 0.0],
                                    [np.inf, -0.01, float(yv.min())]))
        return sol.x

    A0, p0, c0 = fit_once(y)
    res = np.log(np.maximum(A0 * x ** p0 + c0, 1e-12)) - np.log(y)
    rng = np.random.default_rng(seed)
    ps = []
    for _ in range(n_boot):
        yb = np.exp(np.log(y) + rng.choice(res, len(res), replace=True))
        try:
            ps.append(fit_once(yb)[1])
        except Exception:
            continue
    ci = (float(np.percentile(ps, 2.5)), float(np.percentile(ps, 97.5)))
    # curvature on offset-subtracted values
    ys = np.maximum(y - c0, 1e-12)
    X2 = np.column_stack([np.ones_like(lx), lx, lx ** 2])
    c2, *_ = np.linalg.lstsq(X2, np.log(ys), rcond=None)
    r2 = np.log(ys) - X2 @ c2
    dof = max(len(lx) - 3, 1)
    cov2 = ((r2 ** 2).sum() / dof) * np.linalg.inv(X2.T @ X2)
    q_se = float(np.sqrt(cov2[2, 2]))
    curv_ci = (float(c2[2] - 1.96 * q_se), float(c2[2] + 1.96 * q_se))
    return {"slope": float(p0), "offset": float(c0), "amplitude": float(A0),
            "ci95": ci, "curvature_ci": curv_ci,
            "curvature_contains_0": curv_ci[0] <= 0 <= curv_ci[1],
            "n_boot_ok": len(ps)}


def _sstar_job(spec):
    """Certifiable-slack threshold s*(n, m) on the fine slack grid (interpolated at
    cert-freq tau=0.5), for both certificate arms."""
    fam, n, m, beta_floor_ref, reps, key = spec
    world = family_continuum(2) if fam == "cont" else family_nontheorem()
    bstar = world.meta["beta_star"]
    freqs = {"oracle_a": [], "bprime": []}
    for s in SLACK_FINE:
        r = run_cell(world, n=int(n), m=int(m), alpha=ALPHA, beta=bstar - s,
                     delta=DELTA, n_reps=reps, cell_key=(key, float(s), int(n), int(m)),
                     arms=("oracle_a", "bprime"))
        for a in freqs:
            freqs[a].append(r[a]["cert_freq"])
    out = {"n": int(n), "m": int(m)}
    for a, fr in freqs.items():
        fr = np.array(fr)
        idx = np.flatnonzero(fr >= 0.5)
        if len(idx) == 0:
            out[f"s_star_{a}"] = np.inf
        else:
            i = idx[0]
            if i == 0:
                out[f"s_star_{a}"] = float(SLACK_FINE[0])
            else:   # log-linear interpolation between the bracketing slacks
                s0, s1 = SLACK_FINE[i - 1], SLACK_FINE[i]
                f0, f1 = fr[i - 1], fr[i]
                t = (0.5 - f0) / max(f1 - f0, 1e-9)
                out[f"s_star_{a}"] = float(np.exp(np.log(s0) + t * (np.log(s1) - np.log(s0))))
    return out


def cmd_axes(args):
    """R021 (fix m, scale n over >=2 decades) and R022 (fix n, scale m)."""
    n_grid = log_grid(16384, 4194304, 8)     # 2.4 decades, reaches B' regime
    m_grid = log_grid(16384, 4194304, 8)
    # fixed axis SATURATED (16M / 4M): an unsaturated fixed axis adds an additive
    # constant that bends the pure power law (bite-run lesson, M3 round-1 finding)
    specs_n = [("cont", n, 16000000, None, args.reps, "R021") for n in n_grid]
    specs_m = [("cont", 4194304, m, None, args.reps, "R022") for m in m_grid]
    rows_n = pmap(_sstar_job, specs_n, n_jobs=args.jobs, desc="R021 n-scaling")
    rows_m = pmap(_sstar_job, specs_m, n_jobs=args.jobs, desc="R022 m-scaling")
    out = {"run": "R021_R022", "slack_grid": SLACK_FINE.tolist(),
           "n_rows": rows_n, "m_rows": rows_m, "fits": {},
           "fit_form": "s* = A x^p + c (offset = the OTHER axis's law term; a pure "
                       "power fit tests a strawman of the additive law — M3 round-2 "
                       "registered amendment)"}
    for arm in ("oracle_a", "bprime"):
        x = np.array([r["n"] for r in rows_n], float)
        y = np.array([r[f"s_star_{arm}"] for r in rows_n], float)
        fin = np.isfinite(y)
        out["fits"][f"n_axis_{arm}"] = offset_power_fit(x[fin], y[fin]) if fin.sum() >= 5 else None
        x = np.array([r["m"] for r in rows_m], float)
        y = np.array([r[f"s_star_{arm}"] for r in rows_m], float)
        fin = np.isfinite(y)
        out["fits"][f"m_axis_{arm}"] = offset_power_fit(x[fin], y[fin]) if fin.sum() >= 5 else None
    out["success"] = {
        k: bool(f and -0.6 <= f["ci95"][0] and f["ci95"][1] <= -0.4
                and f["curvature_contains_0"])
        for k, f in out["fits"].items() if f}
    save_json("R021_R022_axis_scaling.json", out)


def cmd_modelcmp(args):
    """R023 — functional-form comparison on the pooled s* surface (from R021/R022 rows
    + a 2-D refill grid), 70/30 cells, CV-RMSE + AIC/BIC."""
    world = family_continuum(2)
    kappa = world.meta["kappa"]
    rows = []
    p1 = os.path.join(RESULTS_DIR, "R021_R022_axis_scaling.json")
    if os.path.exists(p1):
        with open(p1) as f:
            ax = json.load(f)
        rows += [(r["n"], r["m"], r["s_star_bprime"]) for r in ax["n_rows"] + ax["m_rows"]]
    grid = log_grid(65536, 2097152, 4)
    specs = [("cont", n, m, None, args.reps, "R023") for n in grid for m in grid]
    rows2 = pmap(_sstar_job, specs, n_jobs=args.jobs, desc="R023 surface refill")
    rows += [(r["n"], r["m"], r["s_star_bprime"]) for r in rows2]
    betas = [world.meta["beta_star"] - 0.06] * len(rows)
    # dedicated beta-varying probes (maximally discriminating vs floor-free, which
    # predicts NO beta dependence at fixed (n,m))
    from .run_m2_b1 import rel_slack_grid
    for beta_v in (0.6, 0.3, 0.15):
        for nm in (1048576, 4194304):
            sth = None
            for s in rel_slack_grid(beta_v):
                wv = family_continuum(2, beta_star_target=beta_v + float(s))
                rv = run_cell(wv, n=nm, m=nm, alpha=ALPHA, beta=beta_v, delta=DELTA,
                              n_reps=args.reps, cell_key=("R023b", beta_v, nm, float(s)),
                              arms=("bprime",))
                if rv["bprime"]["cert_freq"] >= 0.5:
                    sth = float(wv.meta["beta_star"]) - beta_v
                    break
            if sth:
                rows.append((nm, nm, sth))
                betas.append(beta_v)
    # pool the beta-LADDER thresholds (R013): WITHOUT beta variation the additive
    # and floor-free forms are reparameterizations of each other (M3 round-1 finding)
    lad_p = os.path.join(RESULTS_DIR, "R013_beta_ladder.json")
    if os.path.exists(lad_p):
        lad = json.load(open(lad_p))
        for rec in lad["thresholds"].values():
            if rec.get("s_star_bprime"):
                rows.append((rec["n"], rec["m"], rec["s_star_bprime"]))
                betas.append(rec["beta"])      # the law's beta = the FLOOR
    n_a = np.array([r[0] for r in rows], float)
    m_a = np.array([r[1] for r in rows], float)
    y = np.array([r[2] for r in rows], float)
    fin = np.isfinite(y)
    beta_arr = np.array(betas, float)[fin]
    cmp_ = model_comparison(n_a[fin], m_a[fin], beta_arr, kappa, y[fin])
    r2 = surface_r2_heldout(n_a[fin], m_a[fin], beta_arr, kappa, y[fin])
    save_json("R023_model_comparison.json",
              {"run": "R023", "n_cells": int(fin.sum()), "comparison": cmp_,
               "heldout_r2_additive": r2,
               "pilot_anchor_r2": "0.84 (Gate-2; B1(b) bar)",
               "additive_wins": cmp_["_verdict"]["additive_wins_all_nonoverlap"],
               "floor_free_loses": bool(cmp_["_verdict"]["per_form_nonoverlap"]
                                        .get("floor_free", False))})


def cmd_beta0(args):
    """R024 — beta->0 axis vanishing: s*(beta) at theorem-aligned families. Each
    (beta, s) probe rebuilds the world with frontier beta+s (beta_star_target) so the
    slack is real (code-review round-1 CRITICAL fix); slack grid is beta-relative
    (s = beta*geomspace(0.05, 0.4, 8), the c0 = 0.4 regime guard)."""
    from .run_m2_b1 import BETA_LADDER, rel_slack_grid, SLACK_DEV_TOL
    rows = []
    for beta in BETA_LADDER:
        probe, dropped = [], []
        for s in rel_slack_grid(beta):
            world = family_continuum(2, beta_star_target=beta + float(s))
            actual_s = float(world.meta["beta_star"]) - beta
            if not (actual_s > 0 and abs(actual_s / float(s) - 1.0) <= SLACK_DEV_TOL):
                dropped.append({"requested_s": float(s), "actual_s": actual_s})
                continue
            r = run_cell(world, n=262144, m=262144, alpha=ALPHA, beta=beta,
                         delta=DELTA, n_reps=args.reps,
                         cell_key=("R024", float(beta), float(s)),
                         arms=("oracle_a", "bprime"))
            probe.append((actual_s, r["bprime"]["cert_freq"], r["oracle_a"]["cert_freq"]))
        probe.sort(key=lambda t: t[0])
        rec = {"beta": beta, "probes": probe, "dropped_probes": dropped}
        rec["s_star_bprime"] = next((s for (s, fb, fo) in probe if fb >= 0.5), None)
        rec["s_star_oracle_a"] = next((s for (s, fb, fo) in probe if fo >= 0.5), None)
        rows.append(rec)
    xs = np.array([r["beta"] for r in rows if r["s_star_bprime"]], float)
    ys = np.array([r["s_star_bprime"] for r in rows if r["s_star_bprime"]], float)
    fit = loglog_exponent(xs, ys) if len(xs) >= 4 else None
    order = np.argsort(xs)
    save_json("R024_beta_vanishing.json",
              {"run": "R024", "rows": rows, "fit_s_vs_beta": fit,
               "vanishes": bool(len(ys) >= 3 and ys[order][0] <= 0.5 * ys[order][-1]),
               "note": "theorem-aligned families; non-aligned leg = R044"})


def _ksweep_job(spec):
    """Per-(K, n) job retaining PER-REPEAT paired acceptances (code-review round-1
    fix: aggregate signs were powerless). Pairing = same repeat index (shared
    per-(cell, repeat) stream): both arms certify on the same world snapshot."""
    K, ntot, reps = spec
    from .arms import arm_oracle_a, arm_bprime
    from .generators import rng_for, lattice_prefix, equal_blocks
    world = family_kcell(loc_level=2.0)
    beta = world.meta["beta_star"] - 0.12
    cell_of_bin = equal_blocks(world.n_bins, K)
    lattice_bins = lattice_prefix(world, 50)
    cov = world.coverage(); risk = world.risk()
    viol_lambda = (risk[lattice_bins] > ALPHA) | (cov[lattice_bins] < beta)
    n_w, n_r = ntot // 2, ntot - ntot // 2
    m_w, m_f = ntot // 2, ntot - ntot // 2
    acc_a = np.zeros(reps); acc_b = np.zeros(reps)
    cert_a = np.zeros(reps, bool); cert_b = np.zeros(reps, bool)
    viol_b = 0; rho_sum = 0.0; rho_cnt = 0
    for i in range(reps):
        rng = rng_for(("R025", K, ntot), i)
        cwP, cr, lr, cwQ, cf = world.sample_counts(rng, n_w, n_r, m_w, m_f)
        c_src, l_src = world.sample_source(rng, ntot)
        c_tgt = world.sample_target(rng, ntot)
        ra = arm_oracle_a(c_src, l_src, c_tgt, world.w, ALPHA, beta, DELTA, lattice_bins)
        rb = arm_bprime(cwP, cr, lr, cwQ, cf, world.B_class, ALPHA, beta, DELTA,
                        cell_of_bin, lattice_bins)
        ja, jb = ra["chosen"][0], rb["chosen"][0]
        cert_a[i] = ja >= 0; cert_b[i] = jb >= 0
        acc_a[i] = cov[lattice_bins[ja]] if ja >= 0 else 0.0
        acc_b[i] = cov[lattice_bins[jb]] if jb >= 0 else 0.0
        if jb >= 0:
            viol_b += bool(viol_lambda[jb])
            rho_sum += float(rb["rho"][0, jb]); rho_cnt += 1
    from .validity import cp_ucb as _cp
    return {"K": K, "n_total": ntot, "reps": reps,
            "acc_a": float(acc_a[cert_a].mean()) if cert_a.any() else None,
            "acc_b": float(acc_b[cert_b].mean()) if cert_b.any() else None,
            "cert_a": float(cert_a.mean()), "cert_b": float(cert_b.mean()),
            "viol_b_ucb": _cp(viol_b, reps),
            "mean_rho": (rho_sum / rho_cnt) if rho_cnt else None,
            "exceed_count": int((acc_b > acc_a + 1e-12).sum()),
            "nonzero_pairs": int(((acc_b > acc_a + 1e-12)
                                  | (acc_a > acc_b + 1e-12)).sum())}


def cmd_ksweep(args):
    """R025/R026 — K-sweep: rho growth + acceptance degradation; per-K one-sided
    PAIRED sign test on per-repeat acceptances (B' never significantly exceeds A,
    Holm 5%); deficit (A - B') ~ rho regression R^2 >= 0.8."""
    Ks = [4, 8, 16, 32, 64]
    ns = [262144, 1048576, 4194304]
    specs = [(K, n, args.reps) for K in Ks for n in ns]
    rows = pmap(_ksweep_job, specs, n_jobs=args.jobs, desc="R025/R026 K-sweep")
    # per-K exact one-sided sign test over ALL per-repeat pairs (H0: P(B'>A) <= 1/2
    # among non-tied pairs); Holm across K
    exceed_counts, n_pairs = [], []
    for K in Ks:
        sub = [r for r in rows if r["K"] == K]
        exceed_counts.append(sum(r["exceed_count"] for r in sub))
        n_pairs.append(sum(r["nonzero_pairs"] for r in sub))
    any_rej, rej, pvals = holm_exact(exceed_counts, n_pairs, 0.5)   # H0: P(exceed)<=1/2
    # deficit regression
    x = np.array([r["mean_rho"] for r in rows if r["mean_rho"] is not None
                  and r["acc_a"] and r["acc_b"]], float)
    y = np.array([r["acc_a"] - r["acc_b"] for r in rows if r["mean_rho"] is not None
                  and r["acc_a"] and r["acc_b"]], float)
    if len(x) >= 4:
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        r2 = float(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    else:
        coef, r2 = None, None
    save_json("R025_R026_ksweep.json",
              {"run": "R025_R026", "rows": rows,
               "paired_test": {"Ks": Ks, "exceed_counts": exceed_counts,
                               "n_pairs": n_pairs, "holm_any_rejection": bool(any_rej),
                               "bprime_never_exceeds_a": not bool(any_rej)},
               "deficit_regression": {"coef": None if coef is None else coef.tolist(),
                                      "r2": r2, "pass_0p8": bool(r2 and r2 >= 0.8)},
               "rho_growth": [[r["K"], r["n_total"], r["mean_rho"]] for r in rows]})


def cmd_lattice(args):
    """R027/R028 — lattice coarse->fine on the continuum family: required-n vs lattice
    margin; zero-margin floor values show the required-n -> inf signature, refinement
    restores the law."""
    world = family_continuum(2)
    bstar = world.meta["beta_star"]
    lat_sizes = [10, 25, 50, 100, 200]
    # floor chosen INSIDE a coarse-lattice gap: with |Λ|=10 source-quantile prefixes,
    # pick beta = midpoint coverage between two adjacent coarse lattice coverages
    cov = world.coverage()
    coarse_bins = lattice_prefix(world, 10)
    cov_coarse = cov[coarse_bins]
    below = cov_coarse[cov_coarse < bstar]
    c_J = float(below.max())          # last coarse point under the frontier
    # beta INSIDE (c_J, beta*): coarse lattice has NO point with coverage >= beta
    # that is risk-feasible -> required-n = inf signature (true LR_margin failure).
    # Quarter-gap placement keeps the FINE-lattice margin at 0.75*gap (the midpoint
    # left B' borderline at the bracket edge — M3 round-2 fix)
    beta_zero_margin = float(c_J + 0.25 * (bstar - c_J))
    beta_regular = bstar - 0.06
    out_rows = []
    for nl in lat_sizes:
        for tag, beta in (("regular", beta_regular), ("zero_margin", beta_zero_margin)):
            need = None
            for n in log_grid(512, 8_000_000, 18):
                r = run_cell(world, n=int(n), m=4000000, alpha=ALPHA, beta=beta,
                             delta=DELTA, n_reps=args.reps,
                             cell_key=("R027", nl, tag, int(n)),
                             arms=("bprime",), n_lattice=nl)
                if r["bprime"]["cert_freq"] >= 0.8:
                    need = int(n)
                    break
            out_rows.append({"n_lattice": nl, "beta_kind": tag, "beta": beta,
                             "required_n": need})
            print(f"|Λ|={nl:>4} {tag:>11}: required_n = {need}")
    zm = {r["n_lattice"]: r["required_n"] for r in out_rows
          if r["beta_kind"] == "zero_margin"}
    reg = {r["n_lattice"]: r["required_n"] for r in out_rows
           if r["beta_kind"] == "regular"}
    signature = (zm.get(10) is None) and (zm.get(200) is not None)
    save_json("R027_R028_lattice.json",
              {"run": "R027_R028", "rows": out_rows,
               "beta_zero_margin": beta_zero_margin, "beta_regular": beta_regular,
               "lr_margin_signature": bool(signature),
               "refinement_restores": bool(zm.get(200) is not None),
               "note": "pilot-observed LR_margin failure reproduction; guidance feeds the M0 lattice lock (already registered)"})


def _misspec_job(spec):
    """B4.4 cell discipline (round-2 review fix): ONE explicit certificate partition —
    the declared cells are set to equal_blocks(nbins, 16) on the base world (still
    on-class: kcell w is constant on 4 blocks, hence on every refinement), the
    perturbation is defined RELATIVE TO that same partition, and run_cell uses the
    declared cells with no override."""
    mode, eps, ntot, reps = spec
    from .generators import equal_blocks
    base = family_kcell(loc_level=2.0)
    base.cells = equal_blocks(base.n_bins, 16)
    world = perturb_offclass(base, base.cells, eps, mode)
    beta = world.meta["beta_star"] - 0.12
    r = run_cell(world, n=ntot, m=ntot, alpha=ALPHA, beta=beta, delta=DELTA,
                 n_reps=reps, cell_key=("R029", mode, float(eps)),
                 arms=("bprime",))
    t = r["bprime"]
    return {"mode": mode, "eps": eps, "viol_freq": t["viol_freq"],
            "viol_cp_ucb": t["viol_cp_ucb"], "cert_freq": t["cert_freq"]}


def cmd_misspec(args):
    """R029/R030/R031 — misspecified-K sensitivity: violation-vs-eps curves; first
    exceedance eps0 of delta_tol; graceful = no adjacent jump > 2*delta. Necessity of
    the nuisance axis stays OPEN (reported as such)."""
    eps_grid = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    out = {}
    for mode, run_id in (("tilt", "R029"), ("gradient", "R030"), ("tv", "R031")):
        rows = pmap(_misspec_job, [(mode, e, 4194304, args.reps) for e in eps_grid],
                    n_jobs=args.jobs, desc=f"{run_id} {mode}")
        rows = sorted(rows, key=lambda r: r["eps"])
        eps0 = next((r["eps"] for r in rows
                     if r["viol_cp_ucb"] > DELTA_TOL_FACTOR * DELTA), None)
        jumps = [abs(rows[i + 1]["viol_freq"] - rows[i]["viol_freq"])
                 for i in range(len(rows) - 1)]
        out[mode] = {"run": run_id, "rows": rows, "eps0_first_exceedance": eps0,
                     "max_adjacent_jump": float(max(jumps)) if jumps else 0.0,
                     "no_cliff": bool(max(jumps) < 2 * DELTA) if jumps else True}
    save_json("R029_R031_misspec.json",
              {"run": "R029_R030_R031", "modes": out,
               "nuisance_necessity": "OPEN — this ablation measures sensitivity, not necessity"})


def cmd_nontheorem(args):
    """R044 — non-theorem-aligned family: beta->0 + n-axis scaling; exponents may fall
    outside the engineered class — reported honestly (Honesty #6)."""
    world = family_nontheorem()
    bstar = world.meta["beta_star"]
    n_grid = log_grid(16384, 4194304, 8)
    rows = []
    for n in n_grid:
        rec = _sstar_job(("nont", n, 16000000, None, args.reps, "R044"))
        rows.append(rec)
    x = np.array([r["n"] for r in rows], float)
    y = np.array([r["s_star_bprime"] for r in rows], float)
    fin = np.isfinite(y)
    fit = loglog_exponent(x[fin], y[fin]) if fin.sum() >= 4 else None
    # beta-ladder vanishing on this family: per-(beta, s) worlds with frontier at
    # beta+s (same CRITICAL fix as R013/R024)
    from .run_m2_b1 import BETA_LADDER, rel_slack_grid
    lrows = []
    for beta in BETA_LADDER:
        sth = None
        for s in rel_slack_grid(beta):
            wld = family_nontheorem(beta_star_target=beta + float(s))
            r = run_cell(wld, n=262144, m=262144, alpha=ALPHA, beta=beta, delta=DELTA,
                         n_reps=args.reps, cell_key=("R044b", float(beta), float(s)),
                         arms=("bprime",))
            if r["bprime"]["cert_freq"] >= 0.5:
                sth = float(s)
                break
        lrows.append({"beta": beta, "s_star": sth})
    # DIRECTIONAL vanishing check (round-2 review fix): s* at the smallest beta must
    # sit well below s* at the largest beta — shrinkage AS beta decreases.
    ok = sorted((r for r in lrows if r["s_star"]), key=lambda r: r["beta"])
    vanishes = bool(len(ok) >= 3 and ok[0]["s_star"] <= 0.6 * ok[-1]["s_star"])
    save_json("R044_nontheorem.json",
              {"run": "R044", "n_axis_rows": rows, "n_axis_fit": fit,
               "beta_rows": lrows, "beta_vanishing_extends": vanishes,
               "note": "if not extending: limitation stated in paper (Honesty #6)"})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    # plan §B4: MC >= 500 repeats per point everywhere (code-review round-1 fix)
    for name, default_reps in (("axes", 500), ("modelcmp", 500), ("beta0", 500),
                               ("ksweep", 600), ("lattice", 500), ("misspec", 800),
                               ("nontheorem", 500)):
        p = sub.add_parser(name)
        p.add_argument("--reps", type=int, default=default_reps)
        p.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()
    {"axes": cmd_axes, "modelcmp": cmd_modelcmp, "beta0": cmd_beta0,
     "ksweep": cmd_ksweep, "lattice": cmd_lattice, "misspec": cmd_misspec,
     "nontheorem": cmd_nontheorem}[args.cmd](args)


if __name__ == "__main__":
    main()
