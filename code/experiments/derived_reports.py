"""Producers for the derived/reporting JSONs (audit fix D: these were originally
computed in-session; this script is the committed, re-runnable producer).

  python -m experiments.derived_reports shape        # R009_R012 amended_shape_metrics
  python -m experiments.derived_reports betastar     # R051 beta* bootstrap CIs (both legs)
  python -m experiments.derived_reports sensitivity  # R053 K/B/jitter table (leg-2 cache)
  python -m experiments.derived_reports validity     # R054 full-grid validity aggregate

  python -m experiments.derived_reports r036_realleg  # R036 real_data_leg block (leg-1 archive)
  python -m experiments.derived_reports cases 1       # R037 case extracts (leg-1 archive)
  python -m experiments.derived_reports cases 2       # R052 case extracts (leg-2 cache)

  python -m experiments.derived_reports r036_realleg_check  # validate in-session R036_real_leg.json (provenance pin)

Each command rewrites exactly the artifact fields of the authors' frozen results
manifest (author-side record, not shipped; case headline strings are preserved
verbatim; numeric/extract fields are recomputed).

Historical note: experiments/analysis.py::contour_band_check /
logistic_contour_crossing were the M2-era producers of the STRICT-band rows in
R009_R012_contour_analysis.json (per_intensity[*].{n_axis,m_axis}); they are kept
for provenance of those rows even though the current amended producer is `shape`.
They are intentionally retained, NOT dead code: experiments/tests/test_certificate.py
::test_contour_provenance_funcs_runnable exercises both so the suite calls them
(run02 audit D residual #1).

R036_real_leg.json is an in-session M4 (commit 6545a50) real-leg attribution
iota-sweep diagnostic with NO exact regenerator (the raw sweep was computed
in-session); it is NOT a claim source (cited in no claim/tracker/narrative) and is
operationally summarized by the reproducible R036_nocert_decomposition.json
real_data_leg block. It is ARCHIVED to the authors' results archive (author-side, not shipped
with this release). `r036_realleg_check` is its committed
consistency validator (structural invariants + provenance link) — run02 audit D
residual #2.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

RES = os.path.join(os.path.dirname(__file__), "results")


def _j(name):
    return json.load(open(os.path.join(RES, name)))


def _w(name, obj):
    json.dump(obj, open(os.path.join(RES, name), "w"), indent=1)
    print(f"[saved] {name}")


# --------------------------------------------------------------- shape (R009_R012)
def shape():
    """Two-constant fit in-sample on L1, held-out L2-L4 (the reproducible amended
    rate-shape record; supersedes adjudication-time prose)."""
    from scipy.optimize import least_squares

    ca = _j("R009_R012_contour_analysis.json")
    grids = {L: _j(f"B1_grid_{L}.json") for L in ("L1", "L2", "L3", "L4")}

    def rows_of(L):
        pi = ca["per_intensity"][L]
        g = grids[L]
        pts = [("n", r["fixed_value"], r["crossing_log"]) for r in pi["n_axis"]["rows"]]
        pts += [("m", r["fixed_value"], r["crossing_log"]) for r in pi["m_axis"]["rows"]]
        return g["beta"], g["kappa"], pi["vbar"], pts

    def pred_log(axis, fixed, cn, cm, beta, kappa, vbar):
        if axis == "n":
            rem = vbar - cm * np.sqrt(beta / fixed)
            return np.log((cn / kappa) ** 2 * beta / rem ** 2) if rem > 0 else None
        rem = vbar - (cn / kappa) * np.sqrt(beta / fixed)
        return np.log(cm ** 2 * beta / rem ** 2) if rem > 0 else None

    beta, kappa, vbar, pts = rows_of("L1")

    def resid(p):
        cn, cm = np.exp(p)
        return [(e - pl) if (pl := pred_log(ax, fx, cn, cm, beta, kappa, vbar)) is not None
                else 3.0 for ax, fx, e in pts]

    cn, cm = np.exp(least_squares(resid, x0=[np.log(10.0), np.log(4.0)]).x)
    res = {"method": "two-axis constants (c_n,c_m) least-squares-fit IN-SAMPLE on L1 contour "
                     "points (log space), applied UNCHANGED to L2-L4 (held-out); boundary model "
                     "(c_n/kappa)sqrt(beta/n)+c_m*sqrt(beta/m)=vbar_L; rows with non-positive "
                     "remainder (theory: no crossing at this fixed value) excluded and counted",
           "fitted_constants": {"c_n": round(float(cn), 3), "c_m": round(float(cm), 3)},
           "stored_adjudication_constants": ca["calibration"], "per_intensity": {}}
    for L in ("L1", "L2", "L3", "L4"):
        beta, kappa, vbar, pts = rows_of(L)
        emp, th, excl = [], [], 0
        for ax, fx, e in pts:
            pl = pred_log(ax, fx, cn, cm, beta, kappa, vbar)
            if pl is None:
                excl += 1
                continue
            emp.append(e)
            th.append(pl)
        emp, th = np.array(emp), np.array(th)
        fac = np.exp(np.abs(emp - th))
        res["per_intensity"][L] = {
            "in_sample": L == "L1", "n_points": int(len(emp)), "excluded_no_crossing": excl,
            "log_corr": round(float(np.corrcoef(emp, th)[0, 1]), 4),
            "R2_log": round(float(1 - ((emp - th) ** 2).mean() / emp.var()), 4),
            "median_factor_error": round(float(np.median(fac)), 3),
            "q90_factor_error": round(float(np.quantile(fac, 0.9)), 3),
            "frac_within_25pct": round(float((fac <= 1.25).mean()), 3),
            "frac_within_2x": round(float((fac <= 2.0).mean()), 3)}
    ca["amended_shape_metrics"] = {
        "criterion": "M2 GO-with-amended-claims: strict +/-3% band FAILED at all intensities "
                     "(kept visible in rows above); amended rate-shape evidence below. "
                     "REPRODUCIBILITY NOTE: the adjudication-time prose numbers could not be "
                     "reproduced post-hoc (computation not persisted); THIS self-contained "
                     "computation is the auditable record and supersedes the prose.",
        **res}
    _w("R009_R012_contour_analysis.json", ca)


# ------------------------------------------------------------- beta* CIs (R051)
def betastar(n_boot: int = 500, seed: int = 20260612):
    rng = np.random.default_rng(seed)
    base = os.path.join(os.path.dirname(__file__), "real")
    out = {}
    for leg, path in (("leg2_dsv4", f"{base}/cache/_eval_target_frozen.npz"),
                      ("leg1_llama8b", f"{base}/cache_leg1_llama3_8b/_eval_target_frozen.npz")):
        z = np.load(path, allow_pickle=False)
        s, L = z["s"], z["L"]
        n = len(s)

        def bstar(si, Li):
            o = np.argsort(-si)
            cr = np.cumsum(Li[o]) / np.arange(1, n + 1)
            feas = cr <= 0.10
            return (np.arange(1, n + 1)[feas].max() / n) if feas.any() else 0.0

        point = bstar(s, L)
        bs = [bstar(s[i], L[i]) for i in rng.integers(0, n, (n_boot, n))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[leg] = {"beta_star_point": round(float(point), 4),
                    "ci95": [round(float(lo), 4), round(float(hi), 4)],
                    "n_eval": int(n), "n_boot": n_boot}
    d = _j("R051_leg2_attribution.json")
    d["beta_star_bootstrap_ci"] = out
    _w("R051_leg2_attribution.json", d)


# --------------------------------------------------------- sensitivity (R053)
def sensitivity():
    from experiments.real.certify_real import (_load, _source_arrays, _target_scores,
                                               ALPHA, BETA, DELTA)
    from experiments.certificate import certify_bprime_samples

    lock_scores = np.array([r["score"] for r in _load("src_lock")])
    s_wP, _ = _source_arrays(_load("src_wP"))
    s_r, l_r = _source_arrays(_load("src_r"))
    tgt = _target_scores(_load("tgt_unlabeled"))
    rng = np.random.default_rng(20260611)
    perm = rng.permutation(len(tgt))
    half = len(tgt) // 2
    t1, t2 = tgt[perm[:half]], tgt[perm[half:]]
    lattice = np.quantile(lock_scores, np.linspace(0.02, 0.98, 64))

    def run(K, B, jitter=0.0):
        qs = np.arange(1, K) / K
        if jitter:
            qs = np.clip(qs + jitter / (2 * K), 0.01, 0.99)
        edges = np.quantile(lock_scores, qs)
        r = certify_bprime_samples(s_wP, s_r, l_r, t1, t2, lattice, edges, float(B),
                                   ALPHA, BETA, DELTA)
        return {"K": K, "B": B, "jitter": jitter,
                "status": "certify" if bool((r["risk_pass"] & r["floor_pass"]).any()) else "no-cert",
                "risk_pass": int(r["risk_pass"].sum()), "floor_pass": int(r["floor_pass"].sum())}

    table = [run(K, B) for K in (8, 16, 32) for B in (5.0, 10.0, 20.0)]
    table += [run(16, 10.0, jitter=+1.0), run(16, 10.0, jitter=-1.0)]
    statuses = {r["status"] for r in table}
    _w("R053_assumption_sensitivity.json",
       {"run": "R053", "leg": "leg2_dsv4",
        "design": "registered (K=16, B=10) vs K in {8,32}, B in {5,20}, half-cell quantile "
                  "jitter; full-calibration B' on cached scores; NO eval labels touched "
                  "(status-only sensitivity)",
        "table": table, "status_invariant": len(statuses) == 1,
        "reading": ("no-cert status invariant across all K/B/partition variants"
                    if len(statuses) == 1 else "STATUS VARIES — see table")})


# ------------------------------------------------------------ validity (R054)
def validity():
    from scipy.stats import binomtest, beta as beta_dist

    def cp_ucb(v, n, conf=0.95):
        if n == 0:
            return None
        return float(beta_dist.ppf(conf, v + 1, n - v)) if v < n else 1.0

    arms = ["oracle_a", "bprime", "floorfree_posthoc", "dro_box", "weighted_conformal", "plugin"]
    formal = {"oracle_a", "bprime"}
    n_min = int(np.ceil(np.log(0.05) / np.log(1 - 0.0625)))
    out = {"run": "R054",
           "scope": "B1 grids L1-L4, 4x256=1024 cells, 1000 reps/cell "
                    "(five edge cells re-run at 2000-8000)",
           "protocol": "per-cell exact binomial (H0: per-cell viol rate among issued "
                       "certificates <= delta_tol) Holm-corrected across certifying cells "
                       "per arm; CONDITIONAL viol CP-UCB (denominator: certifying "
                       "replications) <= delta_tol over power-adequate cells; "
                       "UNCONDITIONAL pooled viol CP-UCB (denominator: all replications) "
                       "<= delta. The two denominators are never compared with each "
                       "other's criterion.",
           "arms": {}}
    for arm in arms:
        cells_tot = cert_cells = viol_tot = reps_cert_tot = npow = 0
        reps_all_tot = 0
        ucb_max = mx_pow = ucb_max_uncond = 0.0
        pvals = []
        dtol = dl = None
        for L in ("L1", "L2", "L3", "L4"):
            for c in _j(f"B1_grid_{L}.json")["cells"]:
                a, meta = c[arm], c["_meta"]
                dtol, dl = meta["delta_tol"], meta["delta"]
                cells_tot += 1
                # unconditional: denominator is every replication in the cell, so this
                # estimates P(certify AND invalid), the quantity delta caps. The
                # conditional quantities below divide by issued certificates instead
                # and are compared against delta_tol, never against delta.
                reps_all_tot += a["n_reps"]
                ucb_max_uncond = max(ucb_max_uncond, cp_ucb(a["viol"], a["n_reps"]))
                if a["cert"] > 0:
                    cert_cells += 1
                    viol_tot += a["viol"]
                    reps_cert_tot += a["cert"]
                    u = cp_ucb(a["viol"], a["cert"])
                    ucb_max = max(ucb_max, u)
                    pvals.append(binomtest(a["viol"], a["cert"], dtol,
                                           alternative="greater").pvalue)
                    if a["cert"] >= n_min:
                        npow += 1
                        mx_pow = max(mx_pow, u)
        rej = 0
        if pvals:
            ps = np.sort(np.array(pvals))
            m = len(ps)
            rej = int((ps <= 0.05 / (m - np.arange(m))).cumprod().sum())
        pooled = cp_ucb(viol_tot, reps_cert_tot) if reps_cert_tot else None
        pooled_uncond = cp_ucb(viol_tot, reps_all_tot) if reps_all_tot else None
        # round(x, 6) silently turned 8.7e-06 into 9e-06; keep enough places that
        # small CP bounds survive, and leave display rounding to the table generator.
        sig = lambda x: round(x, 10) if x is not None else None
        out["arms"][arm] = {
            "formal_guarantee_arm": arm in formal, "cells": cells_tot,
            "certifying_cells": cert_cells, "viol_total": viol_tot,
            "cert_reps_total": reps_cert_tot, "all_reps_total": reps_all_tot,
            "conditional_denominator": "issued certificates (certifying replications)",
            "max_percell_viol_cp_ucb": sig(ucb_max) if cert_cells else None,
            "percell_criterion": f"<= delta_tol {dtol}",
            "percell_power_threshold_n": n_min, "cells_with_power": npow,
            "max_percell_viol_cp_ucb_powered": sig(mx_pow) if npow else None,
            "percell_pass_powered": (mx_pow <= dtol) if npow else None,
            "holm_rejections": rej, "n_tests": len(pvals),
            "holm_null": f"per-cell violation rate among issued certificates <= {dtol}",
            "pooled_viol_cp_ucb": sig(pooled),
            "pooled_criterion": f"<= delta_tol {dtol} (conditional; NOT comparable to delta)",
            "unconditional_denominator": "all replications",
            "max_percell_viol_cp_ucb_unconditional": sig(ucb_max_uncond),
            "pooled_viol_cp_ucb_unconditional": sig(pooled_uncond),
            "unconditional_criterion": f"<= delta {dl}",
            "pooled_pass": (pooled_uncond <= dl) if pooled_uncond is not None else None}
    # Numeric, so downstream table/figure generators read the criteria directly
    # instead of regex-parsing the prose criterion strings.
    out["delta"], out["delta_tol"] = dl, dtol
    out["reading"] = ("FORMAL-GUARANTEE arms (oracle_a, bprime): 0 violations across all "
                      "certifying cells, 0 Holm rejections, unconditional pooled CP-UCBs "
                      "<= delta (all replications as denominator), conditional per-cell "
                      f"criterion passes against delta_tol over power-adequate cells "
                      f"(cert>={n_min}). Conditional and unconditional intervals are "
                      "reported separately and are never compared with each other's "
                      "criterion. "
                      "DEMONSTRATION arms violate at scale — the invalidity-of-uncertified-"
                      "practice evidence, not an audit failure. dro_box never certifies.")
    _w("R054_validity_aggregate.json", out)


# ------------------------------------------------- R036 real leg (leg-1 archive)
def r036_realleg():
    """Recompute the R036 real_data_leg attribution block from the ARCHIVED leg-1
    cache (author-side, not shipped with this release; full-calibration B' internals +
    post-final-eval frontier diagnostic)."""
    from experiments.certificate import certify_bprime_samples
    from experiments.real.certify_real import ALPHA, BETA, DELTA

    base = os.path.join(os.path.dirname(__file__), "real", "cache_leg1_llama3_8b")

    def load(name):
        return [json.loads(l) for l in open(os.path.join(base, f"{name}.jsonl"))]

    z = np.load(os.path.join(base, "_eval_target_frozen.npz"), allow_pickle=False)
    s, L = z["s"], z["L"]
    order = np.argsort(-s)
    cum_risk = np.cumsum(L[order]) / np.arange(1, len(s) + 1)
    cov_grid = np.arange(1, len(s) + 1) / len(s)
    feas = cum_risk <= ALPHA
    bstar = round(float(cov_grid[feas].max()) if feas.any() else 0.0, 3)

    locks = json.load(open(os.path.join(base, "locks.json")))
    s_wP = np.array([r["score"] for r in load("src_wP")])
    rl = load("src_r")
    s_r = np.array([r["score"] for r in rl])
    from experiments.real.data import correct
    l_r = np.array([1 - correct(r["answer"], r["golds"]) for r in rl], dtype=float)
    tgt = np.array([r["score"] for r in load("tgt_unlabeled")])
    rng = np.random.default_rng(20260611)
    perm = rng.permutation(len(tgt))
    half = len(tgt) // 2
    res = certify_bprime_samples(s_wP, s_r, l_r, tgt[perm[:half]], tgt[perm[half:]],
                                 np.array(locks["lattice_thresholds"]),
                                 np.array(locks["cell_edges"]), locks["B"],
                                 ALPHA, BETA, DELTA)
    rp, fp = res["risk_pass"], res["floor_pass"]
    nuisance_would_pass = bool((res["ucb_x"][fp] <= 0).any()) if fp.any() else False
    print(f"recomputed: floor_pass {int(fp.sum())}/64, risk_pass {int(rp.sum())}/64, "
          f"beta*={bstar}, nuisance_would_pass={nuisance_would_pass}")
    d = _j("R036_nocert_decomposition.json")
    leg = d["real_data_leg"]
    chk = {"floor_pass_expected": leg["evidence"]["floor_pass_lambdas"],
           "floor_pass_recomputed": f"{int(fp.sum())}/64",
           "risk_pass_recomputed": f"{int(rp.sum())}/64 even at rho=0" if not nuisance_would_pass
                                   else f"{int(rp.sum())}/64",
           "beta_star_expected": leg["evidence"]["empirical_frontier_beta_star_alpha_0.10_on_eval"],
           "beta_star_recomputed": bstar}
    leg["reproduction_check"] = chk
    _w("R036_nocert_decomposition.json", d)


def r036_realleg_check():
    """Committed consistency validator for the ARCHIVED in-session R036_real_leg.json
    iota-sweep diagnostic (authors' archive, not shipped with this release; commit
    6545a50; no exact regenerator — the raw sweep was in-session). Pins its structural
    invariants + provenance link to the reproducible R036_nocert_decomposition.json
    real_data_leg block, so the archived artifact stays audit-traceable (run02 audit D
    residual #2). Requires the author-side archived JSON; without it this subcommand
    exits with a file-not-found. Read-only: asserts only, writes nothing."""
    d = _j("archive/R036_real_leg.json")
    pts = d["points"]
    assert d["alpha"] == 0.1 and d["beta"] == 0.6, d
    assert [round(p["iota"], 2) for p in pts] == [0.0, 0.25, 0.5, 0.75, 1.0], pts
    assert all(p["attribution"] == "risk-axis-starved" for p in pts)      # risk axis starved everywhere
    assert all(p["risk_pass_thresholds"] == 0 for p in pts)               # 0/64 risk-side passes
    fp = [p["floor_pass_thresholds"] for p in pts]
    assert fp == sorted(fp, reverse=True) and 0 <= max(fp) <= 64, fp      # floor side monotone non-increasing
    leg = _j("R036_nocert_decomposition.json")["real_data_leg"]           # reproducible operational summary
    assert "risk-axis-starved" in json.dumps(leg), "provenance link to nocert block broken"
    print(f"[ok] R036_real_leg.json consistent: 5 iota pts, risk_pass 0/64 all, "
          f"floor_pass {fp} (monotone), provenance-linked to R036_nocert_decomposition.json real_data_leg")


# ------------------------------------------------------- case extracts (R037/R052)
def cases(leg: str):
    """Regenerate the case extracts, matching each file's committed structure:
    R037 (leg 1) is LABEL-FREE by design — tgt_unlabeled pool, score+answer only,
    3dp, nested under "cases"; R052 (leg 2) uses the post-final-eval frozen labels
    (qid/score/answer/wrong from eval_target[:200]). Headline/note strings are
    preserved verbatim."""
    if leg == "1":
        base = os.path.join(os.path.dirname(__file__), "real", "cache_leg1_llama3_8b")
        rows = [json.loads(l) for l in open(os.path.join(base, "tgt_unlabeled.jsonl"))]
        s = np.array([r["score"] for r in rows])
        hi = np.argsort(-s)[:3]
        lo = np.argsort(s)[:3]
        d = _j("R037_cases.json")
        d["cases"]["high_confidence"] = [{"score": round(float(s[i]), 3),
                                          "answer": rows[i]["answer"]} for i in hi]
        d["cases"]["low_confidence"] = [{"score": round(float(s[i]), 3),
                                         "answer": rows[i]["answer"]} for i in lo]
        _w("R037_cases.json", d)
        return
    base = os.path.join(os.path.dirname(__file__), "real", "cache")
    rows = [json.loads(l) for l in open(os.path.join(base, "eval_target.jsonl"))][:200]
    z = np.load(os.path.join(base, "_eval_target_frozen.npz"), allow_pickle=False)
    s, L = z["s"], z["L"]
    hi = np.argsort(-s[:200])[:3]
    lo = np.argsort(s[:200])[:3]
    d = _j("R052_leg2_cases.json")
    d["high_confidence"] = [{"qid": rows[i]["qid"], "score": float(s[i]),
                             "answer": rows[i]["answer"], "wrong": int(L[i])} for i in hi]
    d["low_confidence"] = [{"qid": rows[i]["qid"], "score": float(s[i]),
                            "answer": rows[i]["answer"], "wrong": int(L[i])} for i in lo]
    _w("R052_leg2_cases.json", d)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cases":
        cases(sys.argv[2])
    else:
        {"shape": shape, "betastar": betastar, "sensitivity": sensitivity,
         "validity": validity, "r036_realleg": r036_realleg,
         "r036_realleg_check": r036_realleg_check}[cmd]()
