"""B1 contour-protocol analysis (R009-R012 criterion (a)) + no-cert rollup.

Registered operationalization (documented here, applied identically to all grids):
- Formal band test arm = oracle-A (Model A is the law's matching-upper reference; the
  B′ contour is overlaid descriptively — its third nuisance term is quantified by
  R015/R026, not by this two-term test).
- Per m-row (m fixed, vary n): tau=0.5 crossing via the monotone logistic estimator;
  per n-column likewise for the m-axis limb. Simultaneous 95% band via Bonferroni
  across tested lines (analysis.contour_band_check).
- Theory curves: n*(intensity) = c_n * Vbar(intensity) * L / (kappa s)^2 with
  Vbar = E_P[(w S (L-alpha))^2] at the working prefix (world parameter — the variance
  proxy U2 carries; this is what makes the boundary intensity-dependent);
  m* = c_m * beta * L / s^2 (weight-free floor: intensity-INdependent — itself a
  registered prediction). Constants (c_n, c_m) calibrated on the L1 grid only and
  FROZEN for L2-L4 (out-of-sample shape+scaling test); L1 is therefore in-sample for
  scale and out-of-sample for shape.
- Success (plan B1(a)): theory inside the simultaneous band on >= 90% of tested lines
  at EVERY intensity. Criterion (b) (held-out r^2 >= 0.84) is carried by the
  R023 surface CV on s*(n,m); criterion (c) by R042.

Usage: python -m experiments.analyze_b1
"""
from __future__ import annotations

import json
import os

import numpy as np

from joblib import Parallel, delayed
from scipy import optimize

from .generators import family_continuum, lattice_prefix
from .runner_util import save_json, RESULTS_DIR

ALPHA, DELTA = 0.2, 0.05
TAU = 0.5
N_BOOT = 200


def _fit_crossing(xs, ks, ns):
    """Fast monotone-logistic tau-crossing fit (no internal bootstrap)."""
    def nll(prm):
        a, b = abs(prm[0]), prm[1]
        p = np.clip(1.0 / (1.0 + np.exp(-a * (xs - b))), 1e-9, 1 - 1e-9)
        return -(ks * np.log(p) + (ns - ks) * np.log(1 - p)).sum()
    x0 = np.array([2.0, xs[np.argmin(abs(ks / np.maximum(ns, 1) - TAU))]])
    fit = optimize.minimize(nll, x0, method="Nelder-Mead",
                            options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 600})
    return float(fit.x[1])


def _vbar(world, beta):
    """Variance proxy at the working prefix: E_P[(w S (L-alpha))^2] - mean^2."""
    cov = world.coverage()
    j = int(np.searchsorted(cov, beta))
    sel = slice(0, j + 1)
    w, p, eta = world.w[sel], world.p[sel], world.eta[sel]
    ex2 = float((p * w ** 2 * (eta * (1 - ALPHA) ** 2 + (1 - eta) * ALPHA ** 2)).sum())
    mean = float((p * w * (eta - ALPHA)).sum())
    return max(ex2 - mean ** 2, 1e-9)


def _crossings(grid, arm, axis):
    """Crossing point per line: axis='n' -> per m-row crossing in log n;
    axis='m' -> per n-column crossing in log m. Only lines that actually cross
    (min freq < tau < max freq) are testable."""
    ns = np.array(grid["ns"]); ms = np.array(grid["ms"])
    cells = {(c["_meta"]["n"], c["_meta"]["m"]): c for c in grid["cells"]}
    lines = []
    outer, inner = (ms, ns) if axis == "n" else (ns, ms)
    for fix in outer:
        xs, fr, nr = [], [], []
        for v in inner:
            key = (v, fix) if axis == "n" else (fix, v)
            if key in cells:
                t = cells[key][arm]
                xs.append(np.log(v)); fr.append(t["cert_freq"]); nr.append(t["n_reps"])
        xs, fr, nr = map(np.array, (xs, fr, nr))
        if len(xs) >= 5 and fr.min() < TAU < fr.max():
            lines.append({"fixed": int(fix), "xs": xs, "freqs": fr, "ns": nr})
    return lines


def _theory_n_of_m(m, vbar, beta, kappa, s, L, a, b):
    """Additive-law boundary: s = a·sqrt(Vbar L / n*)/kappa + b·sqrt(beta L / m)
    => n*(m). Returns nan where the floor term exhausts the slack (untestable row)."""
    rem = s - b * np.sqrt(beta * L / m)
    if rem <= 0:
        return np.nan
    return vbar * L * a * a / (kappa * rem) ** 2


def _theory_m_of_n(n, vbar, beta, kappa, s, L, a, b):
    rem = s - a * np.sqrt(vbar * L / n) / kappa
    if rem <= 0:
        return np.nan
    return beta * L * b * b / rem ** 2


def _line_band(ln, theory_log, level):
    b_hat = _fit_crossing(ln["xs"], np.round(ln["freqs"] * ln["ns"]).astype(int),
                          ln["ns"])
    rng = np.random.default_rng(20260611 + ln["fixed"])
    ks = np.round(ln["freqs"] * ln["ns"]).astype(int)
    p_hat = np.clip(ks / np.maximum(ln["ns"], 1), 1e-9, 1 - 1e-9)
    boots = []
    for _ in range(N_BOOT):
        kb = rng.binomial(ln["ns"], p_hat)
        try:
            boots.append(_fit_crossing(ln["xs"], kb, ln["ns"]))
        except Exception:
            continue
    lo, hi = np.percentile(boots, [level, 100 - level])
    ok = bool(lo <= theory_log <= hi)
    return {"fixed_value": ln["fixed"], "crossing_log": float(b_hat),
            "band_log": (float(lo), float(hi)),
            "theory_log": float(theory_log), "inside": ok}


def _band_test(lines, theory_log, bonf, n_jobs=32):
    """Bonferroni-simultaneous parametric-bootstrap band per line; theory inside?"""
    level = 100.0 * (0.05 / bonf) / 2.0
    rows = Parallel(n_jobs=min(n_jobs, max(len(lines), 1)))(
        delayed(_line_band)(ln, theory_log, level) for ln in lines)
    inside = sum(r["inside"] for r in rows)
    frac = inside / max(len(lines), 1)
    return rows, frac


def main():
    grids = {}
    for i in (1, 2, 3, 4):
        p = os.path.join(RESULTS_DIR, f"B1_grid_L{i}.json")
        if not os.path.exists(p):
            raise SystemExit(f"missing B1_grid_L{i}.json")
        grids[i] = json.load(open(p))
    s = grids[1]["s_work"]
    kappa = grids[1]["kappa"]
    L_const = np.log(grids[1]["n_lattice"] / DELTA)

    # ---- theory curve = THEOREM 1's additive two-axis law (not the asymptotic
    # quadrant cartoon — the empirical boundary is the law's own smooth L-curve):
    #   s = a·sqrt(Vbar·L/n*)/kappa + b·sqrt(beta·L/m*)
    # (a, b) least-squares-calibrated on the L1 oracle crossings ONLY and FROZEN;
    # L2-L4 are out-of-sample in both shape and Vbar-driven scaling.
    out = {"arm_formal": "oracle_a", "s_work": s,
           "theory_model": "additive two-axis law, (a,b) calibrated at L1, "
                           "Vbar-scaled out-of-sample at L2-L4",
           "per_intensity": {}}
    from scipy.optimize import least_squares
    calib = {}
    for i in (1, 2, 3, 4):
        g = grids[i]
        world = family_continuum(i)
        beta = g["beta"]
        vbar = _vbar(world, beta)
        n_lines = _crossings(g, "oracle_a", "n")
        m_lines = _crossings(g, "oracle_a", "m")
        if i == 1:
            ms_l1 = np.array([l["fixed"] for l in n_lines], float)
            cross_l1 = np.array([np.exp(_fit_crossing(
                l["xs"], np.round(l["freqs"] * l["ns"]).astype(int), l["ns"]))
                for l in n_lines])
            ns_l1 = np.array([l["fixed"] for l in m_lines], float)
            crossm_l1 = np.array([np.exp(_fit_crossing(
                l["xs"], np.round(l["freqs"] * l["ns"]).astype(int), l["ns"]))
                for l in m_lines])

            def resid(prm):
                a, b = np.abs(prm)
                rn = [np.log(_theory_n_of_m(m, vbar, beta, kappa, s, L_const, a, b))
                      - np.log(c) for m, c in zip(ms_l1, cross_l1)]
                rm = [np.log(_theory_m_of_n(n, vbar, beta, kappa, s, L_const, a, b))
                      - np.log(c) for n, c in zip(ns_l1, crossm_l1)]
                r = np.array(rn + rm)
                return np.where(np.isfinite(r), r, 3.0)
            sol = least_squares(resid, x0=np.array([1.0, 1.0]),
                                bounds=([0.01, 0.01], [100, 100]))
            calib = {"a": float(abs(sol.x[0])), "b": float(abs(sol.x[1])),
                     "vbar_L1": vbar}
        a_c, b_c = calib["a"], calib["b"]
        n_lines_t, n_theories = [], []
        for l in n_lines:
            th = _theory_n_of_m(l["fixed"], vbar, beta, kappa, s, L_const, a_c, b_c)
            if np.isfinite(th):
                n_lines_t.append(l); n_theories.append(np.log(th))
        m_lines_t, m_theories = [], []
        for l in m_lines:
            th = _theory_m_of_n(l["fixed"], vbar, beta, kappa, s, L_const, a_c, b_c)
            if np.isfinite(th):
                m_lines_t.append(l); m_theories.append(np.log(th))
        bonf = max(len(n_lines_t) + len(m_lines_t), 1)
        level = 100.0 * (0.05 / bonf) / 2.0
        n_rows = Parallel(n_jobs=32)(delayed(_line_band)(ln, th, level)
                                     for ln, th in zip(n_lines_t, n_theories))
        m_rows = Parallel(n_jobs=32)(delayed(_line_band)(ln, th, level)
                                     for ln, th in zip(m_lines_t, m_theories))
        n_frac = (sum(r["inside"] for r in n_rows) / max(len(n_rows), 1))
        m_frac = (sum(r["inside"] for r in m_rows) / max(len(m_rows), 1))
        # descriptive B' overlay: median crossings
        bp_n = _crossings(g, "bprime", "n")
        bp_cross = [float(np.exp(_fit_crossing(l["xs"], np.round(l["freqs"] * l["ns"]).astype(int), l["ns"])))
                    for l in bp_n]
        out["per_intensity"][f"L{i}"] = {
            "vbar": vbar,
            "n_axis": {"rows": n_rows, "fraction_inside": n_frac,
                       "n_testable_lines": len(n_lines_t)},
            "m_axis": {"rows": m_rows, "fraction_inside": m_frac,
                       "n_testable_lines": len(m_lines_t)},
            "pass_90pct": bool(n_frac >= 0.9 and m_frac >= 0.9),
            "bprime_n_crossings_descriptive": bp_cross,
            "in_sample_scale": i == 1,
        }
        # no-cert attribution rollup (B5 inputs logged during runs)
        attr = {"floor_starved": 0, "risk_starved": 0, "nuisance_limited": 0, "mixed": 0}
        for c in g["cells"]:
            for k in attr:
                attr[k] += c["bprime"]["attribution"][k]
        out["per_intensity"][f"L{i}"]["bprime_nocert_attribution"] = attr
    out["calibration"] = calib
    out["criterion_a_pass_all_intensities"] = all(
        v["pass_90pct"] for v in out["per_intensity"].values())
    out["criterion_b_note"] = ("held-out r^2 >= 0.84 carried by R023 surface CV "
                               "(s*(n,m) data); criterion (c) by R042")
    save_json("R009_R012_contour_analysis.json", out)
    print({"criterion_a": out["criterion_a_pass_all_intensities"],
           "per_intensity_pass": {k: v["pass_90pct"]
                                  for k, v in out["per_intensity"].items()}})


if __name__ == "__main__":
    main()
