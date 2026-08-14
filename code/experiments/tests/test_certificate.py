"""R003 — certificate unit tests + null-world calibration + no-oracle-leak assertions
+ budget accounting assertions (EXPERIMENT_PLAN v2 M0; R039 §8 code assertions).

Run:  python -m experiments.tests.test_certificate   (or pytest).
Every formula test pins the implementation to the REGISTERED forms of
the R039 pre-registration record (paper Appendix G) §5/§5b on hand-computed
fixtures, including a PARTIAL-CELL rho_lambda fixture (advisory round-2 MINOR #2).
"""
from __future__ import annotations

import inspect
import math

import numpy as np

from experiments import analysis as A
from experiments import certificate as C
from experiments import validity as V
from experiments.generators import (World, family_continuum, family_null, null_beta,
                                    family_separation, cells_from_source_quantiles,
                                    lattice_prefix, rng_for)

RESULTS = {}


def test_budget_split():
    for delta in (0.05, 0.1):
        for nl in (16, 50, 64):
            d_w, d_r, d_f = C.budget_split(delta, nl)
            assert abs(d_w + nl * d_r + nl * d_f - delta) < 1e-15
            assert d_w == delta / 2 and d_r == delta / (4 * nl) == d_f
    RESULTS["budget_split"] = "PASS"


def test_formula_match_scalars():
    # EB-UCB unit fixture (registered §5b), independent arithmetic
    zbar, vhat, n, gamma = 0.3, 0.04, 100, 0.01
    ln = math.log(2.0 / gamma)
    expect = zbar + math.sqrt(2 * vhat * ln / n) + 7 * ln / (3 * (n - 1))
    got = float(C.eb_ucb_unit(np.array(zbar), np.array(vhat), n, gamma))
    assert abs(got - expect) < 1e-12, (got, expect)
    # back-transform fixture
    mean_x, ex2_x, B, alpha = -0.02, 0.05, 5.0, 0.2
    zb = (mean_x + alpha * B) / B
    vz = (ex2_x - mean_x ** 2) / B ** 2
    exp_x = B * (zb + math.sqrt(2 * vz * ln / n) + 7 * ln / (3 * (n - 1))) - alpha * B
    got_x = float(C.risk_ucb_from_moments(np.array(mean_x), np.array(ex2_x), n, B, alpha, gamma))
    assert abs(got_x - exp_x) < 1e-12
    # floor LCB fixture
    phat, m, gf = 0.62, 400, 0.002
    lnf = math.log(2.0 / gf)
    exp_l = phat - math.sqrt(2 * phat * (1 - phat) * lnf / m) - 7 * lnf / (3 * (m - 1))
    got_l = float(C.bernstein_lcb(np.array(phat), m, gf))
    assert abs(got_l - exp_l) < 1e-12
    # R039 §5 v2 mass-band fixtures
    assert abs(C.mass_L(16, 0.025) - math.log(8 * 16 / 0.025)) < 1e-15
    ph, nn, LL = 0.0625, 30000, C.mass_L(16, 0.025)
    exp_e = math.sqrt(2 * ph * (1 - ph) * LL / nn) + 7 * LL / (3 * (nn - 1))
    assert abs(float(C.bernstein_mass_eps(np.array(ph), nn, LL)) - exp_e) < 1e-15
    RESULTS["formula_match_scalars"] = "PASS"


def test_rho_partial_cell_fixture():
    """Hand-computed rho_lambda where A_lambda CUTS A CELL INTERIOR (samples path).
    Cells: cell0 = score < 0.5, cell1 = score >= 0.5 (edges=[0.5]); threshold t = 0.75
    -> A = {s >= 0.75} is a strict subset of cell1. v2 semantics pinned: cell0 (upper
    edge 0.5 < t) is geometrically excluded; cell1 enters at FULL-cell upper mass
    (p_hat_1 + e_1^P) even though only part of it is accepted.
    """
    scores_wP = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])   # n_w = 8
    scores_wQ = np.array([0.15, 0.35, 0.55, 0.65, 0.85, 0.95])       # m_w = 6
    scores_r = np.array([0.2, 0.6, 0.8, 0.9])
    losses_r = np.array([0, 0, 1, 0])
    scores_f = np.array([0.3, 0.6, 0.8, 0.85, 0.95])
    B, alpha, beta, delta = 5.0, 0.2, 0.3, 0.05
    thresholds = np.array([0.75])
    edges = np.array([0.5])
    res = C.certify_bprime_samples(scores_wP, scores_r, losses_r, scores_wQ, scores_f,
                                   thresholds, edges, B, alpha, beta, delta)
    # hand computation (R039 §5 v2)
    n_lat = 1
    d_w = delta / 2
    K = 2
    L_w = math.log(8 * K / d_w)
    p_hat = np.array([4 / 8, 4 / 8])
    q_hat = np.array([2 / 6, 4 / 6])
    e_p = np.sqrt(2 * p_hat * (1 - p_hat) * L_w / 8) + 7 * L_w / (3 * 7)
    e_q = np.sqrt(2 * q_hat * (1 - q_hat) * L_w / 6) + 7 * L_w / (3 * 5)
    w_hat = np.clip(q_hat / p_hat, 0, B)
    p_tilde = np.maximum(p_hat - e_p, 0.0)
    err = np.where(p_tilde > 0, np.minimum(B, (e_q + w_hat * e_p) / np.where(p_tilde > 0, p_tilde, 1)), B)
    # geometric selection: cell0 upper edge 0.5 < 0.75 -> excluded; cell1 -> included
    rho_expect = (p_hat[1] + e_p[1]) * err[1]
    assert abs(res["rho"][0] - rho_expect) < 1e-12, (res["rho"][0], rho_expect)
    # risk UCB fixture at the same threshold: D_r items with s >= 0.75: 0.8 (L=1), 0.9 (L=0)
    cell_r = np.array([0, 1, 1, 1])
    w_r = w_hat[cell_r]
    x = w_r * (scores_r >= 0.75) * (losses_r - alpha)
    d_r = delta / (4 * n_lat)
    exp_ucb = float(C.risk_ucb_from_moments(np.array(x.mean()), np.array((x ** 2).mean()),
                                            4, B, alpha, d_r))
    assert abs(res["ucb_x"][0] - exp_ucb) < 1e-12
    # floor LCB fixture: D_f items >= 0.75: 3/5
    d_f = delta / (4 * n_lat)
    exp_lcb = float(C.bernstein_lcb(np.array(3 / 5), 5, d_f))
    assert abs(res["floor_lcb"][0] - exp_lcb) < 1e-12
    RESULTS["rho_partial_cell_fixture"] = "PASS"


def test_counts_vs_samples_agreement():
    """Counts path and samples path implement the same registered formulas: build a
    discrete world, materialize the SAME dataset both as counts and as sample arrays
    (score = 1 - bin/n_bins so 'accept high score' == 'accept low bin prefix')."""
    n_bins = 8
    rng = np.random.default_rng(7)
    p = rng.dirichlet(np.ones(n_bins))
    q = rng.dirichlet(np.ones(n_bins))
    eta = np.linspace(0.05, 0.7, n_bins)
    B, alpha, beta, delta = 4.0, 0.2, 0.25, 0.05
    n_w = n_r = 600
    m_w = m_f = 500
    cnt_wP = rng.multinomial(n_w, p)
    cnt_r = rng.multinomial(n_r, p)
    loss_r = rng.binomial(cnt_r, eta)
    cnt_wQ = rng.multinomial(m_w, q)
    cnt_f = rng.multinomial(m_f, q)
    cell_of_bin = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    lattice_bins = np.array([1, 3, 5, 6])      # prefixes; bin 6 cuts cell 3 interior
    res_c = C.certify_bprime_counts(cnt_wP, cnt_r, loss_r, cnt_wQ, cnt_f, B, alpha,
                                    beta, delta, cell_of_bin, lattice_bins)
    # samples: score(bin) = (n_bins - bin)/n_bins; prefix<=b  <=>  score >= (n_bins-b)/n_bins
    sb = lambda b: (n_bins - b) / n_bins
    def expand(cnt, vals=None):
        scores, labels = [], []
        for b in range(n_bins):
            scores += [sb(b)] * int(cnt[b])
            if vals is not None:
                labels += [1] * int(vals[b]) + [0] * int(cnt[b] - vals[b])
        return (np.array(scores), np.array(labels)) if vals is not None else np.array(scores)
    s_wP = expand(cnt_wP)
    s_r, l_r = expand(cnt_r, loss_r)
    s_wQ = expand(cnt_wQ)
    s_f = expand(cnt_f)
    thresholds = np.array([sb(b) for b in lattice_bins])      # descending in b
    # cell edges in score space: cells {0,1},{2,3},{4,5},{6,7} -> edges at score sb(1), sb(3), sb(5)
    edges = np.sort(np.array([sb(1), sb(3), sb(5)]) - 1e-9)
    res_s = C.certify_bprime_samples(s_wP, s_r, l_r, s_wQ, s_f,
                                     np.sort(thresholds), edges, B, alpha, beta, delta)
    # sorted ascending thresholds correspond to REVERSED lattice order
    order = np.argsort(thresholds)
    for key in ("rho", "ucb_x", "floor_lcb"):
        assert np.allclose(res_c[key][0][np.argsort(np.argsort(lattice_bins))][...],
                           res_c[key][0]), "sanity"
        assert np.allclose(np.asarray(res_s[key])[np.argsort(order)], res_c[key][0],
                           atol=1e-10), (key, res_s[key], res_c[key][0])
    RESULTS["counts_vs_samples_agreement"] = "PASS"


def test_no_oracle_leak_signature_and_tamper():
    sig = inspect.signature(C.certify_bprime_counts)
    forbidden = {"world", "eta", "w_true", "q_true", "p_true", "beta_star", "kappa"}
    assert not (set(sig.parameters) & forbidden), "B' signature must not accept oracle fields"
    world = family_continuum(2)
    cells = cells_from_source_quantiles(world, 8)
    lat = lattice_prefix(world, 20)
    rng = rng_for("tamper", 0)
    args = world.sample_counts(rng, 400, 400, 400, 400)
    out1 = C.certify_bprime_counts(*args, world.B_class, 0.2, 0.3, 0.05, cells, lat)
    # tamper the world AFTER split extraction — outputs must be bit-identical
    world.eta = np.clip(world.eta + 0.31, 0, 1)
    world.q = np.roll(world.q, 5)
    out2 = C.certify_bprime_counts(*args, 5.0 if world.B_class == 5.0 else world.B_class,
                                   0.2, 0.3, 0.05, cells, lat)
    assert np.array_equal(out1["chosen"], out2["chosen"])
    assert np.array_equal(out1["rho"], out2["rho"])
    RESULTS["no_oracle_leak"] = "PASS"


def test_weight_free_floor():
    """Floor statistic must not depend on the weight splits (R039 §8.4): perturbing
    cnt_wP / cnt_wQ changes w_hat but must leave floor_lcb bit-identical."""
    world = family_continuum(1)
    cells = cells_from_source_quantiles(world, 8)
    lat = lattice_prefix(world, 20)
    rng = rng_for("wfree", 0)
    cwP, cr, lr, cwQ, cf = world.sample_counts(rng, 400, 400, 400, 400)
    out1 = C.certify_bprime_counts(cwP, cr, lr, cwQ, cf, world.B_class, 0.2, 0.3, 0.05,
                                   cells, lat)
    rng2 = rng_for("wfree", 1)
    cwP2 = rng2.multinomial(400, world.p)
    cwQ2 = rng2.multinomial(400, world.q)
    out2 = C.certify_bprime_counts(cwP2, cr, lr, cwQ2, cf, world.B_class, 0.2, 0.3, 0.05,
                                   cells, lat)
    assert np.array_equal(out1["floor_lcb"], out2["floor_lcb"])
    RESULTS["weight_free_floor"] = "PASS"


def test_choose_tie_break():
    lcb = np.array([[0.5, 0.7, 0.7, 0.2]])
    passing = np.array([[True, True, True, False]])
    assert C._choose(lcb, passing)[0] == 2          # max LCB, tie -> larger acceptance
    assert C._choose(lcb, np.zeros((1, 4), bool))[0] == -1
    RESULTS["choose_tie_break"] = "PASS"


def test_determinism_streams():
    w = family_continuum(3)
    a1 = w.sample_counts(rng_for(("det", 5), 11), 100, 100, 100, 100)
    a2 = w.sample_counts(rng_for(("det", 5), 11), 100, 100, 100, 100)
    for x, y in zip(a1, a2):
        assert np.array_equal(x, y)
    b = w.sample_counts(rng_for(("det", 5), 12), 100, 100, 100, 100)
    assert not all(np.array_equal(x, y) for x, y in zip(a1, b))
    RESULTS["determinism_streams"] = "PASS"


def test_null_world_calibration():
    """Null-world calibration smoke: at an INFEASIBLE floor (beta > beta*), any issued
    certificate is a violation; per-arm violation frequency must satisfy
    CP-UCB <= delta_tol = 1.25*delta for the GUARANTEED arms (oracle-A, B').
    """
    delta = 0.05
    summary = {}
    for kind in (0, 1, 2):
        world = family_null(kind)
        beta = null_beta(world, margin=0.03)
        res = V.run_cell(world, n=4000, m=4000, alpha=world.meta["alpha"], beta=beta,
                         delta=delta, n_reps=400, cell_key=("null", kind),
                         arms=("oracle_a", "bprime"))
        for arm in ("oracle_a", "bprime"):
            t = res[arm]
            summary[f"null{kind}_{arm}"] = {"viol": t["viol"], "n": t["n_reps"],
                                            "cp_ucb": t["viol_cp_ucb"]}
            assert t["viol_cp_ucb"] <= V.DELTA_TOL_FACTOR * delta + 1e-9, \
                (kind, arm, t["viol"], t["viol_cp_ucb"])
    RESULTS["null_world_calibration"] = summary


def test_feasible_world_certifies():
    """Anti-triviality: at a comfortably feasible floor and (n, m) inside the U2
    regime (nuisance axis included: K=16 needs n_w in the few-1e5 range at this
    slack), B' must certify with high probability and stay valid."""
    world = family_continuum(1)
    beta = world.meta["beta_star"] - 0.22
    res = V.run_cell(world, n=500000, m=400000, alpha=0.2, beta=beta, delta=0.05,
                     n_reps=150, cell_key=("feas", 0), arms=("oracle_a", "bprime"))
    assert res["bprime"]["cert_freq"] >= 0.9, res["bprime"]
    assert res["oracle_a"]["cert_freq"] >= 0.9, res["oracle_a"]
    assert res["bprime"]["viol_cp_ucb"] <= 1.25 * 0.05 + 1e-9
    RESULTS["feasible_world_certifies"] = {
        "bprime_cert_freq": res["bprime"]["cert_freq"],
        "oracle_cert_freq": res["oracle_a"]["cert_freq"]}


def test_contour_provenance_funcs_runnable():
    """Provenance guard (run02 audit D residual #1): analysis.logistic_contour_crossing
    and contour_band_check are the M2-era STRICT-band producers (derived_reports.py
    historical note). They are intentionally RETAINED for audit reproducibility, not
    live-pipeline code; this smoke test pins that they still RUN and return the
    registered structure, so they are exercised by the suite rather than 'uncalled
    dead code'. (Smoke/structure only — the crossing CI is bootstrap-based.)"""
    xs_log = np.log(np.array([1e2, 3e2, 1e3, 3e3, 1e4]))
    freqs = np.array([0.08, 0.30, 0.52, 0.74, 0.93])      # monotone, crosses tau=0.5 mid-grid
    ns = np.full(5, 200)
    # logistic_contour_crossing: full exercise incl. its registered bootstrap CI
    b_hat, (lo, hi) = A.logistic_contour_crossing(xs_log, freqs, ns)
    assert np.isfinite(b_hat) and lo <= hi, (b_hat, lo, hi)
    assert xs_log.min() - 5 <= b_hat <= xs_log.max() + 5, b_hat
    # contour_band_check: reachability + registered return structure via the empty-lines
    # fast path (its per-line band delegates to logistic_contour_crossing, exercised above;
    # we avoid the nested 400x400 bootstrap so this stays a fast provenance smoke test)
    band = A.contour_band_check([], bonferroni=3, theory_fn=lambda fv: 0.0)
    assert set(band) >= {"lines", "fraction_inside", "pass_90pct"}, band
    assert band["lines"] == [] and band["fraction_inside"] == 0.0, band
    RESULTS["contour_provenance_funcs_runnable"] = "PASS"


def main():
    import json
    import time
    t0 = time.time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    out = {"suite": "R003 certificate unit tests + null-world calibration",
           "n_tests": len(fns), "all_pass": True, "results": RESULTS,
           "wall_seconds": round(time.time() - t0, 2)}
    import os
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/R003_unit_tests.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({"R003": "ALL PASS", "n_tests": len(fns),
                      "wall_s": out["wall_seconds"]}))


if __name__ == "__main__":
    main()
