"""Six comparison arms + reporting conventions (EXPERIMENT_PLAN v2 §B1 systems list).

Every arm consumes ONLY sampled sufficient statistics + registered constants; the
validity harness (validity.py) judges truth-level outcomes afterwards. Oracle arms
receive the true weight vector as their DEFINING knowledge (Model A) — B′ and all
baselines never see it.

Arms:
  1 arm_oracle_a          — Model A floor-aware certificate (T3 reference; rate ceiling)
  2 arm_bprime            — Model B′ floor-aware budgeted certificate (THE method)
  3 arm_floorfree_posthoc — risk-only certification + post-hoc floor REPORT (simpler dir.)
  4 arm_dro_box           — worst-case-box robust certificate over {w <= B} (DKW-CVaR)
  5 arm_weighted_conformal— IW-CRC-style weighted selective risk control (global, no floor)
  6 arm_plugin            — naive plug-in optimizer (point estimates, no certificate)

Reporting conventions (never arms): escalate-all = no-automatic-answer-guarantee
(chosen == -1 everywhere); oracle-frontier diagnostic (synthetic only, harness-side).

Resource convention (documented for the comparison standard): every arm receives the
same TOTAL (n, m). Arms needing weight estimation (2, 3, 5) split 50/50 per R039 §3;
oracle-A / DRO-box / plug-in have no weight-estimation step and use the full samples.
"""
from __future__ import annotations

import numpy as np

from .certificate import (budget_split, mass_L, bernstein_mass_eps, bernstein_lcb,
                          risk_ucb_from_moments, certify_bprime_counts,
                          certify_oracle_counts, prefix_intersects,
                          rho_lambda_geometric, score_risk_pass_evalue, _choose)

# ----------------------------------------------------------- arm 1/2: certificates


def arm_oracle_a(cnt_src, loss_src, cnt_tgt, w_true, alpha, beta, delta, lattice_bins):
    return certify_oracle_counts(cnt_src, loss_src, cnt_tgt, w_true, alpha, beta, delta,
                                 lattice_bins=lattice_bins)


def arm_bprime(cnt_wP, cnt_r, loss_r, cnt_wQ, cnt_f, B, alpha, beta, delta,
               cell_of_bin, lattice_bins):
    return certify_bprime_counts(cnt_wP, cnt_r, loss_r, cnt_wQ, cnt_f, B, alpha, beta,
                                 delta, cell_of_bin=cell_of_bin, lattice_bins=lattice_bins)


def arm_score_floor(cnt_wP, cnt_r, loss_r, cnt_wQ, cnt_f, B, alpha, beta, delta,
                    cell_of_bin, lattice_bins):
    """Arm 7: floor-augmented SCoRE (R068) — strongest same-target baseline.
    The ŵ / ρ_λ / floor-LCB / budget block is IDENTICAL to certify_bprime_counts
    (verbatim, same estimates); the ONLY difference is the RISK test: a weighted
    betting e-value (certificate.score_risk_pass_evalue) replaces B′'s empirical-Bernstein
    UCB. Head-to-head with arm_bprime isolates 'e-value vs ρ-budgeted Bernstein UCB'."""
    cnt_wP = np.atleast_2d(cnt_wP); cnt_r = np.atleast_2d(cnt_r)
    loss_r = np.atleast_2d(loss_r); cnt_wQ = np.atleast_2d(cnt_wQ); cnt_f = np.atleast_2d(cnt_f)
    R, n_bins = cnt_wP.shape
    cell_of_bin = np.asarray(cell_of_bin, int); K = int(cell_of_bin.max()) + 1
    lattice_bins = np.asarray(lattice_bins, int); n_lat = len(lattice_bins)
    n_w = int(cnt_wP.sum(axis=1)[0]); n_r = int(cnt_r.sum(axis=1)[0])
    m_w = int(cnt_wQ.sum(axis=1)[0]); m_f = int(cnt_f.sum(axis=1)[0])
    d_w, d_r, d_f = budget_split(delta, n_lat)
    L_w = mass_L(K, d_w)
    # ---- ŵ / ρ_λ block: VERBATIM from certify_bprime_counts (identical estimates) ----
    onehot = (cell_of_bin[None, :] == np.arange(K)[:, None])
    p_hat = (cnt_wP @ onehot.T) / n_w
    q_hat = (cnt_wQ @ onehot.T) / m_w
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p_hat > 0, q_hat / np.where(p_hat > 0, p_hat, 1.0), np.inf)
    w_hat = np.where(p_hat > 0, np.clip(ratio, 0.0, B), B)
    e_p = bernstein_mass_eps(p_hat, n_w, L_w)
    e_q = bernstein_mass_eps(q_hat, m_w, L_w)
    p_tilde = np.maximum(p_hat - e_p, 0.0)
    err = np.where(p_tilde > 0,
                   np.minimum(B, (e_q + w_hat * e_p) / np.where(p_tilde > 0, p_tilde, 1.0)), B)
    geom = prefix_intersects(cell_of_bin, lattice_bins, K)
    rho = rho_lambda_geometric(geom, p_hat, e_p, err)
    # ---- ONLY DIFFERENCE FROM B′: SCoRE weighted betting e-value risk test (R068) ----
    risk_pass, logE_mix = score_risk_pass_evalue(w_hat, cnt_r, loss_r, cell_of_bin,
                                                 lattice_bins, rho, n_r, B, alpha, d_r)
    # ---- floor test: VERBATIM from certify_bprime_counts (weight-free) ----
    p_f = np.cumsum(cnt_f, axis=1)[:, lattice_bins] / m_f
    lcb = bernstein_lcb(p_f, m_f, d_f)
    floor_pass = (lcb >= beta) & (lcb > 0.0)
    chosen = _choose(lcb, risk_pass & floor_pass)
    return {"chosen": chosen, "risk_pass": risk_pass, "floor_pass": floor_pass,
            "rho": rho, "logE_mix": logE_mix, "floor_lcb": lcb,
            "splits": (n_w, n_r, m_w, m_f), "budget": (d_w, d_r, d_f)}


# ------------------------------------------- shared estimated-weight risk machinery


def _weighted_risk_internals(cnt_wP, cnt_r, loss_r, cnt_wQ, B, alpha, delta_w, delta_r,
                             cell_of_bin, lattice_bins):
    """Risk-side internals shared by arms 3 and 5 (same ŵ/err/UCB machinery as B′,
    same registered formulas; only the budget and the nuisance aggregation differ
    by arm). Returns (ucb_x, rho_localized, rho_global, e_p)."""
    R, n_bins = cnt_wP.shape
    cell_of_bin = np.asarray(cell_of_bin, int)
    K = int(cell_of_bin.max()) + 1
    lattice_bins = np.asarray(lattice_bins, int)
    n_w = int(cnt_wP.sum(axis=1)[0]); n_r = int(cnt_r.sum(axis=1)[0])
    m_w = int(cnt_wQ.sum(axis=1)[0])
    L_w = mass_L(K, delta_w)                       # R039 §5 v2 form throughout
    onehot = (cell_of_bin[None, :] == np.arange(K)[:, None])
    p_hat = (cnt_wP @ onehot.T) / n_w
    q_hat = (cnt_wQ @ onehot.T) / m_w
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p_hat > 0, q_hat / np.where(p_hat > 0, p_hat, 1.0), np.inf)
    w_hat = np.where(p_hat > 0, np.clip(ratio, 0.0, B), B)
    e_p = bernstein_mass_eps(p_hat, n_w, L_w)
    e_q = bernstein_mass_eps(q_hat, m_w, L_w)
    p_tilde = np.maximum(p_hat - e_p, 0.0)
    err = np.where(p_tilde > 0,
                   np.minimum(B, (e_q + w_hat * e_p) / np.where(p_tilde > 0, p_tilde, 1.0)),
                   B)
    geom = prefix_intersects(cell_of_bin, lattice_bins, K)
    rho_loc = rho_lambda_geometric(geom, p_hat, e_p, err)
    rho_glob = ((p_hat + e_p) * err).sum(axis=1, keepdims=True) * np.ones((1, len(lattice_bins)))
    w_bin = w_hat[:, cell_of_bin]
    v1 = w_bin * (1.0 - alpha)
    v0 = w_bin * (0.0 - alpha)
    s_mean = np.cumsum(loss_r * v1 + (cnt_r - loss_r) * v0, axis=1)[:, lattice_bins] / n_r
    s_ex2 = np.cumsum(loss_r * v1 ** 2 + (cnt_r - loss_r) * v0 ** 2, axis=1)[:, lattice_bins] / n_r
    ucb_x = risk_ucb_from_moments(s_mean, s_ex2, n_r, B, alpha, delta_r)
    return ucb_x, rho_loc, rho_glob, e_p


# --------------------------------------------------- arm 3: floor-free + post-hoc


def arm_floorfree_posthoc(cnt_wP, cnt_r, loss_r, cnt_wQ, cnt_f, B, alpha, beta, delta,
                          cell_of_bin, lattice_bins):
    """Simpler direction: certify risk only (budget delta_w = delta/2 on weights,
    delta_r' = delta/(2|Λ|) on risk tests — no floor tests in the certificate), select
    the MAX-ACCEPTANCE risk-certified λ, then post-hoc CHECK the floor.

    Per the proposal's structural critique ("the chosen λ's coverage is UNCERTIFIED"),
    the PRIMARY claim path is the published-practice point-estimate check
    Ĉ(λ̂) >= β on D_f — no confidence adjustment. A Bernstein-LCB variant is also
    reported as a DIAGNOSTIC column (floor_report_lcb_pass) so the honest-LCB
    behavior of this baseline is visible; it is not the claim path."""
    lattice_bins = np.asarray(lattice_bins, int)
    n_lat = len(lattice_bins)
    d_w = delta / 2.0
    d_r = delta / (2.0 * n_lat)
    ucb_x, rho_loc, _, _ = _weighted_risk_internals(
        cnt_wP, cnt_r, loss_r, cnt_wQ, B, alpha, d_w, d_r, cell_of_bin, lattice_bins)
    risk_pass = ucb_x <= -rho_loc
    R = risk_pass.shape[0]
    chosen = np.full(R, -1)
    any_pass = risk_pass.any(axis=1)
    # max acceptance = largest passing prefix
    rev_idx = np.argmax(risk_pass[:, ::-1], axis=1)
    chosen[any_pass] = (n_lat - 1 - rev_idx)[any_pass]
    m_f = int(np.atleast_2d(cnt_f).sum(axis=1)[0])
    p_f = np.cumsum(np.atleast_2d(cnt_f), axis=1)[:, lattice_bins] / m_f
    lcb_report = bernstein_lcb(p_f, m_f, delta)        # diagnostic only
    floor_point = np.full(R, False)
    floor_lcb = np.full(R, False)
    ok = chosen >= 0
    idx = np.arange(R)[ok]
    floor_point[ok] = p_f[idx, chosen[ok]] >= beta     # PRIMARY: uncertified check
    floor_lcb[ok] = lcb_report[idx, chosen[ok]] >= beta
    return {"chosen": chosen, "risk_pass": risk_pass,
            "floor_report_pass": floor_point,          # claim path (point estimate)
            "floor_report_lcb_pass": floor_lcb,        # diagnostic
            "floor_lcb_report": lcb_report, "ucb_x": ucb_x, "rho": rho_loc}


# --------------------------------------------------------------- arm 4: DRO box


def arm_dro_box(cnt_src, loss_src, cnt_tgt, B, alpha, beta, delta, lattice_bins):
    """Worst-case-box robust certificate over {w : 0 <= w <= B, E_P[w] = 1} — the
    standard DRO-style bound: sup_w E_P[w Z_λ] = CVaR_{1/B}(Z_λ) under P, with
    Z_λ = S_λ (L - α) ∈ {1-α, 0, -α}. The CVaR estimate is upper-bounded by an
    ε-shifted empirical distribution with a PER-LAMBDA budget δ/(2|Λ|) (Bonferroni
    across the lattice — Z_λ is a different variable per λ, so one global DKW band is
    not simultaneously valid; code-review round-1 fix). Floor: weight-free
    Bernstein-LCB on the full target sample at δ/(2|Λ|) per λ. The strongest standard
    robust certificate under the class — not a straw-man (plan §B1 arm 4)."""
    cnt_src = np.atleast_2d(cnt_src); loss_src = np.atleast_2d(loss_src)
    cnt_tgt = np.atleast_2d(cnt_tgt)
    lattice_bins = np.asarray(lattice_bins, int)
    n_lat = len(lattice_bins)
    R = cnt_src.shape[0]
    n = int(cnt_src.sum(axis=1)[0]); m = int(cnt_tgt.sum(axis=1)[0])
    tau = 1.0 / B
    eps = np.sqrt(np.log(2.0 / (delta / (2.0 * n_lat))) / (2.0 * n))   # per-λ DKW
    # empirical three-point masses per prefix λ
    m1 = np.cumsum(loss_src, axis=1)[:, lattice_bins] / n              # Z = 1-α
    macc = np.cumsum(cnt_src, axis=1)[:, lattice_bins] / n
    mneg = macc - m1                                                   # Z = -α
    m0 = 1.0 - macc                                                    # Z = 0
    if eps >= tau:
        cvar_ucb = np.full((R, n_lat), 1.0 - alpha)                    # vacuous bound
    else:
        take = tau - eps
        t1 = np.minimum(m1, take)
        t0 = np.minimum(m0, np.maximum(take - t1, 0.0))
        tneg = np.maximum(take - t1 - t0, 0.0)
        cvar_ucb = (eps * (1.0 - alpha) + t1 * (1.0 - alpha) + t0 * 0.0
                    + tneg * (-alpha)) / tau
    risk_pass = cvar_ucb <= 0.0
    d_f = delta / (2.0 * n_lat)
    p_f = np.cumsum(cnt_tgt, axis=1)[:, lattice_bins] / m
    lcb = bernstein_lcb(p_f, m, d_f)
    floor_pass = (lcb >= beta) & (lcb > 0.0)
    chosen = _choose(lcb, risk_pass & floor_pass)
    return {"chosen": chosen, "risk_pass": risk_pass, "floor_pass": floor_pass,
            "cvar_ucb": cvar_ucb, "floor_lcb": lcb}


# ------------------------------------------- arm 5: weighted-conformal style (IW-CRC)


def arm_weighted_conformal(cnt_wP, cnt_r, loss_r, cnt_wQ, cnt_f, B, alpha, beta, delta,
                           cell_of_bin, lattice_bins):
    """Closest published line, adapted to the selective ratio risk: importance-weighted
    risk control with estimated weights, GLOBAL (non-localized) weight-error budget,
    and NO floor as certified object (coverage is only reported). Differences from B′
    are exactly the paper's contribution surface: no localized ρ_λ, no certified floor,
    no two-resource accounting."""
    lattice_bins = np.asarray(lattice_bins, int)
    n_lat = len(lattice_bins)
    d_w = delta / 2.0
    d_r = delta / (2.0 * n_lat)
    ucb_x, _, rho_glob, _ = _weighted_risk_internals(
        cnt_wP, cnt_r, loss_r, cnt_wQ, B, alpha, d_w, d_r, cell_of_bin, lattice_bins)
    risk_pass = ucb_x <= -rho_glob
    R = risk_pass.shape[0]
    chosen = np.full(R, -1)
    any_pass = risk_pass.any(axis=1)
    rev_idx = np.argmax(risk_pass[:, ::-1], axis=1)
    chosen[any_pass] = (n_lat - 1 - rev_idx)[any_pass]
    m_f = int(np.atleast_2d(cnt_f).sum(axis=1)[0])
    cov_report = np.cumsum(np.atleast_2d(cnt_f), axis=1)[:, lattice_bins] / m_f
    return {"chosen": chosen, "risk_pass": risk_pass, "coverage_report": cov_report,
            "ucb_x": ucb_x, "rho_global": rho_glob, "floor_certified": False}


# ----------------------------------------------------------- arm 6: naive plug-in


def arm_plugin(cnt_src, loss_src, cnt_tgt, B, alpha, beta, cell_of_bin, lattice_bins):
    """Naive plug-in optimizer: max estimated target coverage s.t. weighted plug-in
    ratio risk <= alpha (point estimates, clipped histogram weights, no slack, no
    certificate). Expected floor-/risk-invalid somewhere on the grid — quantifies the
    price of certification (plan §B1 arm 6)."""
    cnt_src = np.atleast_2d(cnt_src); loss_src = np.atleast_2d(loss_src)
    cnt_tgt = np.atleast_2d(cnt_tgt)
    cell_of_bin = np.asarray(cell_of_bin, int)
    lattice_bins = np.asarray(lattice_bins, int)
    K = int(cell_of_bin.max()) + 1
    n = int(cnt_src.sum(axis=1)[0]); m = int(cnt_tgt.sum(axis=1)[0])
    onehot = (cell_of_bin[None, :] == np.arange(K)[:, None])
    p_hat = (cnt_src @ onehot.T) / n
    q_hat = (cnt_tgt @ onehot.T) / m
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p_hat > 0, q_hat / np.where(p_hat > 0, p_hat, 1.0), np.inf)
    w_hat = np.where(p_hat > 0, np.clip(ratio, 0.0, B), B)
    w_bin = w_hat[:, cell_of_bin]
    num = np.cumsum(w_bin * loss_src, axis=1)[:, lattice_bins] / n
    den = np.cumsum(w_bin * cnt_src, axis=1)[:, lattice_bins] / n
    with np.errstate(invalid="ignore", divide="ignore"):
        r_hat = np.where(den > 0, num / den, 0.0)
    c_hat = np.cumsum(cnt_tgt, axis=1)[:, lattice_bins] / m
    ok = (r_hat <= alpha) & (c_hat >= beta)
    R = ok.shape[0]
    chosen = np.full(R, -1)
    any_ok = ok.any(axis=1)
    masked = np.where(ok, c_hat, -np.inf)
    rev = masked[:, ::-1]
    idx_rev = np.argmax(rev, axis=1)
    chosen[any_ok] = (masked.shape[1] - 1 - idx_rev)[any_ok]
    return {"chosen": chosen, "r_hat": r_hat, "c_hat": c_hat, "selected_mask": ok}


# -------------------------------------------------------------- diagnostics


def oracle_frontier_diag(world, alpha: float):
    """Oracle-frontier diagnostic line (synthetic ONLY; never an arm, never on real
    data): the true β*-optimal prefix from world parameters."""
    beta_star, j_star = world.frontier(alpha)
    return {"beta_star": beta_star, "j_star": j_star,
            "coverage": float(world.coverage()[j_star]) if j_star >= 0 else 0.0,
            "risk": float(world.risk()[j_star]) if j_star >= 0 else None}
