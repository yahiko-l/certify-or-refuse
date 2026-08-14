"""B′ / oracle-A certificate kernel — Floor Certification Law experiment suite.

Implements EXACTLY the registered formulas of
the R039 pre-registration record (documented in the paper's Appendix G; §4 budget split, §5 rho_lambda
DKW form, §5b test statistics, §6 output rule). Any change of formula or constants
requires formal M0 re-entry (R039 discipline); unit tests in tests/test_certificate.py
pin these implementations to hand-computed fixtures.

Two input paths, same formulas:
  * counts path  — stratified synthetic worlds (cells = strata, prefix lattice),
                   vectorized over Monte-Carlo repeats.
  * samples path — score arrays + arbitrary cell edges / thresholds (real data and
                   partial-cell fixtures; A_lambda may cut cell interiors).

Model tags: certify_bprime_* = Model B′ (estimated weights, K-cell stratified class);
certify_oracle_* = Model A (true weights known; rate-ceiling reference arm).
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------- budget (R039 §4)


def budget_split(delta: float, n_lattice: int) -> tuple[float, float, float]:
    """Registered split: delta_w = delta/2, delta_r' = delta_f' = delta/(4|Lambda|)."""
    d_w = delta / 2.0
    d_r = delta / (4.0 * n_lattice)
    d_f = delta / (4.0 * n_lattice)
    assert abs(d_w + n_lattice * d_r + n_lattice * d_f - delta) < 1e-12, "budget != delta"
    return d_w, d_r, d_f


# ------------------------------------------------- registered statistics (R039 §5/§5b)


def mass_L(K: int, delta_w: float) -> float:
    """Union accounting for the 2K cell-mass estimates (R039 §5 v2):
    per-event confidence gamma_w = delta_w/(4K), L_w = ln(8K/delta_w)."""
    return float(np.log(8.0 * K / delta_w))


def bernstein_mass_eps(phat: np.ndarray, n: int, L: float) -> np.ndarray:
    """Per-cell empirical-Bernstein band on a cell mass (R039 §5 v2):
    e = sqrt(2 phat (1-phat) L / n) + 7 L / (3 (n-1))."""
    return np.sqrt(2.0 * phat * (1.0 - phat) * L / n) + 7.0 * L / (3.0 * (n - 1.0))


def eb_ucb_unit(zbar: np.ndarray, vhat: np.ndarray, n: int, gamma: float) -> np.ndarray:
    """One-sided Maurer–Pontil empirical-Bernstein UCB for variables in [0,1].

    UCB_Z = Zbar + sqrt(2 Vhat ln(2/gamma) / n) + 7 ln(2/gamma) / (3 (n-1)),
    Vhat = biased (1/n) empirical variance.
    """
    ln = np.log(2.0 / gamma)
    return zbar + np.sqrt(2.0 * vhat * ln / n) + 7.0 * ln / (3.0 * (n - 1.0))


def risk_ucb_from_moments(mean_x: np.ndarray, ex2_x: np.ndarray, n_r: int,
                          B: float, alpha: float, gamma: float) -> np.ndarray:
    """Registered risk UCB: affine transform Z = (X + alpha*B)/B in [0,1], EB on Z,
    back-transform UCB_X = B*UCB_Z - alpha*B.  X = w_hat * S_lambda * (L - alpha)."""
    zbar = (mean_x + alpha * B) / B
    # Var(Z) = Var(X)/B^2 with the biased convention
    vhat_z = np.maximum(ex2_x - mean_x ** 2, 0.0) / (B * B)
    ucb_z = eb_ucb_unit(zbar, vhat_z, n_r, gamma)
    return B * ucb_z - alpha * B


def bernstein_lcb(phat: np.ndarray, m_f: int, gamma: float) -> np.ndarray:
    """Registered floor LCB (Bernoulli, one-sided empirical Bernstein):
    LCB = phat - sqrt(2 phat (1-phat) ln(2/gamma) / m) - 7 ln(2/gamma) / (3 (m-1))."""
    ln = np.log(2.0 / gamma)
    return phat - np.sqrt(2.0 * phat * (1.0 - phat) * ln / m_f) - 7.0 * ln / (3.0 * (m_f - 1.0))


def what_cells(p_hat: np.ndarray, q_hat: np.ndarray, B: float) -> np.ndarray:
    """w_hat_k = clip(q_hat/p_hat, 0, B), with the branch w_hat_k := B when p_hat_k = 0."""
    w = np.full_like(p_hat, B, dtype=float)
    nz = p_hat > 0
    w[nz] = np.clip(q_hat[nz] / p_hat[nz], 0.0, B)
    return w


def err_cells(p_hat: np.ndarray, w_hat: np.ndarray, e_p: np.ndarray, e_q: np.ndarray,
              B: float) -> np.ndarray:
    """Registered per-cell ratio error (R039 §5 v2) with explicit branches:
    p_tilde = max(p_hat - e_k^P, 0); err = B if p_tilde = 0 else
    min(B, (e_k^Q + w_hat e_k^P)/p_tilde)."""
    p_tilde = np.maximum(p_hat - e_p, 0.0)
    err = np.full_like(p_hat, B, dtype=float)
    nz = p_tilde > 0
    err[nz] = np.minimum(B, (e_q[nz] + w_hat[nz] * e_p[nz]) / p_tilde[nz])
    return err


def rho_lambda_geometric(intersects: np.ndarray, p_hat: np.ndarray, e_p: np.ndarray,
                         err: np.ndarray) -> np.ndarray:
    """Registered localized aggregation (R039 §5 v2): only cells geometrically
    intersecting A_lambda enter, each at its full-cell upper mass:
    rho_lambda = sum_{k: cell∩A_lambda ≠ ∅} (p_hat_k + e_k^P) err_k.
    intersects: (|Λ|, K) boolean from LATTICE/CELL GEOMETRY (not data)."""
    contrib = (p_hat + e_p) * err                       # (..., K)
    return np.einsum("...k,jk->...j", contrib, intersects.astype(float))


# --------------------------------------------------------------- counts path (synthetic)


def prefix_intersects(cell_of_bin: np.ndarray, lattice_bins: np.ndarray,
                      K: int) -> np.ndarray:
    """Geometric intersection table for prefix lattices: cell k intersects
    A_j = bins 0..lattice_bins[j] iff the cell's FIRST bin <= lattice_bins[j].
    Pure bin geometry — no data enters. Returns (|Λ|, K) boolean."""
    first_bin = np.array([np.flatnonzero(cell_of_bin == k)[0] for k in range(K)])
    return first_bin[None, :] <= np.asarray(lattice_bins, int)[:, None]


def certify_bprime_counts(cnt_wP, cnt_r, loss_r, cnt_wQ, cnt_f,
                          B: float, alpha: float, beta: float, delta: float,
                          cell_of_bin=None, lattice_bins=None):
    """Model B′ on stratified worlds, prefix lattice over declared micro-bins.

    Inputs (R repeats stacked on axis 0; n_bins = micro-bin granularity of the world):
      cnt_wP (R,n_bins) ~ Mult(n_w, p)   — D_w^P counts (source, weight split)
      cnt_r  (R,n_bins) ~ Mult(n_r, p)   — D_r counts (source, risk split)
      loss_r (R,n_bins) ~ Binom(cnt_r, eta) — losses among D_r items per bin
      cnt_wQ (R,n_bins) ~ Mult(m_w, q)   — D_w^Q counts (target, weight split)
      cnt_f  (R,n_bins) ~ Mult(m_f, q)   — D_f counts (target, floor split)
      cell_of_bin (n_bins,) int — B′ macro-cell of each micro-bin (default: identity,
        cells = bins); lattice_bins (|Λ|,) int — prefix endpoints A_j = bins 0..b_j
        (default: all prefixes). Lattice may cut macro-cell interiors; intersection
        masses are computed empirically per the registered ρ_λ form.

    Consumes ONLY the five count matrices + registered constants — no world object
    (no-oracle-leak; tamper-tested in R003).
    """
    cnt_wP = np.atleast_2d(cnt_wP); cnt_r = np.atleast_2d(cnt_r)
    loss_r = np.atleast_2d(loss_r); cnt_wQ = np.atleast_2d(cnt_wQ)
    cnt_f = np.atleast_2d(cnt_f)
    R, n_bins = cnt_wP.shape
    if cell_of_bin is None:
        cell_of_bin = np.arange(n_bins)
    cell_of_bin = np.asarray(cell_of_bin, int)
    K = int(cell_of_bin.max()) + 1
    if lattice_bins is None:
        lattice_bins = np.arange(n_bins)
    lattice_bins = np.asarray(lattice_bins, int)
    n_lat = len(lattice_bins)
    n_w = int(cnt_wP.sum(axis=1)[0]); n_r = int(cnt_r.sum(axis=1)[0])
    m_w = int(cnt_wQ.sum(axis=1)[0]); m_f = int(cnt_f.sum(axis=1)[0])
    d_w, d_r, d_f = budget_split(delta, n_lat)
    L_w = mass_L(K, d_w)                       # R039 §5 v2 union accounting

    # macro-cell masses and ratio estimates (from the weight splits only)
    onehot = (cell_of_bin[None, :] == np.arange(K)[:, None])          # (K, n_bins)
    p_hat = (cnt_wP @ onehot.T) / n_w                                  # (R,K)
    q_hat = (cnt_wQ @ onehot.T) / m_w
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p_hat > 0, q_hat / np.where(p_hat > 0, p_hat, 1.0), np.inf)
    w_hat = np.where(p_hat > 0, np.clip(ratio, 0.0, B), B)
    e_p = bernstein_mass_eps(p_hat, n_w, L_w)                          # (R,K)
    e_q = bernstein_mass_eps(q_hat, m_w, L_w)
    p_tilde = np.maximum(p_hat - e_p, 0.0)
    err = np.where(p_tilde > 0,
                   np.minimum(B, (e_q + w_hat * e_p) / np.where(p_tilde > 0, p_tilde, 1.0)),
                   B)                                                  # (R,K)
    # registered ρ_λ (v2): Σ over GEOMETRICALLY intersecting cells of (p̂_k + e_k^P)·err_k
    geom = prefix_intersects(cell_of_bin, lattice_bins, K)             # (|Λ|,K)
    rho = rho_lambda_geometric(geom, p_hat, e_p, err)                  # (R,|Λ|)

    # risk statistic on D_r with plug-in ŵ mapped to micro-bins; X = ŵ·S_λ·(L−α),
    # X = 0 outside the prefix.
    w_bin = w_hat[:, cell_of_bin]                                      # (R,n_bins)
    v1 = w_bin * (1.0 - alpha)
    v0 = w_bin * (0.0 - alpha)
    s_mean = np.cumsum(loss_r * v1 + (cnt_r - loss_r) * v0, axis=1)[:, lattice_bins] / n_r
    s_ex2 = np.cumsum(loss_r * v1 ** 2 + (cnt_r - loss_r) * v0 ** 2, axis=1)[:, lattice_bins] / n_r
    ucb_x = risk_ucb_from_moments(s_mean, s_ex2, n_r, B, alpha, d_r)   # (R,|Λ|)
    risk_pass = ucb_x <= -rho

    # floor statistic on D_f (weight-free — never touches ŵ)
    p_f = np.cumsum(cnt_f, axis=1)[:, lattice_bins] / m_f
    lcb = bernstein_lcb(p_f, m_f, d_f)
    floor_pass = (lcb >= beta) & (lcb > 0.0)

    chosen = _choose(lcb, risk_pass & floor_pass)
    return {"chosen": chosen, "risk_pass": risk_pass, "floor_pass": floor_pass,
            "rho": rho, "ucb_x": ucb_x, "floor_lcb": lcb, "lattice_bins": lattice_bins,
            "splits": (n_w, n_r, m_w, m_f), "budget": (d_w, d_r, d_f)}


def certify_oracle_counts(cnt_src, loss_src, cnt_tgt, w_true,
                          alpha: float, beta: float, delta: float,
                          lattice_bins=None):
    """Model A (oracle weights): full labeled source for risk, full target for floor.
    No weight estimation, no rho; budget delta_r' = delta_f' = delta/(2|Lambda|).
    w_true (n_bins,) enters as the KNOWN micro-bin weight vector (the oracle's
    knowledge — this is the reference arm, not a leak; B′ never sees it)."""
    cnt_src = np.atleast_2d(cnt_src); loss_src = np.atleast_2d(loss_src)
    cnt_tgt = np.atleast_2d(cnt_tgt)
    R, n_bins = cnt_src.shape
    if lattice_bins is None:
        lattice_bins = np.arange(n_bins)
    lattice_bins = np.asarray(lattice_bins, int)
    n_lat = len(lattice_bins)
    n = int(cnt_src.sum(axis=1)[0]); m = int(cnt_tgt.sum(axis=1)[0])
    d_r = delta / (2.0 * n_lat); d_f = delta / (2.0 * n_lat)
    B_eff = float(np.max(w_true))             # oracle knows the true range bound
    v1 = (w_true * (1.0 - alpha))[None, :]
    v0 = (w_true * (0.0 - alpha))[None, :]
    s_mean = np.cumsum(loss_src * v1 + (cnt_src - loss_src) * v0, axis=1)[:, lattice_bins] / n
    s_ex2 = np.cumsum(loss_src * v1 ** 2 + (cnt_src - loss_src) * v0 ** 2, axis=1)[:, lattice_bins] / n
    ucb_x = risk_ucb_from_moments(s_mean, s_ex2, n, B_eff, alpha, d_r)
    risk_pass = ucb_x <= 0.0
    p_f = np.cumsum(cnt_tgt, axis=1)[:, lattice_bins] / m
    lcb = bernstein_lcb(p_f, m, d_f)
    floor_pass = (lcb >= beta) & (lcb > 0.0)
    chosen = _choose(lcb, risk_pass & floor_pass)
    return {"chosen": chosen, "risk_pass": risk_pass, "floor_pass": floor_pass,
            "ucb_x": ucb_x, "floor_lcb": lcb, "lattice_bins": lattice_bins}


def _choose(lcb: np.ndarray, passing: np.ndarray) -> np.ndarray:
    """Registered output rule: among passing lambda maximize floor-LCB; ties toward
    larger acceptance (larger prefix index); no pass -> -1."""
    R, K = lcb.shape
    masked = np.where(passing, lcb, -np.inf)
    # argmax with ties -> LARGEST index: reverse, argmax picks first occurrence
    rev = masked[:, ::-1]
    idx_rev = np.argmax(rev, axis=1)
    chosen = (K - 1) - idx_rev
    chosen[~passing.any(axis=1)] = -1
    return chosen


def score_risk_pass_evalue(w_hat, cnt_r, loss_r, cell_of_bin, lattice_bins, rho,
                           n_r: int, B: float, alpha: float, d_r: float, n_grid: int = 8):
    """Floor-augmented SCoRE risk test (R068): a weighted BETTING e-value (mixture over a
    fixed bet grid) for H0 'risk unsafe' E_P[ŵ S_λ (L−α)] >= −ρ_λ; certify-risk iff
    E_mix >= 1/d_r. Conditional on D_w (ŵ, ρ fixed) the D_r points are i.i.d., so each
    fixed-bet product ∏(1+λ_bet U_i) is a valid e-value (E[·|D_w]=∏(1+λ_bet E[U_i])
    <= exp(λ_bet n_r E[U]) <= 1 under H0) and the mixture is valid by averaging; the
    |Λ|·d_r = δ/4 union matches B′'s risk budget. U_i = −(Z_i + ρ_λ),
    Z_i = ŵ(cell_i)·S_{λ,i}·(L_i−α). SAME ŵ/ρ as certify_bprime_counts — only the risk
    test differs (faithful batch SCoRE-style e-value; per-sample betting is the follow-up).
    Returns (risk_pass (R,|Λ|) bool, logE_mix (R,|Λ|))."""
    cnt_r = np.atleast_2d(cnt_r); loss_r = np.atleast_2d(loss_r)
    R, n_bins = cnt_r.shape
    cell_of_bin = np.asarray(cell_of_bin, int); K = int(cell_of_bin.max()) + 1
    lattice_bins = np.asarray(lattice_bins, int); n_lat = len(lattice_bins)
    rho = np.atleast_2d(rho)                                          # (R, |Λ|)
    onehot = (cell_of_bin[None, :] == np.arange(K)[:, None]).astype(float)   # (K, n_bins)
    # per-cell cumulative accepted (loss=1) and accepted total counts up to each prefix
    cl1 = np.cumsum(loss_r[:, None, :] * onehot[None, :, :], axis=2)[:, :, lattice_bins]
    cla = np.cumsum(cnt_r[:, None, :] * onehot[None, :, :], axis=2)[:, :, lattice_bins]
    n1 = cl1; n0 = cla - cl1                                          # (R, K, |Λ|)
    n_non = n_r - cla.sum(axis=1)                                     # (R, |Λ|) non-accepted
    w = w_hat[:, :, None]                                             # (R, K, 1)
    rho3 = rho[:, None, :]                                            # (R, 1, |Λ|)
    U1 = -((1.0 - alpha) * w + rho3)                                 # accepted loss=1 (most neg)
    U0 = alpha * w - rho3                                             # accepted loss=0
    # per-replicate bet grid (depends only on D_w via rho; batch-independent, reproducible)
    u_neg = (1.0 - alpha) * B + rho.max(axis=1) + 1e-12              # (R,) bound on |negative U|
    lam_max = 0.5 / u_neg                                            # (R,); keeps 1+λU > 1/2 > 0
    fracs = np.linspace(1.0 / n_grid, 1.0, n_grid)                   # fixed fractional bet grid
    logE = np.empty((n_grid, R, n_lat))
    for g in range(n_grid):
        lam = fracs[g] * lam_max                                     # (R,) per-replicate bet
        t1 = (n1 * np.log1p(lam[:, None, None] * U1)).sum(axis=1)    # (R, |Λ|)
        t0 = (n0 * np.log1p(lam[:, None, None] * U0)).sum(axis=1)
        tn = n_non * np.log1p(lam[:, None] * (-rho))
        logE[g] = t1 + t0 + tn
    mx = logE.max(axis=0)                                            # log-sum-exp mixture
    logE_mix = mx + np.log(np.exp(logE - mx[None, :, :]).sum(axis=0) / n_grid)
    return logE_mix >= np.log(1.0 / d_r), logE_mix


# --------------------------------------------------------------- samples path (real / fixtures)


def certify_bprime_samples(scores_wP, scores_r, losses_r, scores_wQ, scores_f,
                           thresholds, cell_edges,
                           B: float, alpha: float, beta: float, delta: float):
    """Model B′, sample-level: acceptance A_t = {score >= t}; cells = intervals of the
    score given by cell_edges (K = len(cell_edges)+1); thresholds need NOT align with
    edges (partial intersections handled exactly as registered).

    Single dataset (no repeat axis). Used on real cached outputs and in fixtures.
    """
    scores_wP = np.asarray(scores_wP, float); scores_r = np.asarray(scores_r, float)
    losses_r = np.asarray(losses_r, float); scores_wQ = np.asarray(scores_wQ, float)
    scores_f = np.asarray(scores_f, float)
    thresholds = np.asarray(thresholds, float)
    edges = np.asarray(cell_edges, float)
    K = len(edges) + 1
    n_lat = len(thresholds)
    n_w, n_r = len(scores_wP), len(scores_r)
    m_w, m_f = len(scores_wQ), len(scores_f)
    d_w, d_r, d_f = budget_split(delta, n_lat)
    L_w = mass_L(K, d_w)                           # R039 §5 v2

    cell_wP = np.digitize(scores_wP, edges)        # 0..K-1
    cell_wQ = np.digitize(scores_wQ, edges)
    p_hat = np.bincount(cell_wP, minlength=K) / n_w
    q_hat = np.bincount(cell_wQ, minlength=K) / m_w
    w_hat = what_cells(p_hat, q_hat, B)
    e_p = bernstein_mass_eps(p_hat, n_w, L_w)
    e_q = bernstein_mass_eps(q_hat, m_w, L_w)
    err = err_cells(p_hat, w_hat, e_p, e_q, B)

    cell_r = np.digitize(scores_r, edges)
    w_r = w_hat[cell_r]                            # plug-in weight per D_r item

    # geometric intersection (v2): A_t = {score >= t}; cell k = (edge_{k-1}, edge_k]
    # intersects A_t iff its UPPER boundary exceeds t (last cell: +inf, always).
    upper = np.append(edges, np.inf)
    geom = upper[None, :] > thresholds[:, None]    # (|Λ|, K)
    rho = rho_lambda_geometric(geom, p_hat, e_p, err)
    ucb_x = np.empty(n_lat)
    p_f_hat = np.empty(n_lat)
    for j, t in enumerate(thresholds):
        x = w_r * (scores_r >= t) * (losses_r - alpha)
        mean_x = x.mean(); ex2_x = (x ** 2).mean()
        ucb_x[j] = risk_ucb_from_moments(np.array(mean_x), np.array(ex2_x),
                                         n_r, B, alpha, d_r)
        p_f_hat[j] = (scores_f >= t).mean()
    lcb = bernstein_lcb(p_f_hat, m_f, d_f)
    risk_pass = ucb_x <= -rho
    floor_pass = (lcb >= beta) & (lcb > 0.0)
    passing = risk_pass & floor_pass
    if not passing.any():
        chosen = -1
    else:
        masked = np.where(passing, lcb, -np.inf)
        best = masked.max()
        cand = np.where(masked == best)[0]
        # larger acceptance = LOWER threshold; thresholds ascending -> smallest index
        chosen = int(cand.min())
    return {"chosen": chosen, "risk_pass": risk_pass, "floor_pass": floor_pass,
            "rho": rho, "ucb_x": ucb_x, "floor_lcb": lcb,
            "w_hat": w_hat, "err": err, "e_p": e_p, "e_q": e_q,
            "p_hat": p_hat, "p_f_hat": p_f_hat,
            "budget": (d_w, d_r, d_f)}
