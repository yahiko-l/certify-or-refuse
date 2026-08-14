"""Analysis layer: contour protocol, exponent fits, additive-law model comparison,
rank correlations, no-cert decomposition (EXPERIMENT_PLAN v2 §B1 metrics, §B4.1, §B2).
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

# ----------------------------------------------------- B1 contour protocol (R009-R016)


def logistic_contour_crossing(xs_log: np.ndarray, freqs: np.ndarray, ns: np.ndarray,
                              tau: float = 0.5):
    """Monotone logistic fit of certification frequency along ONE grid line
    (x = log n or log m); returns (crossing point x*, 95% CI for x*) via profile
    bootstrap on the binomial counts. Registered boundary estimator (plan §B1(a))."""
    ks = np.round(freqs * ns).astype(int)

    def nll(params):
        a, b = params       # P(cert) = sigmoid(a (x - b)); a >= 0 enforces monotone
        a = abs(a)
        p = 1.0 / (1.0 + np.exp(-a * (xs_log - b)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(ks * np.log(p) + (ns - ks) * np.log(1 - p)).sum()

    x0 = np.array([2.0, xs_log[np.argmin(abs(freqs - tau))]])
    fit = optimize.minimize(nll, x0, method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
    a_hat, b_hat = abs(fit.x[0]), fit.x[1]
    # parametric bootstrap for the crossing CI
    rng = np.random.default_rng(20260611)
    crossings = []
    for _ in range(400):
        kb = rng.binomial(ns, np.clip(1.0 / (1.0 + np.exp(-a_hat * (xs_log - b_hat))),
                                      1e-9, 1 - 1e-9))
        fb = optimize.minimize(
            lambda prm: -(kb * np.log(np.clip(1/(1+np.exp(-abs(prm[0])*(xs_log-prm[1]))), 1e-9, 1-1e-9))
                          + (ns - kb) * np.log(np.clip(1 - 1/(1+np.exp(-abs(prm[0])*(xs_log-prm[1]))), 1e-9, 1-1e-9))).sum(),
            np.array([a_hat, b_hat]), method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 2000})
        crossings.append(fb.x[1])
    lo, hi = np.percentile(crossings, [2.5, 97.5])
    return float(b_hat), (float(lo), float(hi))


def contour_band_check(lines: list[dict], bonferroni: int, theory_fn) -> dict:
    """Simultaneous-band contour test: per grid line, crossing CI at level
    (1 - 0.05/bonferroni) via widened percentile range; success = theory crossing
    inside the band for >= 90% of lines (registered operationalization of 'arc length'
    on the log-spaced grid). lines: [{xs_log, freqs, ns, fixed_value}]."""
    inside = 0
    rows = []
    level = 100.0 * (0.05 / bonferroni) / 2.0
    for ln in lines:
        b_hat, _ = logistic_contour_crossing(ln["xs_log"], ln["freqs"], ln["ns"])
        rng = np.random.default_rng(20260611 + 7)
        ks = np.round(ln["freqs"] * ln["ns"]).astype(int)
        boots = []
        for _ in range(400):
            kb = rng.binomial(ln["ns"], np.clip(ks / np.maximum(ln["ns"], 1), 1e-9, 1 - 1e-9))
            fr = kb / np.maximum(ln["ns"], 1)
            try:
                bb, _ci = logistic_contour_crossing(ln["xs_log"], fr, ln["ns"])
                boots.append(bb)
            except Exception:
                continue
        lo, hi = np.percentile(boots, [level, 100.0 - level])
        th = theory_fn(ln["fixed_value"])
        ok = bool(lo <= th <= hi)
        inside += ok
        rows.append({"fixed_value": ln["fixed_value"], "crossing": b_hat,
                     "band": (float(lo), float(hi)), "theory": float(th), "inside": ok})
    frac = inside / max(len(lines), 1)
    return {"lines": rows, "fraction_inside": frac, "pass_90pct": frac >= 0.9}


# --------------------------------------------- exponent fits (R014, R021/R022, B4.1)


def loglog_exponent(x: np.ndarray, y: np.ndarray, se_mode: str = "hc3",
                    boot_y: np.ndarray | None = None):
    """OLS log-log slope with HC3 robust SE (registered) or world-bootstrap SE
    (boot_y: (B, len(x)) bootstrap replicates of y). Returns slope, se, ci95,
    curvature check (quadratic term CI containing 0)."""
    lx, ly = np.log(x), np.log(y)
    X = np.column_stack([np.ones_like(lx), lx])
    coef, *_ = np.linalg.lstsq(X, ly, rcond=None)
    resid = ly - X @ coef
    XtXinv = np.linalg.inv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, XtXinv, X)
    u = resid / (1.0 - h)
    cov_hc3 = XtXinv @ (X.T * (u ** 2)) @ X @ XtXinv
    se = float(np.sqrt(cov_hc3[1, 1]))
    if boot_y is not None:
        slopes = []
        for yb in boot_y:
            cb, *_ = np.linalg.lstsq(X, np.log(yb), rcond=None)
            slopes.append(cb[1])
        se = float(np.std(slopes, ddof=1))
    slope = float(coef[1])
    ci = (slope - 1.96 * se, slope + 1.96 * se)
    # curvature: quadratic term
    X2 = np.column_stack([np.ones_like(lx), lx, lx ** 2])
    c2, *_ = np.linalg.lstsq(X2, ly, rcond=None)
    r2 = ly - X2 @ c2
    dof = max(len(lx) - 3, 1)
    s2 = (r2 ** 2).sum() / dof
    cov2 = s2 * np.linalg.inv(X2.T @ X2)
    q_se = float(np.sqrt(cov2[2, 2]))
    curv_ci = (float(c2[2] - 1.96 * q_se), float(c2[2] + 1.96 * q_se))
    return {"slope": slope, "se": se, "ci95": (float(ci[0]), float(ci[1])),
            "curvature_ci": curv_ci, "curvature_contains_0": curv_ci[0] <= 0 <= curv_ci[1]}


# ---------------------------------- B4.1 additive-law model comparison (R023, 70/30 CV)

MODEL_FORMS = ("additive", "n_only", "m_only", "max_form", "floor_free")


def _design(form: str, n, m, beta, kappa):
    fn = np.sqrt(beta / n) / kappa
    fm = np.sqrt(beta / m)
    gn = 1.0 / np.sqrt(n)        # floor-free: no beta scaling
    gm = 1.0 / np.sqrt(m)
    if form == "additive":
        return np.column_stack([fn, fm, np.ones_like(fn)])
    if form == "n_only":
        return np.column_stack([fn, np.ones_like(fn)])
    if form == "m_only":
        return np.column_stack([fm, np.ones_like(fm)])
    if form == "floor_free":
        return np.column_stack([gn, gm, np.ones_like(gn)])
    raise ValueError(form)


def fit_form(form: str, n, m, beta, kappa, y):
    """Least-squares fit of one functional form for the certifiable-slack surface.
    max_form is nonlinear: y = max(a fn, b fm) + c — fitted by scipy least_squares."""
    if form != "max_form":
        X = _design(form, n, m, beta, kappa)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return ("lin", coef)
    fn = np.sqrt(beta / n) / kappa
    fm = np.sqrt(beta / m)

    def resid(prm):
        a, b, c = prm
        return np.maximum(a * fn, b * fm) + c - y

    sol = optimize.least_squares(resid, x0=np.array([1.0, 1.0, 0.0]))
    return ("max", sol.x)


def predict_form(form: str, params, n, m, beta, kappa):
    kind, coef = params
    if kind == "lin":
        return _design(form, n, m, beta, kappa) @ coef
    a, b, c = coef
    return np.maximum(a * np.sqrt(beta / n) / kappa, b * np.sqrt(beta / m)) + c


def model_comparison(n, m, beta, kappa, y, n_boot: int = 500, seed: int = 20260611):
    """Registered comparison: 70/30 cell split, CV-RMSE on held-out cells + AIC/BIC;
    bootstrap CIs on CV-RMSE per form; success = additive wins with non-overlapping
    CIs against every alternative."""
    n = np.asarray(n, float); m = np.asarray(m, float)
    beta = np.asarray(beta, float); y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    n_cells = len(y)
    perm = rng.permutation(n_cells)
    tr = perm[: int(0.7 * n_cells)]
    te = perm[int(0.7 * n_cells):]
    out = {}
    for form in MODEL_FORMS:
        params = fit_form(form, n[tr], m[tr], beta[tr], kappa, y[tr])
        pred_te = predict_form(form, params, n[te], m[te], beta[te], kappa)
        rmse = float(np.sqrt(np.mean((pred_te - y[te]) ** 2)))
        pred_tr = predict_form(form, params, n[tr], m[tr], beta[tr], kappa)
        rss = float(((pred_tr - y[tr]) ** 2).sum())
        k = 3 if form in ("additive", "floor_free", "max_form") else 2
        ntr = len(tr)
        aic = ntr * np.log(rss / ntr) + 2 * k
        bic = ntr * np.log(rss / ntr) + k * np.log(ntr)
        boots = []
        for _ in range(n_boot):
            idx = rng.choice(te, size=len(te), replace=True)
            boots.append(float(np.sqrt(np.mean(
                (predict_form(form, params, n[idx], m[idx], beta[idx], kappa) - y[idx]) ** 2))))
        out[form] = {"cv_rmse": rmse, "cv_rmse_ci": (float(np.percentile(boots, 2.5)),
                                                     float(np.percentile(boots, 97.5))),
                     "aic": float(aic), "bic": float(bic)}
    add_hi = out["additive"]["cv_rmse_ci"][1]
    wins = {f: out[f]["cv_rmse_ci"][0] > add_hi for f in MODEL_FORMS if f != "additive"}
    out["_verdict"] = {"additive_wins_all_nonoverlap": all(wins.values()),
                       "per_form_nonoverlap": wins}
    return out


def surface_r2_heldout(n, m, beta, kappa, y, seed: int = 20260611):
    """B1(b): additive two-axis fit on a random 70% of cells, r² on the held-out 30%."""
    n = np.asarray(n, float); m = np.asarray(m, float)
    beta = np.asarray(beta, float); y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    tr, te = perm[: int(0.7 * len(y))], perm[int(0.7 * len(y)):]
    params = fit_form("additive", n[tr], m[tr], beta[tr], kappa, y[tr])
    pred = predict_form("additive", params, n[te], m[te], beta[te], kappa)
    ss_res = ((y[te] - pred) ** 2).sum()
    ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


# ------------------------------------------------------------- B2 rank correlations


def rank_correlations(required_n: np.ndarray, localized: np.ndarray,
                      global_ess: np.ndarray, n_boot: int = 2000,
                      seed: int = 20260611):
    """Spearman (primary; Kendall secondary — ties from discretized required-n) of
    required-n with the localized functional vs with global ESS; bootstrap CIs over
    worlds. Success (plan §B2): rho_loc >= 0.9 with CI excluding rho_ess, whose CI
    should contain 0.

    Censoring: required_n = inf (bracket cap exceeded) is INFORMATION, not
    missingness — censored worlds enter as TOP-TIED ranks (any common value above the
    finite maximum yields the same Spearman ranks). Dropping them would bias the
    sample toward easy worlds."""
    rng = np.random.default_rng(seed)
    required_n = np.asarray(required_n, float)
    n_censored = int((~np.isfinite(required_n)).sum())
    fin_max = np.nanmax(np.where(np.isfinite(required_n), required_n, np.nan))
    rn = np.where(np.isfinite(required_n), required_n, fin_max * 10.0)
    loc, ges = np.asarray(localized, float), np.asarray(global_ess, float)
    fin = np.isfinite(rn) & np.isfinite(loc) & np.isfinite(ges)
    rn, loc, ges = rn[fin], loc[fin], ges[fin]
    sp_loc = stats.spearmanr(rn, loc).statistic
    sp_ess = stats.spearmanr(rn, ges).statistic
    kd_loc = stats.kendalltau(rn, loc).statistic
    kd_ess = stats.kendalltau(rn, ges).statistic
    bl, be = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rn), len(rn))
        if len(np.unique(rn[idx])) < 3:
            continue
        bl.append(stats.spearmanr(rn[idx], loc[idx]).statistic)
        be.append(stats.spearmanr(rn[idx], ges[idx]).statistic)
    ci_loc = (float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5)))
    ci_ess = (float(np.percentile(be, 2.5)), float(np.percentile(be, 97.5)))
    return {"spearman_localized": float(sp_loc), "spearman_localized_ci": ci_loc,
            "spearman_global_ess": float(sp_ess), "spearman_global_ess_ci": ci_ess,
            "kendall_localized": float(kd_loc), "kendall_global_ess": float(kd_ess),
            "n_worlds": int(fin.sum()), "n_censored_top": n_censored,
            "success": bool(sp_loc >= 0.9 and ci_loc[0] > ci_ess[1]
                            and ci_ess[0] <= 0.0 <= ci_ess[1])}
