"""Validity-Test Protocol + per-cell Monte-Carlo harness (EXPERIMENT_PLAN v2 protocol
section; binds every "violation frequency <= delta" claim).

Three-quantity discipline (Honesty #8): theorem delta, empirical tolerance
delta_tol = 1.25*delta, and the 95% CP / Holm test confidence are distinct everywhere.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import arms as A
from .generators import World, cells_from_source_quantiles, lattice_prefix, rng_for

DELTA_TOL_FACTOR = 1.25     # registered MC-noise margin (plan protocol)
CP_CONF = 0.95              # one-sided Clopper–Pearson confidence
HOLM_LEVEL = 0.05           # Holm-corrected exact test level


def cp_ucb(k: int, n: int, conf: float = CP_CONF) -> float:
    """One-sided Clopper–Pearson upper confidence bound for a binomial proportion."""
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(conf, k + 1, n - k))


def cp_lcb(k: int, n: int, conf: float = CP_CONF) -> float:
    if k <= 0:
        return 0.0
    return float(stats.beta.ppf(1.0 - conf, k, n - k + 1))


def holm_exact(viol_counts, n_reps, delta: float, level: float = HOLM_LEVEL):
    """Holm-corrected exact one-sided binomial tests of H0: p <= delta vs p > delta
    across certified cells. Returns (any_rejection, rejected_indices, pvalues)."""
    viol_counts = np.asarray(viol_counts, int)
    n_reps = np.asarray(n_reps, int)
    pvals = np.array([stats.binom.sf(k - 1, n, delta) if n > 0 else 1.0
                      for k, n in zip(viol_counts, n_reps)])
    order = np.argsort(pvals)
    m = len(pvals)
    rejected = np.zeros(m, bool)
    for rank, idx in enumerate(order):
        if pvals[idx] <= level / (m - rank):
            rejected[idx] = True
        else:
            break
    return bool(rejected.any()), np.flatnonzero(rejected), pvals


def escalation_target_halfwidth(cert_freq: float, tau: float, delta: float) -> float:
    """Registered adaptive margin: CP half-width must be < min(0.25*delta,
    0.5*|cert_freq - tau|) for near-boundary cells."""
    return min(0.25 * delta, 0.5 * abs(cert_freq - tau))


# ------------------------------------------------------------------ attribution (B5)


def attribute_nocert(res: dict, idx: np.ndarray) -> dict:
    """Deterministic no-cert attribution from B′ certificate internals ONLY (the
    data-computable counterpart of the plan's nuisance-regime parenthetical; the
    oracle-side rho > kappa*s/32 check is recorded separately as a diagnostic).

    Categories: floor-axis-starved / risk-axis-starved / nuisance-limited /
    mixed-or-ambiguous (allowed and bounded; expected < 20%).
    """
    out = {"floor_starved": 0, "risk_starved": 0, "nuisance_limited": 0, "mixed": 0}
    risk_pass = res["risk_pass"][idx]
    floor_pass = res["floor_pass"][idx]
    ucb_x = res["ucb_x"][idx]
    for r in range(risk_pass.shape[0]):
        has_f = floor_pass[r].any()
        has_r = risk_pass[r].any()
        if has_r and not has_f:
            out["floor_starved"] += 1
        elif has_f and not has_r:
            # nuisance-limited iff some floor-passing λ would pass risk with rho = 0
            if (ucb_x[r][floor_pass[r]] <= 0.0).any():
                out["nuisance_limited"] += 1
            else:
                out["risk_starved"] += 1
        else:
            out["mixed"] += 1
    return out


# ------------------------------------------------------------------ per-cell harness

ARM_NAMES = ("oracle_a", "bprime", "score_floor", "floorfree_posthoc", "dro_box",
             "weighted_conformal", "plugin")


def run_cell(world: World, n: int, m: int, alpha: float, beta: float, delta: float,
             n_reps: int, cell_key, arms=ARM_NAMES, K_cells: int | None = None,
             n_lattice: int = 50, rep_offset: int = 0,
             collect_truth_margins: bool = False):
    """Monte-Carlo cell: n_reps repeats of all requested arms on `world` at (n, m).

    Sampling uses the registered per-(cell, repeat) PCG64 streams: cell_key is the
    stable cell index, repeats are rep_offset..rep_offset+n_reps-1 (offsets implement
    adaptive escalation reproducibly). Returns per-arm tallies + B′ attribution.
    Truth-level violation: certificate issued AND (R_Q(λ̂) > alpha OR C_Q(λ̂) < beta).

    B′ cells: the world's DECLARED stratified-class cells (on-class by construction);
    K_cells overrides only for the B4.2 K-sweep, where every sweep K refines the
    declared blocks (kcell family guarantees divisibility) — never silent misspec.
    """
    if K_cells is None:
        cell_of_bin = np.asarray(world.cells, int)
    else:
        from .generators import equal_blocks
        cell_of_bin = equal_blocks(world.n_bins, K_cells)
    lattice_bins = lattice_prefix(world, n_lattice)
    cov = world.coverage()
    risk = world.risk()
    viol_lambda = (risk[lattice_bins] > alpha) | (cov[lattice_bins] < beta)

    n_w, n_r = n // 2, n - n // 2
    m_w, m_f = m // 2, m - m // 2
    # --- sample sufficient statistics, per-(cell, repeat) streams
    cwP = np.empty((n_reps, world.n_bins), int); cr = np.empty_like(cwP)
    lr = np.empty_like(cwP); cwQ = np.empty_like(cwP); cf = np.empty_like(cwP)
    c_src = np.empty_like(cwP); l_src = np.empty_like(cwP); c_tgt = np.empty_like(cwP)
    for i in range(n_reps):
        rng = rng_for(cell_key, rep_offset + i)
        a, b, c, d, e = world.sample_counts(rng, n_w, n_r, m_w, m_f)
        cwP[i], cr[i], lr[i], cwQ[i], cf[i] = a, b, c, d, e
        # full-sample arms reuse independent draws from the same stream
        c_src[i], l_src[i] = world.sample_source(rng, n)
        c_tgt[i] = world.sample_target(rng, m)

    results = {}
    B = world.B_class
    for arm in arms:
        if arm == "oracle_a":
            res = A.arm_oracle_a(c_src, l_src, c_tgt, world.w, alpha, beta, delta,
                                 lattice_bins)
        elif arm == "bprime":
            res = A.arm_bprime(cwP, cr, lr, cwQ, cf, B, alpha, beta, delta,
                               cell_of_bin, lattice_bins)
        elif arm == "score_floor":
            res = A.arm_score_floor(cwP, cr, lr, cwQ, cf, B, alpha, beta, delta,
                                    cell_of_bin, lattice_bins)
        elif arm == "floorfree_posthoc":
            res = A.arm_floorfree_posthoc(cwP, cr, lr, cwQ, cf, B, alpha, beta, delta,
                                          cell_of_bin, lattice_bins)
        elif arm == "dro_box":
            res = A.arm_dro_box(c_src, l_src, c_tgt, B, alpha, beta, delta, lattice_bins)
        elif arm == "weighted_conformal":
            res = A.arm_weighted_conformal(cwP, cr, lr, cwQ, cf, B, alpha, beta, delta,
                                           cell_of_bin, lattice_bins)
        elif arm == "plugin":
            res = A.arm_plugin(c_src, l_src, c_tgt, B, alpha, beta, cell_of_bin,
                               lattice_bins)
        else:
            raise ValueError(arm)
        chosen = res["chosen"]
        certified = chosen >= 0
        viol = np.zeros(n_reps, bool)
        viol[certified] = viol_lambda[chosen[certified]]
        true_acc = np.zeros(n_reps)
        true_acc[certified] = cov[lattice_bins[chosen[certified]]]
        tally = {
            "n_reps": n_reps, "cert": int(certified.sum()),
            "viol": int(viol.sum()), "escalate_all": int((~certified).sum()),
            "cert_freq": float(certified.mean()), "viol_freq": float(viol.mean()),
            "viol_cp_ucb": cp_ucb(int(viol.sum()), n_reps),
            "mean_true_acceptance_when_cert": float(true_acc[certified].mean())
                                              if certified.any() else 0.0,
        }
        # OUTPUT-ONLY truth-level margins of the issued λ (default off; used by the T4
        # B′ positive-certification grid). Changes NO statistic, certify decision,
        # sampling, or budget — it only reports the true (R_Q, C_Q) of the certified
        # policy, read from the SAME cov/risk/chosen the violation check already uses.
        if collect_truth_margins and certified.any():
            idx_c = chosen[certified]
            tr_c = risk[lattice_bins[idx_c]]
            tc_c = cov[lattice_bins[idx_c]]
            tally["mean_true_risk_when_cert"] = float(tr_c.mean())
            tally["max_true_risk_when_cert"] = float(tr_c.max())
            tally["min_true_cov_when_cert"] = float(tc_c.min())
            tally["risk_margin_mean"] = float(alpha - tr_c.mean())   # >0 ⇒ under budget
            tally["cov_margin_min"] = float(tc_c.min() - beta)       # ≥0 ⇒ floor held
        # joint-claim accounting for non-floor-certifying arms (plan Table-1 semantics);
        # violation events are counted over ALL repeats (the guarantee is
        # P(claim AND invalid) <= delta, unconditional), per code-review round-1 fix.
        if arm == "floorfree_posthoc":
            claimed = certified & res["floor_report_pass"]      # point-estimate path
            jv = np.zeros(n_reps, bool)
            jv[claimed] = viol_lambda[chosen[claimed]]
            tally.update(claimed_pair=int(claimed.sum()),
                         joint_viol=int(jv.sum()),
                         joint_viol_cp_ucb=cp_ucb(int(jv.sum()), n_reps),
                         joint_viol_cp_lcb=cp_lcb(int(jv.sum()), n_reps),
                         lcb_claimed=int((certified
                                          & res["floor_report_lcb_pass"]).sum()),
                         risk_only_viol=int((risk[lattice_bins[chosen[certified]]]
                                             > alpha).sum()) if certified.any() else 0)
        if arm == "weighted_conformal":
            risk_viol = np.zeros(n_reps, bool)
            risk_viol[certified] = risk[lattice_bins[chosen[certified]]] > alpha
            tally.update(risk_viol=int(risk_viol.sum()),
                         risk_viol_cp_ucb=cp_ucb(int(risk_viol.sum()), n_reps),
                         floor_certified=False)
        if arm == "bprime":
            nocert_idx = np.flatnonzero(~certified)
            tally["attribution"] = attribute_nocert(res, nocert_idx)
            tally["mean_rho_at_chosen"] = float(
                res["rho"][np.flatnonzero(certified), chosen[certified]].mean()) \
                if certified.any() else None
            tally["sum_rho_at_chosen"] = float(
                res["rho"][np.flatnonzero(certified), chosen[certified]].sum()) \
                if certified.any() else 0.0
            # oracle-side diagnostic per plan §B5 parenthetical (rho > kappa*s/32):
            # recorded as a DIAGNOSTIC next to the internal attribution rule.
            kappa = world.meta.get("kappa")
            if kappa and world.meta.get("beta_star"):
                s_slack = world.meta["beta_star"] - beta
                thr = kappa * s_slack / 32.0
                min_rho = res["rho"].min(axis=1)
                tally["diag_nocert_rho_gt_kappa_s_32"] = int(
                    (min_rho[~certified] > thr).sum())
                tally["diag_threshold_kappa_s_32"] = float(thr)
        results[arm] = tally
    results["_meta"] = {"world": world.name, "n": n, "m": m, "alpha": alpha,
                        "beta": beta, "delta": delta,
                        "delta_tol": DELTA_TOL_FACTOR * delta,
                        "K_cells": int(cell_of_bin.max()) + 1,
                        "n_lattice": len(lattice_bins),
                        "cell_key": str(cell_key), "rep_offset": rep_offset,
                        "beta_star": world.meta.get("beta_star"),
                        "B_class": world.B_class}
    return results


def merge_tallies(a: dict, b: dict) -> dict:
    """Merge two run_cell tallies for the SAME arm/cell (adaptive escalation batches).
    Every count is added; every frequency/CP bound is recomputed from merged counts;
    means are cert-weighted; attribution dicts are added key-wise. No field is left
    stale (code-review round-1 MAJOR fix)."""
    out = {}
    n_tot = a["n_reps"] + b["n_reps"]
    add_keys = {"cert", "viol", "escalate_all", "claimed_pair", "joint_viol",
                "lcb_claimed", "risk_only_viol", "risk_viol",
                "diag_nocert_rho_gt_kappa_s_32"}
    for k in add_keys & set(a):
        out[k] = a[k] + b.get(k, 0)
    out["n_reps"] = n_tot
    out["cert_freq"] = out["cert"] / n_tot
    out["viol_freq"] = out["viol"] / n_tot
    out["viol_cp_ucb"] = cp_ucb(out["viol"], n_tot)
    if "joint_viol" in out:
        out["joint_viol_cp_ucb"] = cp_ucb(out["joint_viol"], n_tot)
        out["joint_viol_cp_lcb"] = cp_lcb(out["joint_viol"], n_tot)
    if "risk_viol" in out:
        out["risk_viol_cp_ucb"] = cp_ucb(out["risk_viol"], n_tot)
    cert_w = max(out["cert"], 1)
    out["mean_true_acceptance_when_cert"] = (
        a["mean_true_acceptance_when_cert"] * a["cert"]
        + b["mean_true_acceptance_when_cert"] * b["cert"]) / cert_w
    if "attribution" in a:
        out["attribution"] = {k: a["attribution"][k] + b["attribution"][k]
                              for k in a["attribution"]}
    if "sum_rho_at_chosen" in a:
        out["sum_rho_at_chosen"] = a["sum_rho_at_chosen"] + b.get("sum_rho_at_chosen", 0.0)
        out["mean_rho_at_chosen"] = (out["sum_rho_at_chosen"] / out["cert"]
                                     if out["cert"] > 0 else None)
    for k in ("floor_certified", "diag_threshold_kappa_s_32"):
        if k in a:
            out[k] = a[k]
    return out
