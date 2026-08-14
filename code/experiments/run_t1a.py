"""T1a — B′ positive certification envelope + nuisance-premium law (TPAMI revision).

A NEW SWEEP over the VALIDATED certificate kernel (certificate.py / validity.run_cell).
NO formula or constant changes (R039 discipline intact): this module only chooses the
(n, m, K, B, s) inputs and aggregates onsets — it never touches the registered statistics.
Companion to the T3 Model-B′ nuisance-necessity lower bound (paper/theory_drafts/).

Claim C-T1a (refine-logs/EXPERIMENT_PLAN_TPAMI_REVISION.md): the implementable B′ arm
certifies a CHARACTERIZED region whose budget premium over oracle-A scales as the law's
nuisance dimension predicts —
  * premium ∝ K  (finer partition) at fixed B, and
  * the source split ∝ B²  at fixed K
matching Thm U2: n_w ≳ B²K·p(A)/(κs)², m_w ≳ K·q(A)/(κs)².

Operating point: family_kcell has kappa≈0.35, so s_work=0.12 ⇒ κs≈0.042 — the SAME
operating point as the paper's §5 constant (oracle-A onset 4,096/16,384; B′ 524,288/
1,048,576; 64×). T1a at K=4,B=5 should recover that 64×, then EXTEND it into a premium law.

Worlds:
  * K-sweep refines the declared 4 blocks of family_kcell (256 divisible by every K),
    so every K stays ON-CLASS — the sweep measures the PRICE of a finer partition, never
    silent misspecification (run_cell's K_cells override, B4.2 discipline).
  * B-sweep builds worlds whose ACTUAL weights are identical (all in [1/B, B], no clip
    change for B∈{5,10,20}) so only the certificate's robustness bound B varies — this
    isolates the B² conservatism cost with the world held fixed.

Onset rule: smallest n_total (= n = m) with B′/oracle cert_freq ≥ 0.5 (≥ 0.8 for the
slack-bite, matching R014). Censored (None) when no certification within the n grid — a
finer partition can legitimately exceed budget; censoring is reported, never hidden.

Subcommands:
  python -m experiments.run_t1a kenv       (R050: K-sweep envelope + onsets + premium-vs-K fit)
  python -m experiments.run_t1a bsweep     (R051: B-sweep onsets at K=4 + onset-vs-B fit)
  python -m experiments.run_t1a bsweep_tight (R051b: tight-overlap B-sweep; the ONLY R051b
                                            producer — NOT included in `all`, run it separately)
  python -m experiments.run_t1a slackbite  (R052: B′ onset-vs-slack, the −2 labeled law)
  python -m experiments.run_t1a all        (kenv + bsweep + slackbite + combined summary;
                                            does NOT run bsweep_tight)
  [--reps 600] [--jobs 48] [--smoke]
"""
from __future__ import annotations

import argparse

import numpy as np

from .analysis import loglog_exponent
from .generators import family_kcell, World, equal_blocks, _project_w_onclass
from .validity import run_cell
from .runner_util import save_json, pmap, log_grid

ALPHA, DELTA = 0.2, 0.05
S_WORK = 0.12                      # κs≈0.042 at kcell kappa≈0.35 (paper §5 operating point)
N_LATTICE = 50
LOC = 2.0                         # accepted-region heterogeneity, fixed (edge default)
K_SWEEP = [4, 8, 16, 32, 64]     # refinements of the 4 declared blocks (all on-class)
B_SWEEP = [5.0, 10.0, 20.0]      # certificate robustness bound (world held fixed)
NGRID = log_grid(512, 16777216, 34)         # 2^9 .. 2^24, dense (sharp onsets; resolves plateaus)
SMOKE_NGRID = log_grid(2048, 1048576, 6)
BIG_M = 16_000_000               # floor freed for the slack-bite (mirrors R014)


def _cell_job(spec):
    """One (n_total, K, B, slack, m) Monte-Carlo cell on a fresh kcell world."""
    ntot, K_cells, B_cls, slack, m_val, reps = spec
    world = family_kcell(K_world=4, B_cls=B_cls, loc_level=LOC, alpha=ALPHA)
    beta = float(world.meta["beta_star"]) - slack
    m = int(m_val) if m_val is not None else int(ntot)
    r = run_cell(world, n=int(ntot), m=m, alpha=ALPHA, beta=beta, delta=DELTA,
                 n_reps=reps, arms=("oracle_a", "bprime"), K_cells=int(K_cells),
                 n_lattice=N_LATTICE,
                 cell_key=("T1a", float(B_cls), int(K_cells), float(slack), int(ntot)))
    return (int(ntot), {
        "bprime_cert": r["bprime"]["cert_freq"],
        "oracle_cert": r["oracle_a"]["cert_freq"],
        "bprime_viol_ucb": r["bprime"]["viol_cp_ucb"],
        "oracle_viol_ucb": r["oracle_a"]["viol_cp_ucb"],
        "mean_rho": r["bprime"]["mean_rho_at_chosen"],
        "bprime_acc": r["bprime"]["mean_true_acceptance_when_cert"],
        "oracle_acc": r["oracle_a"]["mean_true_acceptance_when_cert"],
        "bprime_cert_count": r["bprime"]["cert"],
        "oracle_cert_count": r["oracle_a"]["cert"],
        "n_reps": r["bprime"]["n_reps"],
        "beta": beta,
    })


def _onset(rows, key, thr=0.5):
    """Smallest n_total whose cert_freq under `key` ≥ thr; None if never (censored)."""
    for ntot, d in sorted(rows, key=lambda t: t[0]):
        if d[key] is not None and d[key] >= thr:
            return int(ntot)
    return None


def _series(rows):
    return [{"n_total": n, **d} for n, d in sorted(rows, key=lambda t: t[0])]


def _tight_world(B, nbins=256, acc_frac=0.7):
    """On-class K=4 world whose ACCEPTED (low-η) blocks carry weight ≈ acc_frac·B, so the
    true ratio APPROACHES the bound as B grows — the regime where Thm U2's B² source cost
    (n_w ≳ B²K) should bind. Built directly on World/_project_w_onclass: the registered
    generators are untouched. Target side (q uniform, η=0.04+0.6x²) is identical across B,
    so β* and κ are B-invariant — only the certificate's weight-estimation load scales."""
    x = (np.arange(nbins) + 0.5) / nbins
    q = np.full(nbins, 1.0 / nbins)
    eta = 0.04 + 0.6 * x ** 2
    cells = equal_blocks(nbins, 4)
    w_blocks = np.clip(np.array([acc_frac * B, acc_frac * B, 0.5, 0.5]), 1.0 / B, B)
    p = _project_w_onclass(q, w_blocks[cells].astype(float), cells)
    world = World(p=p, q=q, eta=eta, name=f"kcell_tight_B{B:g}", B_class=float(B),
                  cells=cells)
    bstar, jstar = world.frontier(ALPHA)
    world.meta = {"alpha": ALPHA, "beta_star": bstar, "j_star": jstar,
                  "B_class": float(B), "acc_frac": acc_frac}
    return world


def _tight_job(spec):
    ntot, B, reps = spec
    world = _tight_world(B)
    beta = float(world.meta["beta_star"]) - S_WORK
    r = run_cell(world, n=int(ntot), m=int(ntot), alpha=ALPHA, beta=beta, delta=DELTA,
                 n_reps=reps, arms=("oracle_a", "bprime"), K_cells=4, n_lattice=N_LATTICE,
                 cell_key=("T1a_tight", float(B), int(ntot)))
    return (int(ntot), {
        "bprime_cert": r["bprime"]["cert_freq"], "oracle_cert": r["oracle_a"]["cert_freq"],
        "bprime_viol_ucb": r["bprime"]["viol_cp_ucb"], "mean_rho": r["bprime"]["mean_rho_at_chosen"],
        "bprime_acc": r["bprime"]["mean_true_acceptance_when_cert"],
        "beta_star": float(world.meta["beta_star"]), "beta": beta,
    })


def cmd_kenv(args):
    ngrid = SMOKE_NGRID if args.smoke else NGRID
    Ks = [4, 8] if args.smoke else K_SWEEP
    by_K, onset_bp, onset_or = {}, {}, {}
    for K in Ks:
        specs = [(n, K, 5.0, S_WORK, None, args.reps) for n in ngrid]
        rows = pmap(_cell_job, specs, n_jobs=args.jobs, desc=f"T1a kenv K={K}")
        by_K[str(K)] = _series(rows)
        onset_bp[str(K)] = _onset(rows, "bprime_cert")
        onset_or[str(K)] = _onset(rows, "oracle_cert")
    # premium ∝ K fit on finite onsets
    premium = {K: onset_bp[str(K)] / onset_or[str(K)]
               for K in Ks if onset_bp[str(K)] and onset_or[str(K)]}
    Kfit = sorted(premium)
    fitK = (loglog_exponent(np.array(Kfit, float),
                            np.array([premium[K] for K in Kfit], float))
            if len(Kfit) >= 3 else None)
    payload = {
        "run": "R050", "family": "kcell", "loc_level": LOC, "B": 5.0, "s_work": S_WORK,
        "kappa_s_approx": 0.042, "alpha": ALPHA, "delta": DELTA, "n_lattice": N_LATTICE,
        "ngrid": [int(n) for n in ngrid], "K_sweep": Ks,
        "series_by_K": by_K, "onset_bprime": onset_bp, "onset_oracle": onset_or,
        "premium_vs_oracle": {str(k): v for k, v in premium.items()},
        "premium_K_loglog": fitK, "premium_K_pred_slope": 1.0,
        "premium_K_ci_contains_1": (bool(fitK["ci95"][0] <= 1.0 <= fitK["ci95"][1])
                                    if fitK else None),
        "censored_K": [K for K in Ks if not onset_bp[str(K)]],
        "note": "premium ∝ K predicted (each extra cell adds ~const estimation cost; at "
                "n=m the B²K source split binds, B fixed=5). K=4 should recover the paper's "
                "64× (onset_bprime≈524288 vs oracle≈4096-16384) — internal consistency.",
    }
    save_json("R050_t1a_envelope.json", payload)
    print({"onset_bprime": onset_bp, "onset_oracle": onset_or,
           "premium": {k: round(v, 1) for k, v in premium.items()},
           "K_slope": (round(fitK["slope"], 3) if fitK else None),
           "K_ci95": (fitK["ci95"] if fitK else None),
           "ci_contains_1": payload["premium_K_ci_contains_1"]})
    return payload


def cmd_bsweep(args):
    ngrid = SMOKE_NGRID if args.smoke else NGRID
    Bs = [5.0, 10.0] if args.smoke else B_SWEEP
    by_B, onset_bp = {}, {}
    for B in Bs:
        specs = [(n, 4, B, S_WORK, None, args.reps) for n in ngrid]
        rows = pmap(_cell_job, specs, n_jobs=args.jobs, desc=f"T1a bsweep B={B}")
        by_B[str(B)] = _series(rows)
        onset_bp[str(B)] = _onset(rows, "bprime_cert")
    Bf = [B for B in Bs if onset_bp[str(B)]]
    fitB = (loglog_exponent(np.array(Bf, float),
                            np.array([onset_bp[str(B)] for B in Bf], float))
            if len(Bf) >= 3 else None)
    payload = {
        "run": "R051", "family": "kcell", "K": 4, "loc_level": LOC, "s_work": S_WORK,
        "B_sweep": Bs, "series_by_B": by_B, "onset_bprime_by_B": onset_bp,
        "onset_B_loglog": fitB, "onset_B_pred_slope": 2.0,
        "note": "B-sweep holds the ACTUAL world fixed (weights in [1/B,B], no clip change) "
                "— only the certificate robustness bound B varies. Source split ∝ B² predicted "
                "(Thm U2 n_w ≳ B²K); the range term ∝ B can lower the realized slope — reported "
                "as measured, not forced toward 2.",
    }
    save_json("R051_t1a_premium_B.json", payload)
    print({"onset_by_B": onset_bp,
           "B_slope": (round(fitB["slope"], 3) if fitB else None),
           "B_ci95": (fitB["ci95"] if fitB else None)})
    return payload


def cmd_bsweep_tight(args):
    """B-sweep in the TIGHT regime (accepted weight ≈ 0.7·B): the complement to cmd_bsweep
    (fixed small weights). Together they characterize the B-axis in both regimes — flat when
    w≪B, B²-scaling when w≈B — completing the nuisance picture for T3."""
    ngrid = SMOKE_NGRID if args.smoke else NGRID
    Bs = [5.0, 10.0] if args.smoke else [5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0]
    by_B, onset_bp, bstar = {}, {}, {}
    for B in Bs:
        rows = pmap(_tight_job, [(n, B, args.reps) for n in ngrid],
                    n_jobs=args.jobs, desc=f"T1a bsweep_tight B={B}")
        by_B[str(B)] = _series(rows)
        onset_bp[str(B)] = _onset(rows, "bprime_cert")
        bstar[str(B)] = rows[0][1]["beta_star"]
    Bf = [B for B in Bs if onset_bp[str(B)]]
    fitB = (loglog_exponent(np.array(Bf, float),
                            np.array([onset_bp[str(B)] for B in Bf], float))
            if len(Bf) >= 3 else None)
    payload = {
        "run": "R051b", "family": "kcell_tight", "K": 4, "acc_frac": 0.7, "s_work": S_WORK,
        "B_sweep": Bs, "beta_star_by_B": bstar, "series_by_B": by_B,
        "onset_bprime_by_B": onset_bp, "onset_B_loglog": fitB, "onset_B_pred_slope": 2.0,
        "note": "TIGHT regime: accepted weight ≈0.7·B approaches the bound as B grows, so "
                "Thm U2's B² source cost (n_w ≳ B²K) should bind. Contrast cmd_bsweep (R051, "
                "fixed small weights → B-insensitive). β* is B-invariant (target-side), so any "
                "onset growth is pure weight-estimation B² cost. Measured, not forced.",
    }
    save_json("R051b_t1a_premium_B_tight.json", payload)
    print({"regime": "tight (w≈B)", "onset_by_B": onset_bp,
           "B_slope": (round(fitB["slope"], 3) if fitB else None),
           "B_ci95": (fitB["ci95"] if fitB else None),
           "pred_slope": 2.0})
    return payload


def cmd_slackbite(args):
    ngrid = SMOKE_NGRID if args.smoke else NGRID
    world0 = family_kcell(K_world=4, B_cls=5.0, loc_level=LOC, alpha=ALPHA)
    bstar = float(world0.meta["beta_star"])
    cand = [0.11, 0.055] if args.smoke else [0.16, 0.11, 0.08, 0.055, 0.04, 0.028, 0.02]
    slacks = [s for s in cand if bstar - s > 0.05]      # keep a meaningful floor
    by_s, req = {}, {}
    for s in slacks:
        specs = [(n, 4, 5.0, s, BIG_M, args.reps) for n in ngrid]
        rows = pmap(_cell_job, specs, n_jobs=args.jobs, desc=f"T1a slackbite s={s}")
        by_s[str(s)] = _series(rows)
        req[str(s)] = _onset(rows, "bprime_cert", thr=0.8)
    ok = [s for s in slacks if req[str(s)]]
    fit = (loglog_exponent(np.array(ok, float),
                           np.array([req[str(s)] for s in ok], float))
           if len(ok) >= 4 else None)
    payload = {
        "run": "R052", "family": "kcell", "K": 4, "B": 5.0, "m": BIG_M,
        "beta_star": bstar, "slacks": slacks, "required_n_bprime": req,
        "fit": fit, "success_band": [-2.3, -1.7],
        "pass": (bool(fit and -2.3 <= fit["ci95"][0] and fit["ci95"][1] <= -1.7)
                 if fit else False),
        "series_by_slack": by_s,
        "note": "B′ analogue of the oracle bite (R014): floor freed (m=16e6) so the labeled "
                "risk+nuisance source axis bites; predicts −2 like the labeled law. Onset at "
                "cert_freq≥0.8 (R014 convention).",
    }
    save_json("R052_t1a_slackbite.json", payload)
    print({"required_n": req, "slope": (round(fit["slope"], 3) if fit else None),
           "ci95": (fit["ci95"] if fit else None), "pass": payload["pass"]})
    return payload


def cmd_all(args):
    k = cmd_kenv(args)
    b = cmd_bsweep(args)
    s = cmd_slackbite(args)
    summary = {
        "run": "R050_R052_T1A_SUMMARY",
        "premium_K_slope": (k["premium_K_loglog"]["slope"]
                            if k["premium_K_loglog"] else None),
        "premium_K_ci95": (k["premium_K_loglog"]["ci95"]
                           if k["premium_K_loglog"] else None),
        "premium_K_ci_contains_1": k["premium_K_ci_contains_1"],
        "onset_B_slope": (b["onset_B_loglog"]["slope"] if b["onset_B_loglog"] else None),
        "onset_B_ci95": (b["onset_B_loglog"]["ci95"] if b["onset_B_loglog"] else None),
        "slackbite_slope": (s["fit"]["slope"] if s["fit"] else None),
        "slackbite_ci95": (s["fit"]["ci95"] if s["fit"] else None),
        "slackbite_pass_band": s["pass"],
        "onset_bprime_K4": k["onset_bprime"].get("4"),
        "onset_oracle_K4": k["onset_oracle"].get("4"),
        # NB: this is the >=0.5-onset ratio at K=4 (~113x); the paper's "64x" is the
        # SEPARATE full-certification onset ratio (16384 -> ~1.05M). Same envelope, two
        # thresholds -- not the same number. (experiment-audit run01 key-naming fix.)
        "premium_K4_half_onset_ratio": (
            None if not (k["onset_bprime"].get("4") and k["onset_oracle"].get("4"))
            else round(k["onset_bprime"]["4"] / k["onset_oracle"]["4"], 1)),
    }
    save_json("R050_R052_t1a_summary.json", summary)
    print("\n=== T1a SUMMARY ===")
    print(summary)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("kenv", "bsweep", "bsweep_tight", "slackbite", "all"):
        p = sub.add_parser(name)
        p.add_argument("--reps", type=int, default=600)
        p.add_argument("--jobs", type=int, default=48)
        p.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    {"kenv": cmd_kenv, "bsweep": cmd_bsweep, "bsweep_tight": cmd_bsweep_tight,
     "slackbite": cmd_slackbite, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
