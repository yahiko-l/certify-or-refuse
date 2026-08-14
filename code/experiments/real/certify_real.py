"""B3 certification on cached frozen-pipeline outputs (CPU; R033/R034/R035/R045).

Registered protocol (R001/R002/R039): (α, β, δ) = (0.10, 0.60, 0.05); B = 10; K = 16;
|Λ| = 64 (locks.json from D_lock only). 1000 calibration resamples (bootstrap WITHIN
each fixed split — L1), all-resample reporting; L2 = bootstrap over eval items at the
single final-eval step (EvalGuard logs the one label load). Ladder points are
dependent (same corpus) — per-point reporting, never pooled.

Subcommands:
  python -m experiments.real.certify_real main      (R033 + R045: all four arms)
  python -m experiments.real.certify_real ladder    (R034: no-cert onset)
  python -m experiments.real.certify_real stability (R035: jitter + 2nd parameterization)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..certificate import certify_bprime_samples, bernstein_lcb
from .data import correct, load_eval_golds_guarded
from .cache_pipeline import CACHE_DIR

N_RESAMPLES = 1000
SEED = 20260611
ALPHA, BETA, DELTA = 0.10, 0.60, 0.05
BETA_LADDER = [0.30, 0.40, 0.50, 0.60]   # R048 §5 capacity tiers (H/T = .70/.60/.50/.40)


def _load(name):
    return [json.loads(l) for l in open(os.path.join(CACHE_DIR, f"{name}.jsonl"))]


def _locks():
    locks = json.load(open(os.path.join(CACHE_DIR, "locks.json")))
    manifest = json.load(open(os.path.join(CACHE_DIR, "manifest.json")))
    if manifest.get("max_items"):
        raise RuntimeError("cache built with --max-items (debug cache) — refuse "
                           "to certify (review round-1 MAJOR fix)")
    if manifest.get("determinism_check_100") is not True:
        raise RuntimeError("cache determinism check not recorded True — refuse "
                           "to certify (review round-2 fix)")
    return locks


def _source_arrays(rows):
    s = np.array([r["score"] for r in rows], float)
    l = np.array([1 - correct(r["answer"], r["golds"]) for r in rows], int)
    return s, l


def _target_scores(rows):
    return np.array([r["score"] for r in rows], float)


def arms_on_resample(s_wP, s_r, l_r, s_wQ, s_f, locks):
    """All four real-data arms on one calibration resample. Returns per-arm
    (chosen_threshold_index or -1, claimed_pair: bool)."""
    th = np.array(locks["lattice_thresholds"])
    edges = np.array(locks["cell_edges"])
    B = locks["B"]
    res = certify_bprime_samples(s_wP, s_r, l_r, s_wQ, s_f, th, edges,
                                 B, ALPHA, BETA, DELTA)
    out = {"bprime": {"chosen": int(res["chosen"]),
                      "claimed": res["chosen"] >= 0}}
    # floor-free + post-hoc point check (frozen M1 definition): max acceptance
    # (lowest threshold) among risk-passing; claim iff point coverage >= beta
    rp = res["risk_pass"]
    if rp.any():
        j = int(np.flatnonzero(rp).min())            # lowest threshold = max acceptance
        claimed = bool(res["p_f_hat"][j] >= BETA)
        out["floorfree_posthoc"] = {"chosen": j, "claimed": claimed}
    else:
        out["floorfree_posthoc"] = {"chosen": -1, "claimed": False}
    # weighted-conformal: global (non-localized) nuisance, risk-only; coverage reported
    rho_glob = float(((res["p_hat"] + res["e_p"]) * res["err"]).sum())
    rp_g = res["ucb_x"] <= -rho_glob
    if rp_g.any():
        j = int(np.flatnonzero(rp_g).min())
        out["weighted_conformal"] = {"chosen": j, "claimed": False,
                                     "coverage_report": float(res["p_f_hat"][j])}
    else:
        out["weighted_conformal"] = {"chosen": -1, "claimed": False}
    # naive plug-in: point estimates only (uses ŵ cells from the same resample)
    cell_r = np.digitize(s_r, edges)
    w_r = res["w_hat"][cell_r]
    num = np.array([(w_r * (s_r >= t) * l_r).mean() for t in th])
    den = np.array([(w_r * (s_r >= t)).mean() for t in th])
    with np.errstate(invalid="ignore", divide="ignore"):
        r_hat = np.where(den > 0, num / den, 0.0)
    ok = (r_hat <= ALPHA) & (res["p_f_hat"] >= BETA)
    if ok.any():
        j = int(np.flatnonzero(ok).min())
        out["plugin"] = {"chosen": j, "claimed": True}
    else:
        out["plugin"] = {"chosen": -1, "claimed": False}
    return out


def eval_realized(eval_scores, eval_L, t):
    sel = eval_scores >= t
    cov = float(sel.mean())
    risk = float(eval_L[sel].mean()) if sel.any() else 0.0
    return cov, risk


def load_eval(side="target", caller="R033_final_eval", slice_name=None):
    """THE single raw eval-label access (hard-guarded; review round-1 CRITICAL fix).
    On first call: reads raw dev golds through load_eval_golds_guarded (R048: filtered
    to slice_name at the barrier), scores the cached answers, and freezes (scores, L) to
    an artifact. Later steps reuse the frozen artifact — they never re-touch raw labels."""
    manifest = json.load(open(os.path.join(CACHE_DIR, "manifest.json")))
    pool_sha = manifest["pools"][f"eval_{side}"]["sha256"]
    suffix = "" if slice_name is None else f"_{slice_name}"   # slice-specific artifact
    art = os.path.join(CACHE_DIR, f"_eval_{side}{suffix}_frozen.npz")
    if os.path.exists(art):
        z = np.load(art, allow_pickle=False)
        art_slice = str(z["slice"]) if "slice" in z.files else "None"   # legacy npz = whole-dev
        if str(z["pool_sha"]) != pool_sha or art_slice != str(slice_name):
            raise RuntimeError("stale/mismatched frozen eval artifact (cache or slice "
                               "mismatch) — rerun final eval on the current cache")
        return z["s"], z["L"]
    rows = _load(f"eval_{side}")
    golds = load_eval_golds_guarded(side, caller, CACHE_DIR, slice_name=slice_name)
    s = np.array([r["score"] for r in rows], float)
    L = np.array([1 - correct(r["answer"], golds[r["qid"]]) for r in rows], int)
    np.savez(art, s=s, L=L, pool_sha=np.str_(pool_sha), slice=np.str_(str(slice_name)))
    return s, L


def cmd_main(args):
    """R033 + R045: 1000 resamples, all four arms, two-level uncertainty,
    all-resample reporting."""
    locks = _locks()
    th = np.array(locks["lattice_thresholds"])
    src_wP = _load("src_wP"); src_r = _load("src_r")
    s_wP0, _ = _source_arrays(src_wP)
    s_r0, l_r0 = _source_arrays(src_r)
    tgt = _target_scores(_load("tgt_unlabeled"))
    rng = np.random.default_rng(SEED)
    half = len(tgt) // 2
    perm = rng.permutation(len(tgt))
    s_wQ0, s_f0 = tgt[perm[:half]], tgt[perm[half:]]

    per_arm = {a: [] for a in ("bprime", "floorfree_posthoc", "weighted_conformal",
                               "plugin")}
    for i in range(args.resamples):
        rr = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(1000, i)))
        bs = lambda x: x[rr.integers(0, len(x), len(x))]
        idx_r = rr.integers(0, len(s_r0), len(s_r0))
        res = arms_on_resample(bs(s_wP0), s_r0[idx_r], l_r0[idx_r],
                               bs(s_wQ0), bs(s_f0), locks)
        for a, v in res.items():
            per_arm[a].append(v)
        if (i + 1) % 200 == 0:
            print(f"[resample] {i+1}/{args.resamples}")

    # FINAL EVAL (single guarded label load)
    ev_s, ev_L = load_eval("target", caller="R033_final_eval")
    out = {"run": "R033_R045", "alpha": ALPHA, "beta": BETA, "delta": DELTA,
           "delta_tol": 1.25 * DELTA, "resamples": args.resamples, "arms": {}}
    rng2 = np.random.default_rng(SEED + 7)
    for a, rows in per_arm.items():
        claimed = np.array([r["claimed"] for r in rows])
        chosen = np.array([r["chosen"] for r in rows])
        realized = np.array([eval_realized(ev_s, ev_L, th[c]) if c >= 0 else (0.0, 0.0)
                             for c in chosen])
        viol = claimed & ((realized[:, 1] > ALPHA) | (realized[:, 0] < BETA))
        k = int(viol.sum()); nrs = len(rows)
        from ..validity import cp_ucb, cp_lcb
        rec = {"claim_rate": float(claimed.mean()),
               "no_cert_rate": float((chosen < 0).mean()),
               "viol_count": k, "viol_cp_ucb_L1": cp_ucb(k, nrs),
               "lambda_hist": np.bincount(chosen[chosen >= 0],
                                          minlength=len(th)).tolist(),
               "all_resample_reporting": True}
        sel = chosen >= 0
        if sel.any():                       # realized behavior at SELECTED thresholds
            rk = realized[sel, 1]
            rec["selected_realized_risk_mean"] = float(rk.mean())
            rec["selected_risk_violations"] = int((rk > ALPHA).sum())
            rec["selected_risk_viol_cp_ucb"] = cp_ucb(int((rk > ALPHA).sum()), nrs)
        if claimed.any():
            cov_d, risk_d = realized[claimed, 0], realized[claimed, 1]
            rec.update(realized_cov_mean=float(cov_d.mean()),
                       realized_cov_p5=float(np.percentile(cov_d, 5)),
                       realized_risk_mean=float(risk_d.mean()),
                       realized_risk_p95=float(np.percentile(risk_d, 95)))
            # L2: label bootstrap at the modal chosen lambda
            jmod = int(np.bincount(chosen[claimed]).argmax())
            t = th[jmod]
            covs, risks = [], []
            for _ in range(1000):
                bidx = rng2.integers(0, len(ev_s), len(ev_s))
                c, r = eval_realized(ev_s[bidx], ev_L[bidx], t)
                covs.append(c); risks.append(r)
            rec["L2_label_bootstrap_at_modal_lambda"] = {
                "cov_ci": [float(np.percentile(covs, 2.5)), float(np.percentile(covs, 97.5))],
                "risk_ci": [float(np.percentile(risks, 2.5)), float(np.percentile(risks, 97.5))],
                "modal_threshold": float(t)}
        out["arms"][a] = rec
    # decision difference (B3 headline): certified vs floor-free chosen thresholds
    def _modal_claimed(arm):
        ch = [r["chosen"] for r in per_arm[arm] if r["claimed"]]
        return int(np.bincount(ch).argmax()) if ch else None
    bp_modal = _modal_claimed("bprime")
    ff_modal = _modal_claimed("floorfree_posthoc")   # CLAIMED only (review fix)
    out["decision_difference"] = {"bprime_modal_lambda": bp_modal,
                                  "floorfree_modal_lambda": ff_modal,
                                  "differ": (bp_modal != ff_modal),
                                  "note": "modal lambda over CLAIMING resamples; "
                                          "None = arm never claimed"}
    _save("R033_R045_certification.json", out)


def cmd_ladder(args):
    """R034: no-cert onset vs shift intensity (mixtures from cached pools);
    per-point reporting — points are dependent (same corpus), never pooled."""
    locks = _locks()
    src_wP = _load("src_wP"); src_r = _load("src_r")
    s_wP0, _ = _source_arrays(src_wP)
    s_r0, l_r0 = _source_arrays(src_r)
    tgt = _target_scores(_load("tgt_unlabeled"))
    src_u = _target_scores(_load("src_unlabeled_ladder"))
    rng = np.random.default_rng(SEED + 1)
    out = {"run": "R034", "points": []}
    for iota in (0.0, 0.25, 0.5, 0.75, 1.0):
        n_mix = 30000
        k_t = int(iota * n_mix)
        mix = np.concatenate([rng.choice(tgt, k_t, replace=True),
                              rng.choice(src_u, n_mix - k_t, replace=True)])
        rng.shuffle(mix)
        s_wQ, s_f = mix[: n_mix // 2], mix[n_mix // 2:]
        claims = 0; chosen_hist = []
        for i in range(args.resamples):
            rr = np.random.default_rng(np.random.SeedSequence(
                SEED, spawn_key=(2000, int(iota * 100), i)))
            bs = lambda x: x[rr.integers(0, len(x), len(x))]
            idx_r = rr.integers(0, len(s_r0), len(s_r0))
            res = certify_bprime_samples(bs(s_wP0), s_r0[idx_r], l_r0[idx_r],
                                         bs(s_wQ), bs(s_f),
                                         np.array(locks["lattice_thresholds"]),
                                         np.array(locks["cell_edges"]),
                                         locks["B"], ALPHA, BETA, DELTA)
            claims += res["chosen"] >= 0
            chosen_hist.append(int(res["chosen"]))
        out["points"].append({"iota": iota, "cert_rate": claims / args.resamples,
                              "no_cert_rate": 1 - claims / args.resamples,
                              "note": "per-point; dependent across ladder"})
        print(f"[ladder] iota={iota}: cert_rate={claims/args.resamples:.3f}")
    _save("R034_nocert_onset.json", out)


def cmd_stability(args):
    """R035: lattice jitter (± half-step) + uniform-threshold second parameterization.
    Stability metric: certify/no-cert status unchanged AND λ̂ within one step AND
    realized (R_Q, C_Q) point estimates within each other's eval CIs."""
    locks = _locks()
    th0 = np.array(locks["lattice_thresholds"])
    half_step = np.diff(th0).mean() / 2.0
    variants = {"primary": th0,
                "jitter_up": th0 + half_step, "jitter_down": th0 - half_step,
                "uniform_param": np.array(locks["uniform_lattice"])}
    src_wP = _load("src_wP"); src_r = _load("src_r")
    s_wP0, _ = _source_arrays(src_wP)
    s_r0, l_r0 = _source_arrays(src_r)
    tgt = _target_scores(_load("tgt_unlabeled"))
    rng = np.random.default_rng(SEED + 2)
    half = len(tgt) // 2
    perm = rng.permutation(len(tgt))
    s_wQ0, s_f0 = tgt[perm[:half]], tgt[perm[half:]]
    ev_s, ev_L = load_eval("target", caller="R035_stability_eval")
    out = {"run": "R035", "variants": {}}
    base = None
    for name, th in sorted(variants.items()):
        res = certify_bprime_samples(s_wP0, s_r0, l_r0, s_wQ0, s_f0, th,
                                     np.array(locks["cell_edges"]),
                                     locks["B"], ALPHA, BETA, DELTA)
        j = int(res["chosen"])
        rec = {"status": "cert" if j >= 0 else "no-cert"}
        if j >= 0:
            cov, risk = eval_realized(ev_s, ev_L, th[j])
            rec.update(threshold=float(th[j]), realized_cov=cov, realized_risk=risk)
        out["variants"][name] = rec
        if name == "primary":
            base = rec
    same_status = all(v["status"] == base["status"] for v in out["variants"].values())
    out["stability"] = {"status_unchanged": bool(same_status)}
    if base["status"] == "cert":
        ths = [v.get("threshold") for v in out["variants"].values() if "threshold" in v]
        out["stability"]["lambda_within_one_step"] = bool(
            max(ths) - min(ths) <= 2.2 * half_step)
        covs = [v["realized_cov"] for v in out["variants"].values() if "realized_cov" in v]
        risks = [v["realized_risk"] for v in out["variants"].values() if "realized_risk" in v]
        n_ev = len(ev_s)
        se_c = float(np.sqrt(0.25 / n_ev)) * 1.96 * 2
        out["stability"]["rc_within_eval_cis"] = bool(
            (max(covs) - min(covs) <= se_c) and (max(risks) - min(risks) <= se_c))
    _save("R035_lattice_stability.json", out)


def _pilot_frontier():
    """R048 §7: beta_hat*(alpha) on the DISJOINT pilot slice (labels allowed; descriptive
    ONLY — confirms the registered beta-ladder brackets the frontier; never sets a tier,
    never enters the certificate). Same estimator as frontier_sweep.betastar."""
    rows = _load("pilot_target")
    golds = load_eval_golds_guarded("target", "R048_pilot", CACHE_DIR, slice_name="pilot")
    s = np.array([r["score"] for r in rows], float)
    L = np.array([1 - correct(r["answer"], golds[r["qid"]]) for r in rows], int)
    order = np.argsort(-s); Ls = L[order]; k = np.arange(1, len(s) + 1)
    risk = np.cumsum(Ls) / k; cov = k / len(s)
    ok = risk <= ALPHA
    bstar = float(cov[ok].max()) if ok.any() else 0.0
    lo, hi = min(BETA_LADDER), max(BETA_LADDER)
    return {"alpha": ALPHA, "beta_star_pilot": round(bstar, 4), "n_pilot": int(len(s)),
            "mean_loss_pilot": round(float(L.mean()), 4),
            "frontier_above_min_tier": bool(bstar >= lo),   # >= .30: lowest tier feasible-side
            "frontier_below_max_tier": bool(bstar < hi),    # <  .60: highest tier infeasible-side
            "brackets_ladder": bool(lo <= bstar < hi),      # spans feasible AND infeasible tiers
            "note": "pilot-slice diagnostic (best achievable selective point); NOT a "
                    "certificate, does not set any beta tier (R048 §7)"}


def cmd_workload2(args):
    """R048: SQuAD->TriviaQA positive-certification probe. B' certificate over the
    registered 4-tier capacity beta-ladder; pilot frontier on the disjoint pilot slice;
    eval touched once. Per-tier reporting (dependent ladder, never pooled)."""
    from ..validity import cp_ucb
    from .data import assert_safe_cache
    # --- preflight: fail BEFORE any resample or label access unless this is the
    # registered R048 cache (R048 §11/§12; codex re-audit CRITICAL #4) ---
    assert_safe_cache(CACHE_DIR)
    manifest = json.load(open(os.path.join(CACHE_DIR, "manifest.json")))
    if not manifest.get("pilot_split"):
        raise SystemExit("workload2 requires an R048 pilot-split cache")
    if manifest.get("target") != "triviaqa":
        raise SystemExit(f"workload2 expects target=triviaqa, got {manifest.get('target')!r}")
    if manifest.get("max_items"):
        raise SystemExit("workload2 refuses a --max-items debug cache")
    if "DeepSeek-V4-Flash" not in str(manifest.get("model", "")):
        raise SystemExit(f"workload2 expects the registered DeepSeek-V4-Flash model, "
                         f"got {manifest.get('model')!r}")
    if manifest.get("serving_config", {}).get("tokenizer_mode") != "deepseek_v4":
        raise SystemExit("workload2 expects the registered DSV4 serving config "
                         f"(tokenizer_mode=deepseek_v4), got {manifest.get('serving_config')!r}")
    cnt = manifest.get("counts", {})
    if cnt.get("eval_target_slice", 0) < 1000 or cnt.get("pilot_target_slice", 0) < 1000:
        raise SystemExit("workload2: eval and pilot slices must each be >= 1000 (R048 §6)")
    for pool in ("src_wP", "src_r", "tgt_unlabeled", "pilot_target", "eval_target"):
        if not os.path.exists(os.path.join(CACHE_DIR, f"{pool}.jsonl")):
            raise SystemExit(f"workload2 missing required pool {pool}.jsonl")
    locks = _locks()
    th = np.array(locks["lattice_thresholds"]); edges = np.array(locks["cell_edges"])
    B = locks["B"]
    s_wP0, _ = _source_arrays(_load("src_wP"))
    s_r0, l_r0 = _source_arrays(_load("src_r"))
    tgt = _target_scores(_load("tgt_unlabeled"))
    rng = np.random.default_rng(SEED); half = len(tgt) // 2
    perm = rng.permutation(len(tgt)); s_wQ0, s_f0 = tgt[perm[:half]], tgt[perm[half:]]

    chosen = {b: [] for b in BETA_LADDER}
    for i in range(args.resamples):
        rr = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(48, i)))
        bs = lambda x: x[rr.integers(0, len(x), len(x))]
        idx_r = rr.integers(0, len(s_r0), len(s_r0))
        a_wP, a_wQ, a_f, a_r, a_l = bs(s_wP0), bs(s_wQ0), bs(s_f0), s_r0[idx_r], l_r0[idx_r]
        for b in BETA_LADDER:                          # frozen certificate, different beta only
            res = certify_bprime_samples(a_wP, a_r, a_l, a_wQ, a_f, th, edges,
                                         B, ALPHA, b, DELTA)
            chosen[b].append(int(res["chosen"]))
        if (i + 1) % 200 == 0:
            print(f"[w2 resample] {i+1}/{args.resamples}")

    pilot = _pilot_frontier()                          # disjoint pilot slice (descriptive)
    ev_s, ev_L = load_eval("target", caller="R048_final_eval", slice_name="eval")  # ONCE
    out = {"run": "R048_workload2", "alpha": ALPHA, "delta": DELTA,
           "target": manifest.get("target"), "model": manifest.get("model"),
           "beta_ladder": BETA_LADDER, "resamples": args.resamples,
           "n_eval_slice": int(len(ev_s)), "pilot_frontier": pilot, "tiers": {}}
    for b in BETA_LADDER:
        ch = np.array(chosen[b]); claimed = ch >= 0
        realized = np.array([eval_realized(ev_s, ev_L, th[c]) if c >= 0 else (0.0, 0.0)
                             for c in ch])
        viol = claimed & ((realized[:, 1] > ALPHA) | (realized[:, 0] < b))
        k = int(viol.sum())
        rec = {"beta": b, "claim_rate": float(claimed.mean()),
               "no_cert_rate": float((~claimed).mean()),
               "viol_count": k, "viol_cp_ucb": cp_ucb(k, len(ch)),
               "lambda_hist": (np.bincount(ch[claimed], minlength=len(th)).tolist()
                               if claimed.any() else [])}
        if claimed.any():
            cov_d, risk_d = realized[claimed, 0], realized[claimed, 1]
            rec.update(realized_cov_mean=float(cov_d.mean()),
                       realized_cov_p5=float(np.percentile(cov_d, 5)),
                       realized_risk_mean=float(risk_d.mean()),
                       realized_risk_p95=float(np.percentile(risk_d, 95)))
            # L2: eval-item bootstrap at the modal claimed lambda (R048 §10)
            jmod = int(np.bincount(ch[claimed]).argmax()); t = th[jmod]
            rng2 = np.random.default_rng(SEED + 7 + int(round(b * 100)))
            covs, risks = [], []
            for _ in range(1000):
                bidx = rng2.integers(0, len(ev_s), len(ev_s))
                c, r = eval_realized(ev_s[bidx], ev_L[bidx], t)
                covs.append(c); risks.append(r)
            rec["L2_at_modal_lambda"] = {
                "modal_threshold": float(t),
                "cov_ci": [float(np.percentile(covs, 2.5)), float(np.percentile(covs, 97.5))],
                "risk_ci": [float(np.percentile(risks, 2.5)), float(np.percentile(risks, 97.5))]}
        # positive cert at this tier: certifies in a substantial fraction AND the
        # certified pairs hold on untouched eval (0 violations). Per-tier, never pooled.
        rec["positive_cert"] = bool(claimed.mean() >= 0.5 and k == 0)
        out["tiers"][f"{b:.2f}"] = rec
    _save("R048_workload2_certification.json", out)
    print({"pilot_beta_star": pilot["beta_star_pilot"],
           "tiers": {b: (round(out["tiers"][f"{b:.2f}"]["claim_rate"], 3),
                         out["tiers"][f"{b:.2f}"]["viol_count"],
                         out["tiers"][f"{b:.2f}"]["positive_cert"]) for b in BETA_LADDER}})


def _save(name, payload):
    from ..runner_util import save_json
    save_json(name, payload)


def main():
    global CACHE_DIR
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _cd(p):
        p.add_argument("--cache-dir", dest="cache_dir", default=None,
                       help="cache dir (R048 second workload); overrides $REAL_CACHE_DIR")
    m = sub.add_parser("main"); m.add_argument("--resamples", type=int, default=N_RESAMPLES); _cd(m)
    l = sub.add_parser("ladder"); l.add_argument("--resamples", type=int, default=300); _cd(l)
    _cd(sub.add_parser("stability"))
    w2 = sub.add_parser("workload2"); w2.add_argument("--resamples", type=int, default=N_RESAMPLES); _cd(w2)
    args = ap.parse_args()
    if getattr(args, "cache_dir", None):
        CACHE_DIR = os.path.abspath(args.cache_dir)
    {"main": cmd_main, "ladder": cmd_ladder, "stability": cmd_stability,
     "workload2": cmd_workload2}[args.cmd](args)


if __name__ == "__main__":
    main()
