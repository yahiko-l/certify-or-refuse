# Reproduction commands

Every command is run from `code/` so that `python -m experiments.…` resolves the package:

```bash
cd code
export PYTHONPATH="$PWD"        # so `experiments` is importable
```

Determinism: all synthetic runs are seeded (per-cell RNG via `experiments.generators.rng_for`,
master seed pinned in code: `experiments.generators.MASTER_SEED = 20260611`, per the
pre-registration record described in the paper's Appendix G). Each command writes its JSON
under `code/experiments/results/` (created on first run; abbreviated `results/` below) with
byte-stable numbers (Monte-Carlo means up to the registered repetition count). Result JSONs
are **not shipped** in this code + data release: the reference numbers live in the paper, and
the commands below regenerate them. The `--reps`/`--jobs` defaults below reproduce the *paper*
numbers; pass smaller `--reps` (or `--smoke`, where available) for a fast check.

> **Wall-clock** is given for the 8×CPU-heavy defaults; small-`--reps` smoke runs are ~1 min each.

---

## 0. Sanity (≈2 min, do this first)

```bash
python -m experiments.tests.test_certificate     # R003 — kernel formula + no-oracle-leak + budget tests (expect 11/11)
python -m experiments.run_smoke                   # end-to-end certify-or-refuse on a toy world → results/R004_smoke.json
```

---

## 1. Synthetic claims (CPU only)

### Claim 1 — Bite divergence (required-n vs slack slope −2.002, CI [−2.142, −1.862])
```bash
python -m experiments.run_m2_b1 bite              # → results/R014_bite_divergence.json
```

### Claim 2 — Two-axis functional form (consistency checks; theorem carries the claim)
```bash
python -m experiments.run_m3 axes                 # → results/R021_R022_axis_scaling.json
python -m experiments.run_m3 modelcmp             # → results/R023_model_comparison.json  (held-out surface CV R²=0.9202)
```

### Claim 3 — Phase-diagram contour (qualitative rate-shape only)
```bash
# 1) build the four B1 intensity grids (this is also the input to Claims 4 & 5)
for i in 1 2 3 4; do python -m experiments.run_m2_b1 grid --intensity $i; done   # → results/B1_grid_L{1..4}.json
# 2) contour analysis + the AUTHORITATIVE amended shape metrics
python -m experiments.analyze_b1                                                  # → results/R009_R012_contour_analysis.json
python -m experiments.derived_reports shape                                       # rewrites amended_shape_metrics (authoritative)
```

### Claim 4 — Localized geometry (B′ fam1 Spearman .936 vs global ESS .026)
```bash
for f in 1 2; do python -m experiments.run_m2_b2 build --fam $f; done             # → results/B2_pairs_fam{1,2}.json
for f in 1 2; do for a in bprime oracle_a; do \
    python -m experiments.run_m2_b2 sweep --fam $f --arm $a; done; done           # → results/B2_required_n_fam*_*.json
# rank correlations (localized vs global ESS) — the `analyze` subcommand is the R020
# producer; it hard-gates on the four sweeps above having been written first:
python -m experiments.run_m2_b2 analyze                                           # → results/R020_rank_correlations.json
# localized-envelope premium (TPAMI revision, supporting evidence):
python -m experiments.run_t1a all                                                 # → results/R050_t1a_envelope.json, R051_t1a_premium_B.json, R052_t1a_slackbite.json
# `all` runs kenv/bsweep/slackbite only; the tight-overlap B-sweep is a separate subcommand
# and is the sole producer of R051b:
python -m experiments.run_t1a bsweep_tight                                        # → results/R051b_t1a_premium_B_tight.json
```

### Claim 5 — Validity of formal arms + practical constants
```bash
# Full-grid validity aggregate (oracle-A 583/1024, B′ 8/1024, 0 violations, demo-arms violate at scale):
python -m experiments.derived_reports validity                                    # → results/R054_validity_aggregate.json  (consumes B1_grid_L{1..4})
# Deep audit subset (4000 reps):
python -m experiments.run_m2_b1 audit                                             # → results/R042_validity_audit.json
# B′ positive-certification EXISTENCE proof (K=4 — use `bprime_positive`, NOT `all`):
python -m experiments.run_t4 bprime_positive                                      # → results/R066_t4_bprime_positive.json
```

### Claim 5 (baseline) — Floor-augmented SCoRE e-value (arm 7), same-condition head-to-head vs B′
```bash
python -m experiments.run_r068_audit              # deep audit 6 cells × 4000 reps → results/R068_score_floor_audit.json
python -m experiments.run_r068_fullgrid           # 1024-cell matched-reps count   → results/R068_fullgrid_count.json
```

### Two-axis CI-separation (dense resolving run, §5 / Appendix G)
```bash
python -m experiments.run_t2 ci_separation --n-points 13 --reps 800 \
       --out-name R063_t2c_ci_separation_dense.json                               # → results/R063_t2c_ci_separation_dense.json
```

### Claim 7 (synthetic half) — Misspecification-to-breakage (within-cell w·η, ε≥0.8)
```bash
python -m experiments.run_t3 breakage                                             # → results/R064_t3_misspec_breakage.json
```

---

## 2. Real-LLM legs (Claims 6 & 7 — SQuAD→NewsQA)

The public package ships text-free frozen eval arrays (no result JSONs). That is enough to
regenerate the reported frontier maps without redistributing corpus text or model weights.
The refusal/violation tallies come from the full certificate resampling, which needs the
cached JSONL pools, so they belong to Path B.

### Path A — Zero-GPU frontier regeneration from the frozen eval arrays
The shipped `data/real_cache_leg2_deepseek/` and `data/real_cache_leg1_llama3/` directories
contain `_eval_target_frozen.npz`, `locks.json`, `manifest.json`, and the eval-access log. They
do **not** contain the full per-pool `*.jsonl` caches expected by `certify_real.py`, so no
`certify_real` command is listed for this path. `frontier_sweep` reads the frozen arrays
directly; stage them into the cache locations it expects, then run it:

```bash
# from code/ — stage the shipped frozen arrays where frontier_sweep looks for them:
mkdir -p experiments/real/cache_leg1_llama3_8b experiments/real/cache
cp ../data/real_cache_leg1_llama3/_eval_target_frozen.npz  experiments/real/cache_leg1_llama3_8b/
cp ../data/real_cache_leg2_deepseek/_eval_target_frozen.npz experiments/real/cache/
python -m experiments.real.frontier_sweep    # → results/T1bi_frontier_leg{1,2}.json

python - <<'PY'
import json
for name, frontier_path in [
    ("leg1", "experiments/results/T1bi_frontier_leg1.json"),
    ("leg2", "experiments/results/T1bi_frontier_leg2.json"),
]:
    frontier = json.load(open(frontier_path))
    a10 = next(row for row in frontier["headline"] if abs(row["alpha"] - 0.10) < 1e-12)
    print(f"{name}: beta_star(.10)={a10['beta_star']:.3f}, ci95={a10['ci95']}")
PY
```

This regenerates the paper's post-hoc frontier headline, β̂*(.10)=.050/.209 for leg 1/2,
against the pre-registered SLA floor β=0.60. The refusal/violation tallies reported in the
paper (1,000/1,000 B′ refusals per leg, 0 violations) come from the full certificate
resampling and are reproduced by `certify_real` after the Path B cache rebuild.

### Path B — Regenerate the cache from scratch (GPU; one-time, ≤2–4 GPU-days/leg)
**This is an illustrative outline, not a turnkey recipe**: it requires the external model
weights, `requirements-real.txt`, and days of GPU time; for the full CLIs run
`python -m experiments.real.cache_pipeline --help`, `python -m experiments.real.build_locks
--help`, and `python -m experiments.real.certify_real <subcommand> --help` (the bare
`--help` lists the subcommands; flags such as `--cache-dir` live on each subcommand). See `data/README_DATA.md` for the exact serving env (vLLM 0.22.1;
DeepSeek-V4-Flash needs `CUDA_HOME`, `NCCL_P2P_DISABLE=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`,
`kv_cache_dtype=fp8`).

Two constraints the outline must respect: (i) rebuild into a **fresh** `--cache-dir` —
`build_locks` refuses the protected frozen-leg staging names (`cache`,
`cache_leg1_llama3_8b`) via `assert_safe_cache`, and a fresh directory also avoids mixing
a rebuild with the frozen originals; (ii) one rebuild per leg, with leg 2 passing its own
`--deviation` record (R046/A2) so the manifest does not inherit leg 1's.

```bash
# leg 1 (Meta-Llama-3-8B-Instruct) — frozen pipeline, scores only, label-blind on target:
CUDA_VISIBLE_DEVICES=0 python -m experiments.real.cache_pipeline \
    --cache-dir cache_leg1_rebuild --tp 1 --model <PATH_TO_Meta-Llama-3-8B-Instruct>
# leg 2 (DeepSeek-V4-Flash):
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.real.cache_pipeline \
    --cache-dir cache_leg2_rebuild --tp 4 --model <PATH_TO_DeepSeek-V4-Flash> \
    --tokenizer-mode deepseek_v4 --deviation <R046_A2_RECORD> \
    --kv-cache-dtype fp8 --trust-remote-code --no-think --think-kwarg thinking --max-num-seqs 256
# then, per leg, point build_locks / certify_real at the same fresh cache dir
# (each takes --cache-dir; the REAL_CACHE_DIR env var also works).
```
Each cache step writes a `manifest.json` with sha256 of every pool and a 100-item
determinism check; compare it against the shipped manifest to confirm a faithful rebuild.

---

## 3. Claim → command → output → first-party number (one-glance map)

| Paper claim | Command (from `code/`) | Output JSON | Headline number |
|---|---|---|---|
| C1 Bite divergence | `run_m2_b1 bite` | `R014_bite_divergence.json` | slope −2.002, CI [−2.142,−1.862] |
| C1 Bite bootstrap CI | `python code/experiments/results/R014_bite_bootstrap_ci.py` (post-hoc; run after `run_m2_b1 bite`, reads `R014_bite_divergence.json` from the same results dir) | `R014_bite_bootstrap_ci.json` | 20,000-resample slope CI, seed 20260624 |
| C2 Two-axis form | `run_m3 axes` / `run_m3 modelcmp` | `R021_R022_axis_scaling.json`, `R023_model_comparison.json` | surface CV R²=0.9202 |
| C3 Contour | `run_m2_b1 grid` ×4 → `analyze_b1` → `derived_reports shape` | `R009_R012_contour_analysis.json` | L1 corr .997; held-out .981/.963/.890 |
| C4 Localized geometry | `run_m2_b2 build/sweep` → `run_m2_b2 analyze`; `run_t1a all` + `run_t1a bsweep_tight` | `R020_rank_correlations.json`, `B2_pairs_fam1.json`, `R051b_t1a_premium_B_tight.json` | Spearman .936 vs global ESS .026 |
| C5 Validity grid | `derived_reports validity`, `run_m2_b1 audit` | `R054_validity_aggregate.json`, `R042_validity_audit.json` | oracle-A 583/1024, B′ 8/1024, 0 violations |
| C5 B′ existence | `run_t4 bprime_positive` | `R066_t4_bprime_positive.json` | cert_freq→1.0 at n=m=262,144 (s=.20) |
| C5 SCoRE baseline | `run_r068_audit`, `run_r068_fullgrid` | `R068_score_floor_audit.json`, `R068_fullgrid_count.json` | 0/24000 viol; ≥ B′ in 1024/1024, > in 8 |
| C2/§5 CI-separation | `run_t2 ci_separation --n-points 13 --reps 800 --out-name R063_…` | `R063_t2c_ci_separation_dense.json` | dense two-axis separation |
| C6 Real audit | frontier regeneration from frozen arrays (Path A: `real.frontier_sweep`); refusal tallies need the full-cache rerun (Path B: `real.certify_real {main,ladder,stability}`) | `T1bi_frontier_leg{1,2}.json`; Path B adds `R033_R045_*`, `R034_*` | 2000/2000 refusals, 0 viol; β̂*=.050 / .209 |
| C7 Refusal robustness | `derived_reports sensitivity` (K/B/jitter grid, leg-2 cache), `run_t3 breakage` | `R053_assumption_sensitivity.json`, `R064_t3_misspec_breakage.json` | 11/11 no-cert; breakage at ε≥0.8 |
| C7 Lattice stability (real) | `real.certify_real stability` | `R035_lattice_stability.json` | jitter + second parameterization |

> **Do not run** `run_t4 all`/`run_t4 floorbite` or `run_t1a` floor-bite variants for paper
> reproduction: those regenerate exploratory runs (R065, archived) that **do not** back any
> paper claim. Use the exact subcommands above.
