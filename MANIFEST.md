# Package manifest — what ships, what does not, and why

This is the release tree for the paper: a **code + data** artifact. It is assembled from the
working repository by whitelisting the paths below, removing local machine paths/model
directories, and emitting `CHECKSUMS.sha256`. Result JSONs, figure scripts, and
pre-registration documents are deliberately not shipped: the paper itself presents every
reported number and figure, and the commands in `REPRODUCE.md` regenerate the results from
the code and data here. The one exception is `code/experiments/results/R014_bite_bootstrap_ci.py`,
a post-hoc analysis script that ships because a reported interval depends on it; the JSON it
consumes and the JSON it writes are still regenerated, not shipped.

```
floor-certification-artifact/
├── README.md                         # landing page
├── REPRODUCE.md                      # commands → outputs → claims (Path B: outline)
├── MANIFEST.md                       # this file
├── LICENSE                           # Apache License 2.0 (full text)
├── NOTICE                            # attribution
├── LICENSE-NOTE.md                   # code (Apache-2.0) vs third-party data licensing
├── requirements.txt                  # synthetic-run deps (pinned)
├── requirements-real.txt             # real-LLM caching deps (vLLM)
├── CHECKSUMS.sha256                  # sha256 of every shipped file (generated)
│
├── code/
│   └── experiments/
│       ├── __init__.py
│       ├── arms.py                    # the 7 arms incl. arm_score_floor (SCoRE+floor)
│       ├── certificate.py            # UCB/LCB, budget_split, score_risk_pass_evalue
│       ├── validity.py               # truth-level violation judge, ARM_NAMES, run_cell
│       ├── generators.py             # worlds, lattices, cells, rng_for (MASTER_SEED pinned)
│       ├── analysis.py               # B1 contour protocol helpers
│       ├── analyze_b1.py             # → R009_R012_contour_analysis.json
│       ├── derived_reports.py        # shape (R009_R012) + validity (R054) aggregates
│       ├── runner_util.py            # save_json, pmap, log_grid
│       ├── run_m1.py                 # M1 baseline
│       ├── run_m2_b1.py              # B1 grids / ladder / bite / edge / audit
│       ├── run_m2_b2.py              # B2 pair build / required-n sweep
│       ├── run_m3.py                 # axes / modelcmp / beta0 / ksweep / lattice / misspec / nontheorem
│       ├── run_t1a.py                # localized-envelope premium (kenv/bsweep/bsweep_tight/slackbite/all)
│       ├── results/
│       │   └── R014_bite_bootstrap_ci.py   # post-hoc bootstrap CI on the R014 bite slope;
│       │                                   # runs write their JSONs into this same directory
│       ├── run_t2.py                 # bite_families / bite_geometry / contour_drift / ci_separation
│       ├── run_t3.py                 # misspecification-to-breakage (R064)
│       ├── run_t4.py                 # bprime_positive (R066) — see exclusion note on floorbite/R065
│       ├── run_r068_audit.py         # SCoRE+floor deep audit
│       ├── run_r068_fullgrid.py      # SCoRE+floor 1024-cell count
│       ├── run_smoke.py              # end-to-end smoke
│       ├── tests/
│       │   ├── __init__.py
│       │   └── test_certificate.py   # R003 — formula pinning, no-oracle-leak, budget
│       └── real/
│           ├── __init__.py
│           ├── data.py               # R002 corpus handling (SQuAD→NewsQA), EvalGuard
│           ├── cache_pipeline.py     # GPU frozen-pipeline cache (Path B)
│           ├── build_locks.py        # λ-lattice + K=16 cells from D_lock
│           ├── certify_real.py       # R033/R034/R035/R045 certification
│           └── frontier_sweep.py     # post-hoc β̂*(α) map from the frozen eval arrays
│
└── data/
    ├── README_DATA.md                # availability + corpus licensing + regeneration
    ├── export_scores_only.py         # per-pool scores-only export helper (option a)
    ├── real_cache_leg2_deepseek/     # text-free frozen leg-2 eval scores+losses
    │   ├── _eval_target_frozen.npz   # eval-set scores s + 0/1 losses L (no text)
    │   ├── locks.json                # λ-lattice + K-cell locks (numeric)
    │   ├── manifest.json             # source-cache sha256, model snapshot hash, determinism check
    │   └── _eval_access.log          # access-guard record (single line)
    └── real_cache_leg1_llama3/       # text-free frozen leg-1 eval scores+losses
        ├── _eval_target_frozen.npz
        ├── locks.json
        ├── manifest.json
        └── _eval_access.log
```

## Intentionally EXCLUDED (and why)

| Excluded | Reason |
|---|---|
| Result JSONs (synthetic and real) | The paper presents every reported number. `REPRODUCE.md` regenerates the synthetic ones from pinned seeds and the real frontier maps from the shipped frozen arrays (Path A); the real refusal tallies require the Path B cache rebuild from external model weights. Not shipping them keeps the artifact code + data only. |
| Figure/table generator scripts | The paper presents the figures; the generators read only regenerated result JSONs and hard-code no numbers. |
| Pre-registration documents | Protocol provenance is documented in the paper (Appendix G); the operative locks are pinned in the release itself: the synthetic master seed in code (`experiments.generators.MASTER_SEED = 20260611`) and the real-leg numeric locks (λ-lattice, K cells) in the shipped `data/*/locks.json`. |
| Model weights (Meta-Llama-3-8B-Instruct, DeepSeek-V4-Flash) | External; redistribute via Hugging Face / ModelScope, not here. Provenance sha256 in each cache `manifest.json`. |
| `experiments/real/data/triviaqa_train.jsonl.gz` (341 MB) | Belongs to an exploratory dead-end (R048); never used by a paper claim; exceeds host push limits. |
| `experiments/archive/{R048,R065,R067}/` and their result JSONs | Exploratory runs that **back no paper claim** (archived/paused). Omitted to keep the artifact in 1:1 correspondence with the paper. |
| `R048_second_workload_positive_cert.md` prereg | Documents the excluded R048 dead-end only. |
| `quarantine_qwen36_partial/` | Partial Qwen3.6 outputs, never inspected; not a result source. |
| `__pycache__/`, `*.pyc`, editor/IDE files | Build noise. |
| Review/audit scratch (`PAPER_REVIEW_*`, `PROOF_AUDIT_*`, `*.review.json`, `refine-logs/`) | Internal review provenance, not reproduction inputs. |
| Repo home paths `/<home>/<user>/…` and local model dirs | Scrubbed from public-facing manifests and provenance records. |
| Staging scripts/checklists | Author-side only; not needed to reproduce the paper claims. |

## Size budget

- Code + docs: well under 1 MB.
- With the text-free frozen eval artifacts: **≈0.6 MB total** → fits a public repository.
  Full JSONL caches or per-pool score exports should be hosted separately if redistributed.

## Note on free text

The numbers-only stance (no corpus redistribution) holds: **no `golds` (corpus gold answers),
questions, passages, generated answers, or full JSONL pools are shipped.** The frozen eval
arrays carry scores and 0/1 losses only. See `data/README_DATA.md` for the real-cache options.

## Integrity

- `CHECKSUMS.sha256` pins every other shipped file (all files except itself). Each real artifact directory additionally carries
  its own `manifest.json` with source-cache sha256 records, the model snapshot hash, and a
  100-item determinism check.
- The synthetic numbers are reproducible from seeds alone (master seed pinned in
  `experiments.generators`); no shipped JSON is needed to regenerate them.
