# Real-data availability, licensing, and regeneration

The real-LLM legs use a **covariate-shift** audit: source = **SQuAD** (MRQA format),
target = **NewsQA**. Two capability legs share the corpus:

| Leg | Model | Native precision | β̂*(.10) | Role |
|---|---|---|---|---|
| 1 | Meta-Llama-3-8B-Instruct | bf16 | .050 [.035, .130] | primary |
| 2 | DeepSeek-V4-Flash | FP8-block e4m3 + FP4-expert | .209 [.160, .281] | conditional capability-ladder re-entry |

Both legs sit far below the floor β = .60 → the honest **frontier-infeasible** map.
(α, β, δ) = (0.10, 0.60, 0.05); B = 10; K = 16; |Λ| = 64. Seed 20260611.

---

## 1. What the shipped frozen artifacts contain (and do NOT)

`real_cache_leg2_deepseek/` and `real_cache_leg1_llama3/` each contain the text-free
frozen eval artifact `_eval_target_frozen.npz` (`s`: model score, `L`: 0/1 loss),
`locks.json`, `manifest.json`, and `_eval_access.log`. These files are enough to
regenerate the post-hoc eval-set frontier maps (`T1bi_frontier_leg{1,2}.json`) zero-GPU;
see `REPRODUCE.md` Path A.

They do **not** contain model weights, corpus text, generated answers, gold answers, or
the full per-pool `*.jsonl` caches (`src_lock`, `src_wP`, `src_r`,
`src_unlabeled_ladder`, `tgt_unlabeled`, `eval_target`, `eval_source`). The current
`certify_real.py` and `build_locks.py` commands expect those full JSONL pools, so an
end-to-end real certificate rerun requires regenerating the full cache from external
model weights (see `REPRODUCE.md` §2 Path B) or adding a loader for per-pool scores-only
exports if you produce them.

## 2. Corpus licensing — redistribution option

> **This release ships option (a): numbers-only eval artifacts.** No `answer`, `golds`,
> question, or passage text is redistributed, so neither the SQuAD nor the NewsQA corpus
> license is triggered. To regenerate the full text cache, use Path B (`REPRODUCE.md` §2)
> with the external model weights. The redistribution options are documented below for
> completeness.

| Option | What you ship in `data/` | Pros / cons |
|---|---|---|
| **(a) Frozen eval scores + losses only** *(current release)* | `_eval_target_frozen.npz` plus provenance (`manifest.json`, `locks.json`, access log) — **no `answer`/`golds` text** | Cleanest: redistributes only numbers. Regenerates the **frontier-sweep maps** (β̂*(α)) zero-GPU (`REPRODUCE.md` Path A). Full `certify_real` reruns require option (b), option (c), or a future scores-only loader with per-pool score exports. |
| **(a+) Per-pool scores + losses only** *(not shipped here)* | per-pool `<pool>.scores.jsonl` (`{qid, score, n_tokens, L}`, emitted by `export_scores_only.py`) plus frozen eval arrays | Still text-free; would provide the numeric basis for every certificate decision after adding the scores-only pool loader noted in `export_scores_only.py`. |
| **(b) Full JSONL cache** | the `*.jsonl` pools as-is (includes QA text) | Maximal reproducibility, but you redistribute SQuAD (CC BY-SA 4.0 — attribution + share-alike) and **NewsQA** (Microsoft Research license — **redistribution restricted**; you likely must ship *ids + a download/derive script*, not the text). |
| **(c) Regenerate** | ship neither cache; ship only `code/experiments/real/` + this guide | Smallest, no licensing exposure; requires reviewers to have GPUs + model access (Path B). |

NewsQA's license is the binding constraint — confirm its current terms before choosing (b).

## 3. External models (never redistributed here)

| Model | Source | Notes |
|---|---|---|
| Meta-Llama-3-8B-Instruct | Hugging Face `meta-llama/Meta-Llama-3-8B-Instruct` | leg 1; bf16, greedy, `max_new_tokens=64` |
| DeepSeek-V4-Flash | Hugging Face / ModelScope (native FP8/FP4 checkpoint) | leg 2; **no bf16 variant exists** — served in native precision |

The leg-2 serving env is non-default and recorded in the cache `manifest.json`:
vLLM **0.22.1**, `tp=4`, `tokenizer_mode=deepseek_v4`, `kv_cache_dtype=fp8`,
`trust_remote_code`, `thinking=False`, `max_num_seqs=256`, and env
`CUDA_HOME=/usr/local/cuda-<ver>`, `NCCL_P2P_DISABLE=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`,
`HF_HUB_OFFLINE=1`. Determinism is amended (vLLM batching is not bit-exact): the gate is
≥0.90 answer agreement on a 100-item slice + sha256-pinned cache files.

## 4. Provenance notes

- In the authors' working tree, leg 2 overwrote the canonical-path result JSONs and leg-1
  copies were preserved separately; result JSONs are not shipped in this release, and
  `certify_real` regenerates them per leg (`REPRODUCE.md` Path B).
- The model deviations (leg 1 Meta-Llama-3-8B-Instruct; leg 2 DeepSeek-V4-Flash) were availability-triggered
  and adjudicated in the pre-registration record (R002 trigger; R046 + Amendment A2),
  documented in the paper's Appendix G.
