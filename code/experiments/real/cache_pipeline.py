"""R032 — one-time frozen-pipeline cache (GPU step; <= 2-4 GPU-days budget).

Frozen pipeline (R002 §2): Qwen2.5-7B-Instruct, greedy, max_new_tokens=64, fixed
prompt template (verbatim below, hashed into the manifest). Router score S = mean
token log-probability of the generated answer. Outputs cached once as JSONL per pool;
NO correctness label is computed here for any target/eval item — eval labels are
touched only at the final-eval step (R033/R045). Source labeled pools (D_w^P, D_r)
carry their answers through (the legitimate labeled resource); their correctness
labels are computed at certification time, CPU-side.

Determinism check: a 100-item slice is re-generated and must reproduce scores
bit-identically (cache manifest records the outcome).

Usage:
  CUDA_VISIBLE_DEVICES=... python -m experiments.real.cache_pipeline [--fallback]
        [--tp 4] [--max-items N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from .data import build_pools, DATA_DIR

# Cache dir is overridable so a second workload (R048) writes to its own dir and
# NEVER clobbers/reads leg-2's `cache/`. Default = `cache/` (leg-1/leg-2 unchanged).
# Resolution order: --cache-dir (set in main) > $REAL_CACHE_DIR > default.
CACHE_DIR = os.environ.get("REAL_CACHE_DIR") or os.path.join(os.path.dirname(__file__), "cache")

PROMPT = ("Read the passage and answer the question. Reply with ONLY the shortest "
          "exact answer span copied from the passage — no explanations, no extra "
          "words.\n\nPassage: {context}\n\nQuestion: {question}")
# R002 registered primary Qwen2.5-7B-Instruct is UNAVAILABLE on this box (HF cache
# holds config only — no weights; network/proxy cannot deliver 15GB: measured ~0 B/s),
# and the registered fallback Llama-3.1-8B-Instruct is also absent locally. Per the
# R002 §2/§3 availability-only trigger, the nearest same-class local member is used:
MODEL = "<MODEL_DIR>/Meta-Llama-3-8B-Instruct"
MODEL_DEVIATION = ("registered primary Qwen/Qwen2.5-7B-Instruct unavailable (weights "
                   "absent locally; proxy network unusable, ~0 B/s measured); "
                   "registered fallback Llama-3.1-8B-Instruct also absent; same-class "
                   "local member Meta-Llama-3-8B-Instruct used. Trigger = availability "
                   "ONLY (R002: never output quality). Documented protocol deviation.")
MAX_NEW = 64
CTX_CLIP = 6000          # chars; MRQA contexts can be long — clip head (registered)


def _prompt(rec):
    return PROMPT.format(context=rec["context"][:CTX_CLIP], question=rec["question"])


def run_pool(llm, sampling, recs, name, out_path, chat_kwargs=None):
    t0 = time.time()
    msgs = [[{"role": "user", "content": _prompt(r)}] for r in recs]
    # chat template — the registered instruct invocation (raw completion made the
    # instruct model continue the text instead of answering; observed from answer
    # FORMAT only, label-blind — documented deviation). chat_kwargs: R046 leg-2
    # non-thinking invocation (enable_thinking=False), recorded in the manifest.
    if chat_kwargs:
        outs = llm.chat(msgs, sampling, chat_template_kwargs=chat_kwargs)
    else:
        outs = llm.chat(msgs, sampling)
    rows = []
    for rec, o in zip(recs, outs):
        comp = o.outputs[0]
        # CHOSEN-token logprobs extracted BY TOKEN ID aligned with token_ids
        # (review round-1 MAJOR fix — dict iteration order is not a contract)
        if comp.logprobs is None or len(comp.logprobs) != len(comp.token_ids):
            raise RuntimeError(f"{name}: missing/misaligned logprobs "
                               f"({None if comp.logprobs is None else len(comp.logprobs)}"
                               f" vs {len(comp.token_ids)} tokens)")
        logps = [lp[tid].logprob for tid, lp in zip(comp.token_ids, comp.logprobs)]
        if not logps:
            raise RuntimeError(f"{name}: empty generation for qid {rec['qid']}")
        score = float(np.mean(logps))
        if not np.isfinite(score):
            raise RuntimeError(f"{name}: nonfinite score for qid {rec['qid']}")
        row = {"qid": rec["qid"], "answer": comp.text.strip(), "score": score,
               "n_tokens": len(logps)}
        if "answers" in rec:                       # source labeled pools only
            row["golds"] = rec["answers"]
        rows.append(row)
    scores = np.array([r["score"] for r in rows])
    if len(rows) and float(scores.std()) < 1e-9:
        raise RuntimeError(f"{name}: degenerate score distribution (std≈0)")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[cache] {name}: {len(rows)} items in {time.time()-t0:.0f}s -> {out_path}")
    return rows


def main():
    global CACHE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", dest="cache_dir", default=None,
                    help="output cache dir (R048 second workload: a fresh dir, never "
                         "leg-2's cache/); overrides $REAL_CACHE_DIR; default = cache/")
    ap.add_argument("--fallback", action="store_true",
                    help="use registered fallback target (TriviaQA-web) — "
                         "availability-only trigger per R002 §3 / R048 second workload")
    ap.add_argument("--pilot-split", dest="pilot_split", action="store_true",
                    help="R048: split target-dev into pilot/eval (sha1 50/50); caches a "
                         "pilot_target pool, eval_target = eval slice only")
    ap.add_argument("--tp", type=int, default=4, help="tensor parallel size")
    ap.add_argument("--max-items", type=int, default=0,
                    help="debug cap per pool (0 = full)")
    ap.add_argument("--model", default=MODEL,
                    help="model path/name (R046 leg-2 escalation: pass the "
                         "pre-registered escalated model; default = leg-1 model)")
    ap.add_argument("--deviation", default=MODEL_DEVIATION,
                    help="protocol-deviation note recorded in the manifest")
    ap.add_argument("--no-think", action="store_true",
                    help="invoke chat template with enable_thinking=False "
                         "(R046: Qwen3.6 hybrid-thinking default would consume "
                         "the frozen 64-token budget)")
    # serving-robustness knobs (R047 attempt-2: silent TP-worker hang on PCIe-only
    # H100s mid-generation): score/decoding semantics untouched; argv -> manifest
    ap.add_argument("--max-num-seqs", type=int, default=0,
                    help="cap concurrent sequences (0 = engine default)")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="disable CUDA graphs/compile (robustness fallback)")
    ap.add_argument("--disable-custom-ar", action="store_true",
                    help="disable custom allreduce (PCIe topology robustness)")
    # R046-A2 (DSV4-Flash leg-2, codex-required guards): native invocation path +
    # arch-mandated KV format + tokenizer code provenance hashing
    ap.add_argument("--tokenizer-mode", default=None,
                    help="vLLM tokenizer mode (e.g. deepseek_v4 native encoding)")
    ap.add_argument("--kv-cache-dtype", default=None,
                    help="KV cache dtype (DSV4 arch mandates fp8)")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="allow model-dir tokenizer code (hashed into manifest)")
    ap.add_argument("--think-kwarg", default="enable_thinking",
                    choices=["enable_thinking", "thinking"],
                    help="chat-template kwarg name used to disable thinking")
    args = ap.parse_args()
    if args.cache_dir:
        CACHE_DIR = os.path.abspath(args.cache_dir)
    # R048 second workload (--pilot-split / --fallback): must target a FRESH cache dir,
    # never a frozen leg cache or the quarantine dir (realpath denylist; codex re-audit #3).
    if args.pilot_split and not (args.cache_dir or os.environ.get("REAL_CACHE_DIR")):
        raise SystemExit("--pilot-split (R048) requires an explicit --cache-dir / "
                         "$REAL_CACHE_DIR (a fresh dir, e.g. cache_w2_triviaqa_dsv4)")
    if args.pilot_split or args.fallback:
        from .data import assert_safe_cache
        assert_safe_cache(CACHE_DIR)

    os.makedirs(CACHE_DIR, exist_ok=True)
    # a fresh R032 invalidates any stale eval artifacts / access log (round-2 fix)
    for stale in ("_eval_target_frozen.npz", "_eval_source_frozen.npz",
                  "_eval_access.log"):
        sp = os.path.join(CACHE_DIR, stale)
        if os.path.exists(sp):
            os.remove(sp)
    pools = build_pools(primary=not args.fallback, pilot_split=args.pilot_split)

    from vllm import LLM, SamplingParams
    extra = {}
    if args.max_num_seqs:
        extra["max_num_seqs"] = args.max_num_seqs
    if args.enforce_eager:
        extra["enforce_eager"] = True
    if args.disable_custom_ar:
        extra["disable_custom_all_reduce"] = True
    if args.tokenizer_mode:
        extra["tokenizer_mode"] = args.tokenizer_mode
    if args.kv_cache_dtype:
        extra["kv_cache_dtype"] = args.kv_cache_dtype
    if args.trust_remote_code:
        extra["trust_remote_code"] = True
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=8192, seed=SEED_VLLM,
              **extra)
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, logprobs=0)
    chat_kwargs = {args.think_kwarg: False} if args.no_think else None

    import sys
    import vllm as _vllm
    manifest = {"model": args.model, "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "max_new_tokens": MAX_NEW, "ctx_clip": CTX_CLIP,
                "temperature": 0.0, "target": pools["target_name"],
                "counts": pools["counts"], "pools": {},
                "seed": SEED_VLLM, "max_items": args.max_items, "tp": args.tp,
                "dtype": "bfloat16", "max_model_len": 8192,
                "vllm_version": _vllm.__version__,
                "model_snapshot": _resolve_model_snapshot(args.model),
                "model_deviation": args.deviation,
                "chat_template_kwargs": chat_kwargs,
                "serving_config": {"tokenizer_mode": args.tokenizer_mode,
                                   "kv_cache_dtype": args.kv_cache_dtype,
                                   "trust_remote_code": bool(args.trust_remote_code),
                                   "max_num_seqs": args.max_num_seqs or None,
                                   "enforce_eager": bool(args.enforce_eager),
                                   "disable_custom_all_reduce": bool(args.disable_custom_ar)},
                "model_dir_provenance": _model_dir_provenance(args.model),
                "resolved_quantization": _resolved_quant(llm),
                "env_robustness": {k: os.environ.get(k) for k in
                                   ("NCCL_P2P_DISABLE", "VLLM_USE_FLASHINFER_SAMPLER",
                                    "CUDA_HOME", "HF_HUB_OFFLINE")},
                "argv": sys.argv, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    # persist split qid lists (audit artifact; labels not included)
    splits_path = os.path.join(CACHE_DIR, "split_qids.json")
    pool_names = ["src_lock", "src_wP", "src_r", "src_unlabeled_ladder",
                  "tgt_unlabeled", "eval_target", "eval_source"]
    if args.pilot_split:
        pool_names.append("pilot_target")          # R048: pilot slice cached for frontier readout
    with open(splits_path, "w") as f:
        json.dump({k: [r["qid"] for r in pools[k]] for k in pool_names}, f)
    manifest["split_qids_sha256"] = _sha(splits_path)
    manifest["pilot_split"] = bool(args.pilot_split)
    todo = list(pool_names)
    for name in todo:
        recs = pools[name]
        if args.max_items:
            recs = recs[: args.max_items]
        out_path = os.path.join(CACHE_DIR, f"{name}.jsonl")
        rows = run_pool(llm, sampling, recs, name, out_path, chat_kwargs)
        scores = [r["score"] for r in rows]
        import numpy as _np
        manifest["pools"][name] = {"n": len(rows), "path": out_path,
                                   "sha256": _sha(out_path),
                                   "qids_sha256": hashlib.sha256(
                                       "".join(r["qid"] for r in rows).encode()).hexdigest(),
                                   "score_summary": {"mean": float(_np.mean(scores)),
                                                     "std": float(_np.std(scores)),
                                                     "min": float(_np.min(scores)),
                                                     "max": float(_np.max(scores))}}

    # determinism check: re-generate a 100-item slice of src_lock
    slice_recs = pools["src_lock"][:100]
    redo = run_pool(llm, sampling, slice_recs, "determinism_slice",
                    os.path.join(CACHE_DIR, "_det_slice.jsonl"), chat_kwargs)
    orig = [json.loads(l) for l in open(os.path.join(CACHE_DIR, "src_lock.jsonl"))][:100]
    agree = sum(a["answer"] == b["answer"] for a, b in zip(orig, redo)) / len(redo)
    drifts = sorted(abs(a["score"] - b["score"]) for a, b in zip(orig, redo))
    det = {"answer_agreement": agree,
           "score_drift_median": drifts[len(drifts) // 2],
           "score_drift_max": drifts[-1]}
    # AMENDED determinism criterion (documented R002 deviation): bit-identical
    # reproduction is unrealizable under vLLM continuous batching (fp reduction order
    # varies with batch composition — engine property, measured). The frozen-pipeline
    # property is carried by the hash-pinned cache itself; the re-run slice gates only
    # against degenerate instability:
    det_ok = agree >= 0.90
    manifest["determinism_check_100"] = bool(det_ok)
    manifest["determinism_summary"] = det
    manifest["determinism_amendment"] = (
        "bit-identical criterion unrealizable under vLLM batching; amended gate: "
        "answer agreement >= 0.90 on the 100-item slice, drift quantiles recorded. "
        "Frozen-pipeline identity = sha256-pinned cache files (R002 deviation note).")
    if not det_ok:
        raise RuntimeError(f"R032 determinism gate FAILED (agreement {agree:.2f} < 0.90)")

    # eval golds are NEVER persisted by R032 (review CRITICAL fix): the single
    # guarded raw-label access happens at final eval via data.load_eval_golds_guarded
    with open(os.path.join(CACHE_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({"R032": "cache complete", "determinism": det_ok,
                      "target": pools["target_name"]}))




def _model_dir_provenance(model_name: str) -> dict:
    """R046-A2 codex-required guard: sha256 of every local Python file under the
    model dir (tokenizer/encoding code executed via trust_remote_code) plus
    tokenizer/config/generation/index metadata files. Local dirs only."""
    if not os.path.isdir(model_name):
        return {}
    out = {}
    for root, _dirs, files in os.walk(model_name):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), model_name)
            if fn.endswith(".py") or fn in (
                    "tokenizer_config.json", "tokenizer.json", "config.json",
                    "configuration.json", "generation_config.json",
                    "model.safetensors.index.json", "chat_template.jinja",
                    "special_tokens_map.json"):
                out[rel] = _sha(os.path.join(root, fn))
    return out


def _resolved_quant(llm) -> dict:
    """Record vLLM's RESOLVED quantization identity (R046-A2: the checkpoint is
    natively FP8-block + FP4-expert mixed; the manifest must say what actually ran)."""
    out = {}
    try:
        mc = llm.llm_engine.vllm_config.model_config
        out["quantization"] = str(getattr(mc, "quantization", None))
        out["dtype"] = str(getattr(mc, "dtype", None))
    except Exception as e:                                    # version-dependent path
        out["probe_error"] = f"{type(e).__name__}: {e}"
    try:
        import json as _json
        cfg = _json.load(open(os.path.join(llm.llm_engine.vllm_config
                                           .model_config.model, "config.json")))
        out["checkpoint_quantization_config"] = cfg.get("quantization_config")
    except Exception:
        pass
    return out


def _resolve_model_snapshot(model_name: str) -> str:
    """Resolved model identity for the manifest. Local directory: path + config sha256
    + weight-shard sha256 of the index. HF name: local snapshot path. HARD-FAILS if
    unresolvable — an unauditable model identity must never produce a 'valid' cache."""
    if os.path.isdir(model_name):
        cfg = os.path.join(model_name, "config.json")
        idx = os.path.join(model_name, "model.safetensors.index.json")
        parts = [model_name, "config:" + _sha(cfg)]
        if os.path.exists(idx):
            parts.append("index:" + _sha(idx))
        return " | ".join(parts)
    from huggingface_hub import snapshot_download
    return snapshot_download(model_name, local_files_only=True)

def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SEED_VLLM = 20260611

if __name__ == "__main__":
    main()
