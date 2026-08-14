"""R002-locked corpus handling: MRQA-format SQuAD (source) -> NewsQA (target),
fallback TriviaQA-web. Label-blind discipline per R039 §3 / R002 §1:

- Source labeled pool: 40k subsample -> D_lock 20% / D_w^P 40% / D_r 40% (seed 20260611).
- Target unlabeled pool: 40k NewsQA-train subsample; answers STRUCTURALLY DROPPED by
  the loader (UnlabeledRecord has no answer field).
- Source-domain unlabeled pool: 20k disjoint SQuAD-train subsample (ladder mixtures).
- Eval sets: NewsQA dev (all) + SQuAD dev (ladder mixtures); labels behind EvalGuard —
  exactly one read, at final evaluation; every access logged.
- Correctness rule (registered): token-F1 vs any gold >= 0.5 (official SQuAD
  normalization); L = 1{F1 < 0.5}.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import re
import string
import urllib.request

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED = 20260611

MRQA = "https://s3.us-east-2.amazonaws.com/mrqa/release/v2"
URLS = {
    "squad_train": f"{MRQA}/train/SQuAD.jsonl.gz",
    "squad_dev": f"{MRQA}/dev/SQuAD.jsonl.gz",
    "newsqa_train": f"{MRQA}/train/NewsQA.jsonl.gz",
    "newsqa_dev": f"{MRQA}/dev/NewsQA.jsonl.gz",
    # registered fallback (availability-only trigger, R002 §3)
    "triviaqa_train": f"{MRQA}/train/TriviaQA-web.jsonl.gz",
    "triviaqa_dev": f"{MRQA}/dev/TriviaQA-web.jsonl.gz",
}


def fetch(name: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.jsonl.gz")
    if not os.path.exists(path):
        print(f"[data] downloading {name} ...")
        urllib.request.urlretrieve(URLS[name], path)
    return path


def load_mrqa(path: str, limit: int | None = None, include_answers: bool = True,
              keep_qid=None):
    """Rows from an MRQA jsonl.gz. include_answers=False NEVER reads qa["answers"]
    (true label-free path — labels are not even materialized in memory; review
    round-2 CRITICAL fix): rows are (qid, context, question) triples.

    keep_qid: optional predicate qid->bool. When given, answers are extracted/stored
    ONLY for qids that pass — used by the slice-aware eval guard so that a single
    target-dev slice (e.g. pilot) is materialized WITHOUT ever storing the other
    slice's labels (R048 §12; codex round-3 CRITICAL fix)."""
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        next(f)                                              # MRQA header line (unused)
        for line in f:
            obj = json.loads(line)
            ctx = obj["context"]
            for qa in obj["qas"]:
                qid = qa["qid"]
                if keep_qid is not None and not keep_qid(qid):
                    continue                                 # skip before touching answers
                if include_answers:
                    out.append((qid, ctx, qa["question"], qa.get("answers", [])))
                else:
                    out.append((qid, ctx, qa["question"]))
                if limit and len(out) >= limit:
                    return out
    return out


# ------------------------------------------------------- official SQuAD F1 pieces


def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, golds: list[str]) -> float:
    best = 0.0
    p_toks = _norm(pred).split()
    for g in golds:
        g_toks = _norm(g).split()
        common = collections.Counter(p_toks) & collections.Counter(g_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(p_toks) if p_toks else 0.0
        rec = num_same / len(g_toks) if g_toks else 0.0
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def correct(pred: str, golds: list[str]) -> int:
    """Registered correctness: token-F1 >= 0.5 (L = 1 - correct)."""
    return int(token_f1(pred, golds) >= 0.5)


# ------------------------------------------------------------- pools and splits


def sha1_slice(qid: str) -> str:
    """R048 §6 registered target-dev split (reproducible, language-independent — NOT
    Python hash()): pilot iff int(sha1(qid),16) % 2 == 0 (≈50/50), else eval."""
    return "pilot" if int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 2 == 0 else "eval"


_PROTECTED_CACHES = ("cache", "cache_leg1_llama3_8b", "quarantine_qwen36_partial")


def assert_safe_cache(cache_dir: str) -> None:
    """Refuse to operate the R048 second workload on a frozen leg cache (leg-2 `cache/`,
    leg-1 `cache_leg1_llama3_8b/`) or the quarantined Qwen3.6 dir. realpath-based, so
    trailing slashes / symlinks cannot bypass it; also blocks sub-paths thereof."""
    rp = os.path.realpath(cache_dir)
    base = os.path.dirname(os.path.abspath(__file__))          # experiments/real
    for name in _PROTECTED_CACHES:
        prot = os.path.realpath(os.path.join(base, name))
        if rp == prot or rp.startswith(prot + os.sep):
            raise SystemExit(f"refusing R048 operation on protected cache {name!r} "
                             f"({rp}); use a fresh dir, e.g. cache_w2_triviaqa_dsv4")


def load_eval_golds_guarded(side: str, caller: str, cache_dir: str,
                            slice_name: str | None = None) -> dict:
    """HARD guard for raw eval labels (R039 §6.5 + R048 §6). Registered callers and the
    target-dev slice each MAY touch:
      R033_final_eval / R035_stability_eval -> None (whole dev; leg-1/leg-2 only)
      R048_final_eval -> 'eval'  (touched ONCE, the final verdict)
      R048_pilot      -> 'pilot' (SLA-tier bracket confirmation; disjoint slice)
    A caller may touch ONLY its registered slice, ONCE per (caller, side, slice). On an
    R048 pilot-split cache, TARGET labels are reachable ONLY via R048_pilot/R048_final_eval
    (legacy whole-dev callers are blocked from peeking either slice). Whole-set access and
    sliced access are mutually exclusive on a side (legacy log entries normalized to
    slice=None). When a slice is given, the returned golds are FILTERED to that slice at
    the barrier, so a caller never even sees the other slice's labels."""
    registered = {"R033_final_eval": None, "R035_stability_eval": None,
                  "R048_final_eval": "eval", "R048_pilot": "pilot"}
    if caller not in registered:
        raise PermissionError(f"eval-label access by unregistered caller {caller!r}")
    allowed = registered[caller]
    if allowed is not None and slice_name != allowed:
        raise PermissionError(f"{caller!r} may access only slice {allowed!r}, "
                              f"not {slice_name!r}")
    manifest = json.load(open(os.path.join(cache_dir, "manifest.json")))
    # R048 caches: target labels ONLY via the two R048 callers (block leg-1/2 callers).
    if manifest.get("pilot_split") and side == "target" \
            and caller not in ("R048_pilot", "R048_final_eval"):
        raise PermissionError(f"pilot-split (R048) cache: target labels only via "
                              f"R048_pilot/R048_final_eval, not {caller!r}")
    log_path = os.path.join(cache_dir, "_eval_access.log")
    prior = [json.loads(l) for l in open(log_path)] if os.path.exists(log_path) else []
    slc = lambda p: p.get("slice", None)             # legacy entries -> whole-dev (None)
    same_side = [p for p in prior if p.get("side") == side]
    if any(p.get("caller") == caller and slc(p) == slice_name for p in same_side):
        raise PermissionError(f"repeat eval-label access {caller}/{side}/{slice_name} — "
                              "reuse the frozen eval artifact instead")
    if allowed is None:                              # whole-set caller: must be the ONLY access
        if same_side:
            raise PermissionError(f"whole-dev eval blocked: side {side} already accessed "
                                  f"by {[ (p.get('caller'), slc(p)) for p in same_side]}")
    else:                                            # sliced caller: NO whole-dev access may precede
        if any(slc(p) is None for p in same_side):
            raise PermissionError(f"sliced eval blocked: a whole-dev access on side {side} "
                                  f"already exists {[p.get('caller') for p in same_side]}")
    with open(log_path, "a") as f:
        f.write(json.dumps({"caller": caller, "side": side, "slice": slice_name}) + "\n")
    fname = f"{manifest['target']}_dev" if side == "target" else "squad_dev"
    # Materialize answers ONLY for the requested slice's qids (filter at load time, before
    # any answer is stored): a pilot read never touches eval-slice labels (R048 §12).
    keep = None if slice_name is None else (lambda q: sha1_slice(q) == slice_name)
    rows = load_mrqa(fetch(fname), keep_qid=keep)
    return {r[0]: r[3] for r in rows}


def build_pools(primary: bool = True, pilot_split: bool = False):
    """Returns dict of pools per R002 §1. UNLABELED pools carry NO answer field;
    eval golds are NEVER materialized here (review round-1 CRITICAL fix) — they are
    read once, at final eval, through load_eval_golds_guarded."""
    tgt = "newsqa" if primary else "triviaqa"
    rng = np.random.default_rng(SEED)
    sq_train = load_mrqa(fetch("squad_train"))                     # labeled pool source ONLY
    sq_train_u = load_mrqa(fetch("squad_train"), include_answers=False)  # ladder pool: label-free load
    tg_train = load_mrqa(fetch(f"{tgt}_train"), include_answers=False)
    sq_dev = load_mrqa(fetch("squad_dev"), include_answers=False)
    tg_dev = load_mrqa(fetch(f"{tgt}_dev"), include_answers=False)

    idx = rng.permutation(len(sq_train))
    src_lab_idx = idx[:40000]
    src_unl_idx = idx[40000:60000]                  # disjoint ladder pool
    tgt_idx = rng.permutation(len(tg_train))[:40000]

    def rec_l(rows, ids):                            # labeled record
        return [{"qid": rows[i][0], "context": rows[i][1], "question": rows[i][2],
                 "answers": rows[i][3]} for i in ids]

    def rec_u(rows, ids):                            # unlabeled: answers DROPPED
        return [{"qid": rows[i][0], "context": rows[i][1], "question": rows[i][2]}
                for i in ids]

    src_pool = rec_l(sq_train, src_lab_idx)
    perm = rng.permutation(40000)
    # R048 §6: optional pilot/eval split of the TARGET dev set by the registered sha1
    # qid hash (≈50/50). pilot labels are allowed (SLA-tier bracket confirmation); eval
    # is touched once at final eval. Default off -> leg-1/leg-2 (whole dev = eval).
    if pilot_split:
        eval_idx = [i for i in range(len(tg_dev)) if sha1_slice(tg_dev[i][0]) == "eval"]
        pilot_idx = [i for i in range(len(tg_dev)) if sha1_slice(tg_dev[i][0]) == "pilot"]
    else:
        eval_idx, pilot_idx = list(range(len(tg_dev))), []
    pools = {
        "target_name": tgt,
        "src_lock": [src_pool[i] for i in perm[:8000]],          # 20%
        "src_wP": [src_pool[i] for i in perm[8000:24000]],       # 40%
        "src_r": [src_pool[i] for i in perm[24000:40000]],       # 40%
        "src_unlabeled_ladder": rec_u(sq_train_u, src_unl_idx),
        "tgt_unlabeled": rec_u(tg_train, tgt_idx),
        "eval_target": rec_u(tg_dev, eval_idx),                  # questions only (eval slice)
        "eval_source": rec_u(sq_dev, range(len(sq_dev))),
        "counts": {"squad_train": len(sq_train), f"{tgt}_train": len(tg_train),
                   "squad_dev": len(sq_dev), f"{tgt}_dev": len(tg_dev),
                   "eval_target_slice": len(eval_idx),
                   "pilot_target_slice": len(pilot_idx)},
    }
    if pilot_split:
        pools["pilot_target"] = rec_u(tg_dev, pilot_idx)
        assert len(eval_idx) >= 1000, "R048 §6: eval slice alone must have N_eval >= 1000"
        assert len(pilot_idx) >= 1000, "R048 §6: pilot slice must have >= 1000"
    else:
        assert pools["counts"][f"{tgt}_dev"] >= 2000, "N_eval >= 2000 metadata gate"
    return pools
