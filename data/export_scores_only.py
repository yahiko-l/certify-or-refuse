#!/usr/bin/env python3
"""Export a TEXT-FREE numbers-only view of a frozen real-data cache (option (a)).

For each pool `<name>.jsonl` it writes `<name>.scores.jsonl` containing only
`{qid, score, n_tokens, L?}` — dropping the corpus-licensed text fields `answer`
(model generation) and `golds` (gold answers). The 0/1 loss `L = 1 - correct(answer, golds)`
is *precomputed* here for labeled pools (those carrying `golds`) using the registered
correctness rule (token-F1 >= 0.5, official SQuAD normalization), so no text needs to ship.

Unlabeled pools (`tgt_unlabeled`, `src_unlabeled_ladder`) have no golds -> no `L`.
Eval pools' losses are already in the text-free `_eval_target_frozen.npz` (`s`, `L`).

Usage (normally invoked by stage_release.sh):
    python export_scores_only.py --cache /path/to/cache --out /path/to/dst

This produces the numeric basis for every certificate decision. Consuming it directly
in `certify_real.py` is a one-function change (a scores-only pool loader that reads
`L` instead of recomputing it from `answer`/`golds`); until then, use it for inspection
and ship `requirements-real.txt` Path B for full end-to-end regeneration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _find_repo_root(start: str) -> str:
    """Walk up from the cache dir to the repo root containing experiments/real/data.py."""
    d = os.path.abspath(start)
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "experiments", "real", "data.py")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    raise SystemExit("could not locate repo root (experiments/real/data.py) above " + start)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="source cache dir (with *.jsonl pools)")
    ap.add_argument("--out", required=True, help="destination dir for *.scores.jsonl")
    args = ap.parse_args()

    sys.path.insert(0, _find_repo_root(args.cache))
    from experiments.real.data import correct  # registered token-F1>=0.5 rule

    os.makedirs(args.out, exist_ok=True)
    pools = sorted(f for f in os.listdir(args.cache)
                   if f.endswith(".jsonl") and not f.startswith("_"))
    for fn in pools:
        src = os.path.join(args.cache, fn)
        dst = os.path.join(args.out, fn.replace(".jsonl", ".scores.jsonl"))
        n = n_labeled = 0
        with open(src) as fin, open(dst, "w") as fout:
            for line in fin:
                r = json.loads(line)
                out = {"qid": r["qid"], "score": r["score"], "n_tokens": r.get("n_tokens")}
                if "golds" in r and r["golds"] is not None:
                    out["L"] = 1 - correct(r["answer"], r["golds"])
                    n_labeled += 1
                fout.write(json.dumps(out) + "\n")
                n += 1
        print(f"  {fn:28s} -> {os.path.basename(dst):36s} ({n} rows, {n_labeled} labeled)")
    print(f">> text-free export complete: {len(pools)} pools (no `answer`/`golds` shipped)")


if __name__ == "__main__":
    main()
