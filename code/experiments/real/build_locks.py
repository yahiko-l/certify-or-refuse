"""R039 §1/§2 lock instantiation: λ-lattice (64 source-score quantile thresholds) and
K=16 equal-mass cell edges, computed ONLY from the D_lock cache (independent
construction split). Emits locks.json with hashes — the lattice artifact.

Also emits the SECOND parameterization (B4.3 real leg, R035): 64 uniform thresholds
on [min, max] of the D_lock scores — sensitivity only, never the primary.

Usage: python -m experiments.real.build_locks
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

# Overridable cache dir (R048 second workload); default = cache/ (leg-1/leg-2 unchanged).
CACHE_DIR = os.environ.get("REAL_CACHE_DIR") or os.path.join(os.path.dirname(__file__), "cache")


def main():
    global CACHE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", dest="cache_dir", default=None,
                    help="cache dir holding src_lock.jsonl; overrides $REAL_CACHE_DIR")
    args = ap.parse_args()
    if args.cache_dir:
        CACHE_DIR = os.path.abspath(args.cache_dir)
    # Never (re)build locks into a frozen leg cache or the quarantine dir (codex re-audit #3).
    from .data import assert_safe_cache
    assert_safe_cache(CACHE_DIR)
    rows = [json.loads(l) for l in open(os.path.join(CACHE_DIR, "src_lock.jsonl"))]
    scores = np.array([r["score"] for r in rows], float)
    lattice = np.quantile(scores, np.arange(1, 65) / 65.0)          # |Λ| = 64
    edges = np.quantile(scores, np.arange(1, 16) / 16.0)            # K = 16 cells
    uniform = np.linspace(scores.min(), scores.max(), 66)[1:-1]     # 64 uniform
    locks = {"n_lock": len(scores),
             "lattice_thresholds": lattice.tolist(),
             "cell_edges": edges.tolist(),
             "uniform_lattice": uniform.tolist(),
             "B": 10.0, "K": 16, "n_lattice": 64,
             "alpha": 0.10, "beta": 0.60, "delta": 0.05,
             "source": "D_lock cache only (R039 v2 independent construction split)"}
    blob = json.dumps(locks, sort_keys=True).encode()
    locks["sha256"] = hashlib.sha256(blob).hexdigest()
    out = os.path.join(CACHE_DIR, "locks.json")
    with open(out, "w") as f:
        json.dump(locks, f, indent=1)
    print(json.dumps({"locks": out, "sha256": locks["sha256"],
                      "n_lock_scores": len(scores)}))


if __name__ == "__main__":
    main()
