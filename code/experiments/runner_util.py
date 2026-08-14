"""Shared launcher utilities: JSON output discipline, parallel cell maps, grids."""
from __future__ import annotations

import json
import os
import time

import numpy as np
from joblib import Parallel, delayed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def save_json(name: str, payload: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    payload = dict(payload)
    payload.setdefault("_written_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=_np_default)
    print(f"[saved] {path}")
    return path


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def pmap(fn, items, n_jobs: int = 16, desc: str = ""):
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(fn)(it) for it in items)
    if desc:
        print(f"[{desc}] {len(items)} units in {time.time() - t0:.1f}s")
    return out


def log_grid(lo: float, hi: float, k: int) -> np.ndarray:
    return np.unique(np.round(np.exp(np.linspace(np.log(lo), np.log(hi), k))).astype(int))
