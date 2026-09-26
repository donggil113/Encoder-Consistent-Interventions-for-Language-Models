"""Document-cluster bootstrap (standard library only).

Rows are first averaged within each document (the independent unit); the
bootstrap then resamples documents with replacement. Intervals are descriptive
uncertainty for operational decisions, not significance tests.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Sequence

from .dose import quantile


def per_unit_means(rows: Iterable[dict], unit: str, value: Callable[[dict], float]) -> Dict[object, float]:
    acc: Dict[object, List[float]] = defaultdict(list)
    for r in rows:
        v = value(r)
        if v is not None and v == v:  # drop None/NaN explicitly; count them elsewhere
            acc[r[unit]].append(v)
    return {u: statistics.fmean(v) for u, v in acc.items()}


def cluster_bootstrap(unit_values: Dict[object, float], stat: Callable[[Sequence[float]], float] = statistics.median,
                      n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    units = sorted(unit_values, key=str)
    vals = [unit_values[u] for u in units]
    if not vals:
        return {"n_units": 0, "estimate": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    boots = sorted(stat([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(n_boot))
    a = (1.0 - level) / 2.0
    return {"n_units": len(vals), "estimate": stat(vals), "lo": quantile(boots, a), "hi": quantile(boots, 1 - a),
            "n_boot": n_boot, "seed": seed, "level": level}


def paired_unit_differences(rows: Iterable[dict], unit: str, key: Callable[[dict], tuple], method_col: str,
                            a: str, b: str, value: Callable[[dict], float]) -> Dict[object, float]:
    """Per unit: mean over matched (key) pairs of value(a) - value(b). Unmatched rows are skipped."""
    by: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    unit_of: Dict[tuple, object] = {}
    for r in rows:
        if r[method_col] in (a, b):
            v = value(r)
            if v is None or v != v:
                continue
            k = key(r)
            by[k][r[method_col]] = v
            unit_of[k] = r[unit]
    acc: Dict[object, List[float]] = defaultdict(list)
    for k, d in by.items():
        if a in d and b in d:
            acc[unit_of[k]].append(d[a] - d[b])
    return {u: statistics.fmean(v) for u, v in acc.items()}
