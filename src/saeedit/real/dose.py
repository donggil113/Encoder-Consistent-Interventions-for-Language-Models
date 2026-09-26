"""Calibration-only feature pool and doses (standard library only).

Everything here is computed from calibration documents and frozen to JSON
before any test window is encoded. ``freeze`` stores a content hash that the
run stage re-checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from typing import Dict, List, Sequence


def quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of an already sorted sequence (numpy 'linear' rule)."""
    if not sorted_vals:
        raise ValueError("empty")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q outside [0, 1]")
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


def feature_pool(fire_counts: Sequence[int], n_tokens: int, lo: float, hi: float) -> List[int]:
    """Features whose calibration firing density lies in [lo, hi]."""
    return [i for i, c in enumerate(fire_counts) if lo <= c / n_tokens <= hi]


def sample_features(pool: Sequence[int], n: int, seed: int) -> List[int]:
    if len(pool) < n:
        raise ValueError(f"pool has {len(pool)} features < {n}")
    return sorted(random.Random(seed).sample(list(pool), n))


def target_doses(pos_acts: Dict[int, List[float]], quantiles: Sequence[float], extra_mult_of_last: float = 2.0) -> Dict[int, dict]:
    """Per feature: activation quantiles of *positive* calibration activations, plus
    ``extra_mult_of_last`` times the last quantile. a_max is recorded but not used as a dose."""
    out = {}
    for j, vals in pos_acts.items():
        s = sorted(v for v in vals if v > 0.0)
        if not s:
            out[j] = {"status": "NO_POSITIVE_CALIBRATION_ACTIVATIONS"}
            continue
        qs = {f"q{int(round(q * 100))}": quantile(s, q) for q in quantiles}
        last = qs[f"q{int(round(quantiles[-1] * 100))}"]
        qs[f"{extra_mult_of_last:g}xq{int(round(quantiles[-1] * 100))}"] = extra_mult_of_last * last
        out[j] = {"status": "OK", "doses": qs, "a_max_recorded_not_used": s[-1], "n_positive": len(s)}
    return out


def norm_budgets(resid_norms: Sequence[float], multipliers: Sequence[float]) -> dict:
    ref = statistics.median(resid_norms)
    return {"reference_median_norm": ref, "budgets": {f"{m:g}": m * ref for m in multipliers},
            "n_positions": len(resid_norms)}


def freeze(obj: dict, path: str) -> str:
    """Write canonical JSON and return its sha256 (recorded in the manifest)."""
    blob = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
    with open(path, "wb") as f:
        f.write(blob)
    return hashlib.sha256(blob).hexdigest()


def load_frozen(path: str, expected_sha256: str) -> dict:
    with open(path, "rb") as f:
        blob = f.read()
    got = hashlib.sha256(blob).hexdigest()
    if got != expected_sha256:
        raise RuntimeError(f"calibration file changed: {got} != {expected_sha256}")
    return json.loads(blob)
