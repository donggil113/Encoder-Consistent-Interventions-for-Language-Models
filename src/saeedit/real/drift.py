"""Decomposition of an edit's SAE-re-encoded effect.

For target ``j`` with intended post-activation change ``alpha`` and activations
``a0`` (before) / ``a1`` (after) with active masks ``m0`` / ``m1``:

* target_change = a1[j] - a0[j]; target_err_rel = |target_change - alpha| / alpha
* drift_orig_active_kept : l2 of changes on i != j with m0[i] and m1[i]
* drift_deactivated      : l2 of changes on i != j with m0[i] and not m1[i]
* drift_orig_active      : l2 over both groups (all originally active non-targets)
* drift_newly_active     : l2 of changes on i != j with not m0[i] and m1[i]
* drift_all_nontarget    : l2 over every i != j

Features inactive before and after have zero change for ReLU/JumpReLU/TopK, so
the three groups partition the non-target change. Drifts are reported absolute
and relative to ``scale`` (alpha for target-matched rows, the norm budget for
equal-norm rows).

``decompose`` is the standard-library reference; ``decompose_torch`` is the
batched version used on real SAEs and is checked against it when torch exists.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

KEYS = ("drift_orig_active_kept", "drift_deactivated", "drift_orig_active", "drift_newly_active",
        "drift_all_nontarget")


def decompose(a0: Sequence[float], a1: Sequence[float], m0: Sequence[bool], m1: Sequence[bool], j: int,
              alpha: float, scale: float) -> Dict[str, float]:
    kept = deact = new = other = 0.0
    n_kept = n_deact = n_new = 0
    for i, (x0, x1, b0, b1) in enumerate(zip(a0, a1, m0, m1)):
        if i == j:
            continue
        d2 = (x1 - x0) ** 2
        if b0 and b1:
            kept += d2
            n_kept += 1
        elif b0:
            deact += d2
            n_deact += 1
        elif b1:
            new += d2
            n_new += 1
        else:
            other += d2  # zero for ReLU/JumpReLU/TopK; kept for other activations
    tc = a1[j] - a0[j]
    out = {
        "target_change": tc,
        "target_err_rel": abs(tc - alpha) / abs(alpha) if alpha else float("nan"),
        "target_active_before": int(bool(m0[j])),
        "target_active_after": int(bool(m1[j])),
        "drift_orig_active_kept": math.sqrt(kept),
        "drift_deactivated": math.sqrt(deact),
        "drift_orig_active": math.sqrt(kept + deact),
        "drift_newly_active": math.sqrt(new),
        "drift_inactive_to_inactive": math.sqrt(other),
        "drift_all_nontarget": math.sqrt(kept + deact + new + other),
        "n_orig_active": n_kept + n_deact,
        "n_deactivated": n_deact,
        "n_newly_active": n_new,
    }
    for k in KEYS:
        out[k + "_rel"] = out[k] / scale if scale else float("nan")
    return out


def decompose_torch(a0, a1, m0, m1, j, alpha, scale):
    """Batched version. Shapes: a*/m* [B, m]; j, alpha, scale [B]. Returns dict of [B] tensors."""
    import torch

    B = a0.shape[0]
    rows = torch.arange(B)
    notj = torch.ones_like(m0, dtype=torch.bool)
    notj[rows, j] = False
    d2 = (a1 - a0) ** 2
    kept_m = m0 & m1 & notj
    deact_m = m0 & ~m1 & notj
    new_m = ~m0 & m1 & notj
    other_m = ~m0 & ~m1 & notj
    kept = (d2 * kept_m).sum(-1)
    deact = (d2 * deact_m).sum(-1)
    new = (d2 * new_m).sum(-1)
    other = (d2 * other_m).sum(-1)
    tc = a1[rows, j] - a0[rows, j]
    out = {
        "target_change": tc,
        "target_err_rel": (tc - alpha).abs() / alpha.abs(),
        "target_active_before": m0[rows, j].long(),
        "target_active_after": m1[rows, j].long(),
        "drift_orig_active_kept": kept.sqrt(),
        "drift_deactivated": deact.sqrt(),
        "drift_orig_active": (kept + deact).sqrt(),
        "drift_newly_active": new.sqrt(),
        "drift_inactive_to_inactive": other.sqrt(),
        "drift_all_nontarget": (kept + deact + new + other).sqrt(),
        "n_orig_active": (m0 & notj).sum(-1),
        "n_deactivated": deact_m.sum(-1),
        "n_newly_active": new_m.sum(-1),
    }
    for k in KEYS:
        out[k + "_rel"] = out[k] / scale
    return out
