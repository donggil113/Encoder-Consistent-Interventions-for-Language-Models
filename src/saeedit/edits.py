"""Edit constructors and the achieved-vs-intended evaluation.

An *edit problem* is: base activation ``x``, target feature ``j``, intended
post-activation change ``+alpha`` on ``j``, and a protected feature set ``P``
whose activations should not change. The decoder edit ``x + alpha * D_j``
*claims* to realise it; re-encoding measures what it *achieves*.

All correction methods here work on the pre-activation map ``x -> E x + b``,
restricted to the rows ``S u P``. That map is affine, so for ReLU the
correction is exact as long as no unprotected feature crosses its threshold.
For JumpReLU/TopK the local derivative ignores jumps; nothing here is a global
inverse of those encoders.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import linalg as la
from .linalg import Vector
from .sae_toy import ToySAE


@dataclass
class EditProblem:
    sae: ToySAE
    x: Vector
    j: int
    alpha: float
    protected: List[int]
    rtol: float = 1e-10

    # derived
    p: Vector = field(init=False)
    a: Vector = field(init=False)
    mask: List[bool] = field(init=False)
    v_target: float = field(init=False)
    dp_req: float = field(init=False)
    tau: float = field(init=False)

    def __post_init__(self) -> None:
        if self.j in self.protected:
            raise ValueError("target cannot be protected")
        self.p = self.sae.pre(self.x)
        self.a = self.sae.activate(self.p)
        self.mask = self.sae.active_mask(self.p)
        # Intended post-activation value of the target; if active, post = pre.
        self.v_target = self.a[self.j] + self.alpha
        self.dp_req = self.v_target - self.p[self.j]
        self.tau = abs(self.alpha) * la.norm(self.sae.decoder_col(self.j))

    @property
    def rows(self) -> List[int]:
        return [self.j] + list(self.protected)

    def system(self, extra_protected: Sequence[int] = ()) -> tuple:
        """Pre-activation constraint system ``M delta = t`` on S u P (u extra)."""
        idx = self.rows + [i for i in extra_protected if i not in self.rows]
        M = la.take_rows(self.sae.E, idx)
        t = [self.dp_req] + [0.0] * (len(idx) - 1)
        return M, t, idx

    def target_attainable(self) -> bool:
        """Range feasibility of the intended target value (ignores other features)."""
        act, v = self.sae.act, self.v_target
        if act == "identity":
            return True
        if act == "relu":
            return v > 0.0
        if act == "jumprelu":
            return v > self.sae.theta
        return v > 0.0  # topk: also needs to rank in the top k; checked after the edit


# ---------------------------------------------------------------- methods


def _result(delta: Optional[Vector], status: str = "OK", **info) -> dict:
    return {"delta": delta, "status": status, **info}


def edit_decoder(pb: EditProblem) -> dict:
    return _result(la.scale(pb.alpha, pb.sae.decoder_col(pb.j)))


def edit_decoder_rescaled(pb: EditProblem) -> dict:
    """Decoder direction with the coefficient chosen so the target's
    pre-activation moves by exactly ``dp_req`` (strongest 1-D rescaling)."""
    dj = pb.sae.decoder_col(pb.j)
    g = la.dot(pb.sae.E[pb.j], dj)
    if abs(g) < 1e-12:
        return _result(None, "FAIL", reason="encoder row orthogonal to decoder column")
    coef = pb.dp_req / g
    return _result(la.scale(coef, dj), coef=coef, sign_flip=(coef * pb.alpha < 0.0))


def edit_encoder_grad(pb: EditProblem) -> dict:
    """Least-norm edit for the target row alone: dp_req * e_j / ||e_j||^2."""
    ej = pb.sae.E[pb.j]
    n2 = la.dot(ej, ej)
    if n2 == 0.0:
        return _result(None, "FAIL", reason="zero encoder row")
    return _result(la.scale(pb.dp_req / n2, ej))


def edit_jacobian_ln(pb: EditProblem, extra_protected: Sequence[int] = ()) -> dict:
    """Minimum-norm (least-squares if infeasible) solution of M delta = t (dense SVD)."""
    M, t, idx = pb.system(extra_protected)
    sol = la.lstsq_min_norm(M, t, pb.rtol)
    r = sol["residual"]
    tn = la.norm(t)
    return _result(sol["x"], residual_rel=(la.norm(r) / tn if tn else 0.0), rank=sol["rank"],
                   n_constraints=len(idx), residual=r, M=M, t=t)


def edit_jacobian_ln_cgls(pb: EditProblem, max_iter_factor: int = 4, rel_tol: float = 1e-12) -> dict:
    """Same target as ``edit_jacobian_ln`` but matrix-free: only JVPs (M v) and
    VJPs (M^T u) are used. In a real model these would be autodiff calls; here
    the operator is the gathered encoder rows."""
    M, t, idx = pb.system()
    n = pb.sae.d
    sol = la.cgls(lambda v: la.matvec(M, v), lambda u: la.rmatvec(M, u), t, n,
                  rel_tol=rel_tol, max_iter=max_iter_factor * max(n, len(idx)))
    r = la.sub(t, la.matvec(M, sol["x"]))
    tn = la.norm(t)
    return _result(sol["x"], residual_rel=(la.norm(r) / tn if tn else 0.0), n_jvp=sol["n_jvp"],
                   n_vjp=sol["n_vjp"], cgls_iters=sol["iters"], cgls_converged=sol["converged"])


def edit_jacobian_local_naive(pb: EditProblem) -> dict:
    """Uses the a.e. derivative of the *post*-activation at x. For an inactive
    ReLU target this row is zero, so the edit cannot move the target at all."""
    J = pb.sae.local_jacobian_rows(pb.x, pb.rows)
    t = [pb.alpha] + [0.0] * len(pb.protected)
    sol = la.lstsq_min_norm(J, t, pb.rtol)
    tn = la.norm(t)
    return _result(sol["x"], residual_rel=la.norm(sol["residual"]) / tn if tn else 0.0,
                   rank=sol["rank"])


def edit_jacobian_ln_repair(pb: EditProblem, max_iters: int = 3) -> dict:
    """Least-norm correction plus a bounded active-set repair: unprotected
    features that switch on/off are added as equality constraints (pre-change 0),
    which is conservative (an inequality would suffice for inactive ones)."""
    extra: List[int] = []
    res = edit_jacobian_ln(pb, extra)
    for it in range(max_iters + 1):
        p_new = pb.sae.pre(la.add(pb.x, res["delta"]))
        m_new = pb.sae.active_mask(p_new)
        crossed = [i for i in range(pb.sae.m)
                   if i != pb.j and i not in pb.protected and i not in extra and m_new[i] != pb.mask[i]]
        if not crossed:
            res["repair_iters"] = it
            res["repair_converged"] = True
            res["n_extra_protected"] = len(extra)
            return res
        if it == max_iters:
            break
        extra.extend(crossed)
        res = edit_jacobian_ln(pb, extra)
    res["repair_iters"] = max_iters
    res["repair_converged"] = False
    res["n_extra_protected"] = len(extra)
    return res


def edit_jacobian_ln_normmatched(pb: EditProblem) -> dict:
    res = edit_jacobian_ln(pb)
    n = la.norm(res["delta"])
    if n == 0.0:
        return _result(None, "FAIL", reason="zero least-norm solution")
    return _result(la.scale(pb.tau / n, res["delta"]))


def edit_trust_region(pb: EditProblem, budget_mult: float = 1.0) -> dict:
    M, t, _ = pb.system()
    tau = budget_mult * pb.tau
    sol = la.trust_region_lstsq(M, t, tau, pb.rtol)
    tn = la.norm(t)
    return _result(sol["x"], lam=sol["lam"], budget_active=sol["active"], budget=tau,
                   residual_rel=la.norm(sol["residual"]) / tn if tn else 0.0)


def edit_random_normmatched(pb: EditProblem, rng: random.Random) -> dict:
    v = [rng.gauss(0.0, 1.0) for _ in range(pb.sae.d)]
    return _result(la.scale(pb.tau / la.norm(v), v))


def run_method(name: str, pb: EditProblem, rng: random.Random, cfg: Optional[dict] = None) -> dict:
    cfg = cfg or {}
    if name == "decoder":
        return edit_decoder(pb)
    if name == "decoder_rescaled":
        return edit_decoder_rescaled(pb)
    if name == "encoder_grad":
        return edit_encoder_grad(pb)
    if name == "jacobian_ln":
        return edit_jacobian_ln(pb)
    if name == "jacobian_ln_cgls":
        return edit_jacobian_ln_cgls(pb, cfg.get("cgls_max_iter_factor", 4), cfg.get("cgls_rel_tol", 1e-12))
    if name == "jacobian_local_naive":
        return edit_jacobian_local_naive(pb)
    if name == "jacobian_ln_repair":
        return edit_jacobian_ln_repair(pb, cfg.get("repair_max_iters", 3))
    if name == "jacobian_ln_normmatched":
        return edit_jacobian_ln_normmatched(pb)
    if name == "trust_region_normmatched":
        return edit_trust_region(pb, 1.0)
    if name.startswith("trust_region_x"):
        return edit_trust_region(pb, float(name[len("trust_region_x"):]))
    if name == "random_normmatched":
        return edit_random_normmatched(pb, rng)
    raise ValueError(name)


# ---------------------------------------------------------------- evaluation


def evaluate(pb: EditProblem, delta: Vector, tol: float = 1e-8) -> Dict[str, float]:
    """Intended vs achieved change after re-encoding ``x + delta``."""
    sae = pb.sae
    p_new = sae.pre(la.add(pb.x, delta))
    a_new = sae.activate(p_new)
    m_new = sae.active_mask(p_new)
    da = la.sub(a_new, pb.a)
    alpha = abs(pb.alpha)
    j = pb.j
    prot = set(pb.protected)
    leak_all = math.sqrt(sum(da[i] ** 2 for i in range(sae.m) if i != j))
    leak_p = math.sqrt(sum(da[i] ** 2 for i in prot))
    leak_u = math.sqrt(sum(da[i] ** 2 for i in range(sae.m) if i != j and i not in prot))
    cross_on = sum(1 for i in range(sae.m) if i != j and m_new[i] and not pb.mask[i])
    cross_off = sum(1 for i in range(sae.m) if i != j and pb.mask[i] and not m_new[i])
    target_crossed = m_new[j] != pb.mask[j]
    # local (a.e.) linear prediction of the post-activation change at x
    J = sae.local_jacobian_rows(pb.x, range(sae.m))
    da_pred = la.matvec(J, delta)
    pred_err = la.norm(la.sub(da, da_pred)) / alpha
    tgt_err = abs(da[j] - pb.alpha) / alpha
    leak_rel = leak_all / alpha
    den = tgt_err ** 2 + leak_rel ** 2
    return {
        "target_achieved": da[j],
        "target_rel_err": tgt_err,
        "leak_rel_all": leak_rel,
        "leak_rel_protected": leak_p / alpha,
        "leak_rel_unprotected": leak_u / alpha,
        "leak_max_rel": max((abs(da[i]) for i in range(sae.m) if i != j), default=0.0) / alpha,
        "mismatch_diag_share": (tgt_err ** 2 / den) if den > 0 else float("nan"),
        "n_crossings": cross_on + cross_off,
        "n_cross_on": cross_on,
        "n_cross_off": cross_off,
        "target_crossed": int(target_crossed),
        "local_pred_err": pred_err,
        "local_pred_exact": int(pred_err <= tol),
        "edit_norm": la.norm(delta),
        "norm_ratio": la.norm(delta) / pb.tau if pb.tau else float("nan"),
    }
