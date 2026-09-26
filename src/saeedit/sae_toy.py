"""Toy SAE: linear encoder rows E (m x d), bias b, decoder D (d x m), activation.

Features are ``a = act(E x + b)``. The decoder is only used to build edits
(``x + alpha * D[:, j]``); "achieved" edits are always measured by re-encoding.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence

from . import linalg as la
from .linalg import Matrix, Vector

ACTIVATIONS = ("identity", "relu", "jumprelu", "topk")


@dataclass
class ToySAE:
    E: Matrix            # m x d encoder rows
    b: Vector            # m
    D: Matrix            # d x m, columns are decoder directions
    act: str = "identity"
    theta: float = 0.0   # JumpReLU threshold
    k: int = 0           # TopK size
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.act not in ACTIVATIONS:
            raise ValueError(f"unknown activation {self.act!r}")

    @property
    def m(self) -> int:
        return len(self.E)

    @property
    def d(self) -> int:
        return len(self.E[0])

    def pre(self, x: Sequence[float]) -> Vector:
        return la.add(la.matvec(self.E, x), self.b)

    def active_mask(self, p: Sequence[float]) -> List[bool]:
        if self.act == "identity":
            return [True] * len(p)
        if self.act == "relu":
            return [v > 0.0 for v in p]
        if self.act == "jumprelu":
            return [v > self.theta for v in p]
        # topk: the k largest strictly positive pre-activations (ties broken by index)
        order = sorted(range(len(p)), key=lambda i: (-p[i], i))
        keep = set(i for i in order[: self.k] if p[i] > 0.0)
        return [i in keep for i in range(len(p))]

    def activate(self, p: Sequence[float]) -> Vector:
        if self.act == "identity":
            return list(p)
        mask = self.active_mask(p)
        return [v if on else 0.0 for v, on in zip(p, mask)]

    def encode(self, x: Sequence[float]) -> Vector:
        return self.activate(self.pre(x))

    def decoder_col(self, j: int) -> Vector:
        return la.column(self.D, j)

    def local_jacobian_rows(self, x: Sequence[float], idx: Sequence[int]) -> Matrix:
        """Rows of d a / d x at ``x`` (a.e. derivative). Inactive rows are zero.

        For JumpReLU/TopK this derivative ignores the jumps at the active-set
        boundary: it is a local derivative, not a global inverse.
        """
        mask = self.active_mask(self.pre(x))
        return [list(self.E[i]) if mask[i] else [0.0] * self.d for i in idx]


# ---------------------------------------------------------------- builders


def gaussian_vec(n: int, rng: random.Random) -> Vector:
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def normalize(v: Sequence[float]) -> Vector:
    nv = la.norm(v)
    return [x / nv for x in v]


def random_orthonormal(d: int, k: int, rng: random.Random) -> Matrix:
    """d x k matrix with orthonormal columns (modified Gram-Schmidt, applied twice)."""
    if k > d:
        raise ValueError("k > d")
    cols: List[Vector] = []
    while len(cols) < k:
        v = gaussian_vec(d, rng)
        for _ in range(2):
            for q in cols:
                v = la.axpy(-la.dot(q, v), q, v)
        nv = la.norm(v)
        if nv < 1e-8:
            continue
        cols.append([x / nv for x in v])
    return la.transpose(cols)


def orthonormal_decoder(d: int, m: int, rng: random.Random) -> Matrix:
    return random_orthonormal(d, m, rng)


def equiangular_decoder(d: int, m: int, c: float, rng: random.Random) -> Matrix:
    """Unit-norm columns with every pairwise cosine exactly ``c`` (0 <= c < 1).

    D_j = sqrt(1-c) q_j + sqrt(c) q_0 with orthonormal q_0..q_m; needs m+1 <= d.
    """
    if m + 1 > d:
        raise ValueError("equiangular construction needs m + 1 <= d")
    if not 0.0 <= c < 1.0:
        raise ValueError("coherence must be in [0, 1)")
    Q = random_orthonormal(d, m + 1, rng)
    a, g = math.sqrt(1.0 - c), math.sqrt(c)
    return [[a * Q[i][j + 1] + g * Q[i][0] for j in range(m)] for i in range(d)]


def gaussian_decoder(d: int, m: int, rng: random.Random) -> Matrix:
    cols = [normalize(gaussian_vec(d, rng)) for _ in range(m)]
    return la.transpose(cols)


def build_decoder(kind: str, d: int, m: int, rng: random.Random, coherence: float = 0.0) -> Matrix:
    if kind == "orthonormal":
        return orthonormal_decoder(d, m, rng)
    if kind == "equiangular":
        return equiangular_decoder(d, m, coherence, rng)
    if kind == "gaussian":
        return gaussian_decoder(d, m, rng)
    raise ValueError(kind)


def log_uniform(lo: float, hi: float, rng: random.Random) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def build_encoder(kind: str, D: Matrix, rng: random.Random, beta: float = 0.5,
                  scale_range: Sequence[float] | None = None, rtol: float = 1e-10) -> Matrix:
    """Encoder rows from a decoder.

    tied        E = D^T
    pinv        E = D^+ (biorthogonal when D has full column rank)
    scaled_tied E = diag(s) D^T
    interp      E = diag(s) ((1-beta) D^T + beta D^+)
    ``s`` is log-uniform in ``scale_range`` (all ones when None, except scaled_tied needs it).
    """
    Dt = la.transpose(D)
    if kind == "tied":
        base = Dt
    elif kind == "pinv":
        base = la.pinv(D, rtol)
    elif kind == "scaled_tied":
        base = Dt
        if scale_range is None:
            raise ValueError("scaled_tied needs scale_range")
    elif kind == "interp":
        P = la.pinv(D, rtol)
        base = [[(1.0 - beta) * a + beta * p for a, p in zip(ra, rp)] for ra, rp in zip(Dt, P)]
    else:
        raise ValueError(kind)
    if scale_range is None:
        return [list(row) for row in base]
    s = [log_uniform(scale_range[0], scale_range[1], rng) for _ in base]
    return [la.scale(si, row) for si, row in zip(s, base)]


def sparse_base_point(D: Matrix, support: int, coef_range: Sequence[float], noise: float,
                      rng: random.Random) -> Vector:
    """x = sum_{j in S} z_j D_j + noise, |S| = support, z_j ~ U(coef_range)."""
    d, m = la.shape(D)
    idx = rng.sample(range(m), support)
    x = [0.0] * d
    for j in idx:
        x = la.axpy(rng.uniform(coef_range[0], coef_range[1]), la.column(D, j), x)
    return la.axpy(noise, gaussian_vec(d, rng), x)
