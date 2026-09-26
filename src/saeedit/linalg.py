"""Minimal dense linear algebra in pure Python (standard library only).

The approved first-run environment has no numpy/torch, so matrices are
row-major ``list[list[float]]`` and vectors are ``list[float]``. Toy sizes are
at most 64 x 64, so O(n^3) pure-Python routines are adequate.

Routines here are verified numerically in ``tests/``; numerical agreement is a
check, not a proof of correctness.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

Matrix = List[List[float]]
Vector = List[float]


# ---------------------------------------------------------------- basics


def zeros(r: int, c: int) -> Matrix:
    return [[0.0] * c for _ in range(r)]


def eye(n: int) -> Matrix:
    out = zeros(n, n)
    for i in range(n):
        out[i][i] = 1.0
    return out


def shape(A: Sequence[Sequence[float]]) -> Tuple[int, int]:
    return (len(A), len(A[0]) if A else 0)


def transpose(A: Matrix) -> Matrix:
    return [list(col) for col in zip(*A)]


def dot(u: Sequence[float], v: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def norm(v: Sequence[float]) -> float:
    return math.sqrt(dot(v, v))


def add(u: Sequence[float], v: Sequence[float]) -> Vector:
    return [a + b for a, b in zip(u, v)]


def sub(u: Sequence[float], v: Sequence[float]) -> Vector:
    return [a - b for a, b in zip(u, v)]


def scale(a: float, v: Sequence[float]) -> Vector:
    return [a * x for x in v]


def axpy(a: float, x: Sequence[float], y: Sequence[float]) -> Vector:
    """Return a*x + y."""
    return [a * xi + yi for xi, yi in zip(x, y)]


def matvec(A: Matrix, v: Sequence[float]) -> Vector:
    return [sum(a * b for a, b in zip(row, v)) for row in A]


def rmatvec(A: Matrix, u: Sequence[float]) -> Vector:
    """Return A^T u without forming A^T."""
    c = len(A[0])
    out = [0.0] * c
    for ui, row in zip(u, A):
        if ui != 0.0:
            for k, a in enumerate(row):
                out[k] += ui * a
    return out


def matmul(A: Matrix, B: Matrix) -> Matrix:
    Bt = transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def column(A: Matrix, j: int) -> Vector:
    return [row[j] for row in A]


def take_rows(A: Matrix, idx: Sequence[int]) -> Matrix:
    return [list(A[i]) for i in idx]


def frob(A: Matrix) -> float:
    return math.sqrt(sum(a * a for row in A for a in row))


def max_abs_diff(A: Matrix, B: Matrix) -> float:
    return max(abs(a - b) for ra, rb in zip(A, B) for a, b in zip(ra, rb))


def rel_diff(u: Sequence[float], v: Sequence[float]) -> float:
    """||u - v|| / max(||u||, ||v||, tiny)."""
    den = max(norm(u), norm(v), 1e-300)
    return norm(sub(u, v)) / den


# ---------------------------------------------------------------- SVD


def svd(A: Matrix, tol: float = 1e-14, max_sweeps: int = 80) -> Tuple[Matrix, Vector, Matrix]:
    """Thin SVD via one-sided (Hestenes) Jacobi rotations.

    Returns ``(U, s, V)`` with ``A = U diag(s) V^T``; ``U`` is r x k,
    ``V`` is c x k, ``k = min(r, c)``, ``s`` is sorted in descending order.
    Columns of ``U`` belonging to (numerically) zero singular values are zero
    vectors; callers must threshold ``s`` before using them.
    """
    r, c = shape(A)
    if r < c:
        U, s, V = svd(transpose(A), tol, max_sweeps)
        return V, s, U

    cols = [column(A, j) for j in range(c)]
    vcols = [[1.0 if i == j else 0.0 for i in range(c)] for j in range(c)]
    for _ in range(max_sweeps):
        rotated = False
        for p in range(c - 1):
            for q in range(p + 1, c):
                ap, aq = cols[p], cols[q]
                alpha = dot(ap, ap)
                beta = dot(aq, aq)
                if alpha == 0.0 or beta == 0.0:
                    continue
                gamma = dot(ap, aq)
                if abs(gamma) <= tol * math.sqrt(alpha * beta):
                    continue
                rotated = True
                zeta = (beta - alpha) / (2.0 * gamma)
                t = math.copysign(1.0, zeta) / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                cs = 1.0 / math.sqrt(1.0 + t * t)
                sn = cs * t
                cols[p] = [cs * x - sn * y for x, y in zip(ap, aq)]
                cols[q] = [sn * x + cs * y for x, y in zip(ap, aq)]
                vp, vq = vcols[p], vcols[q]
                vcols[p] = [cs * x - sn * y for x, y in zip(vp, vq)]
                vcols[q] = [sn * x + cs * y for x, y in zip(vp, vq)]
        if not rotated:
            break

    svals = [norm(col) for col in cols]
    order = sorted(range(c), key=lambda j: -svals[j])
    s = [svals[j] for j in order]
    ucols = []
    for j in order:
        sj = svals[j]
        ucols.append([x / sj for x in cols[j]] if sj > 0.0 else [0.0] * r)
    U = transpose(ucols)
    V = transpose([vcols[j] for j in order])
    return U, s, V


def _threshold(s: Vector, rtol: float) -> float:
    return (s[0] if s else 0.0) * rtol


def rank(A: Matrix, rtol: float = 1e-10) -> int:
    _, s, _ = svd(A)
    thr = _threshold(s, rtol)
    return sum(1 for x in s if x > thr)


def pinv(A: Matrix, rtol: float = 1e-10) -> Matrix:
    """Moore-Penrose pseudo-inverse (c x r) with relative cutoff ``rtol``."""
    U, s, V = svd(A)
    r, c = shape(A)
    thr = _threshold(s, rtol)
    out = zeros(c, r)
    for k, sk in enumerate(s):
        if sk <= thr:
            continue
        inv = 1.0 / sk
        for i in range(c):
            vik = V[i][k] * inv
            if vik == 0.0:
                continue
            row = out[i]
            for j in range(r):
                row[j] += vik * U[j][k]
    return out


def lstsq_min_norm(A: Matrix, b: Sequence[float], rtol: float = 1e-10) -> dict:
    """Minimum-norm least-squares solution of ``A x ~= b`` via the SVD.

    Returns a dict with ``x``, ``residual`` (b - A x), ``rank`` and the
    singular values. When ``residual`` is nonzero it is itself a Fredholm
    certificate of infeasibility: ``A^T r = 0`` and ``r . b = ||r||^2 > 0``.
    """
    U, s, V = svd(A)
    thr = _threshold(s, rtol)
    c = shape(A)[1]
    x = [0.0] * c
    rnk = 0
    for k, sk in enumerate(s):
        if sk <= thr:
            continue
        rnk += 1
        coef = sum(U[i][k] * b[i] for i in range(len(b))) / sk
        for i in range(c):
            x[i] += coef * V[i][k]
    residual = sub(b, matvec(A, x))
    return {"x": x, "residual": residual, "rank": rnk, "s": s}


def trust_region_lstsq(A: Matrix, b: Sequence[float], tau: float, rtol: float = 1e-10,
                       iters: int = 200) -> dict:
    """Solve ``min ||A x - b||`` subject to ``||x|| <= tau``.

    Convex trust-region subproblem: the solution is the min-norm LS solution
    if its norm is <= tau (lambda = 0); otherwise it is the ridge solution
    ``x(lam) = (A^T A + lam I)^-1 A^T b`` with ``||x(lam)|| = tau`` (lam > 0),
    found by bisection. The returned point always satisfies ``||x|| <= tau``.
    """
    U, s, V = svd(A)
    thr = _threshold(s, rtol)
    c = shape(A)[1]
    comps = []  # (s_k, u_k . b, k)
    for k, sk in enumerate(s):
        if sk > thr:
            comps.append((sk, sum(U[i][k] * b[i] for i in range(len(b))), k))

    def solve(lam: float) -> Vector:
        x = [0.0] * c
        for sk, ck, k in comps:
            coef = sk * ck / (sk * sk + lam) if lam > 0.0 else ck / sk
            for i in range(c):
                x[i] += coef * V[i][k]
        return x

    def xnorm(lam: float) -> float:
        tot = 0.0
        for sk, ck, _ in comps:
            coef = sk * ck / (sk * sk + lam) if lam > 0.0 else ck / sk
            tot += coef * coef
        return math.sqrt(tot)

    n0 = xnorm(0.0)
    if n0 <= tau:
        x = solve(0.0)
        return {"x": x, "lam": 0.0, "active": False, "residual": sub(b, matvec(A, x))}
    atb = math.sqrt(sum((sk * ck) ** 2 for sk, ck, _ in comps))
    lo, hi = 0.0, atb / max(tau, 1e-300)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if xnorm(mid) > tau:
            lo = mid
        else:
            hi = mid
    x = solve(hi)
    return {"x": x, "lam": hi, "active": True, "residual": sub(b, matvec(A, x))}


# ---------------------------------------------------------------- matrix-free


def cgls(jvp: Callable[[Vector], Vector], vjp: Callable[[Vector], Vector], b: Sequence[float],
         n: int, damp: float = 0.0, rel_tol: float = 1e-12, max_iter: int = 100) -> dict:
    """CGLS for ``min ||A x - b||^2 + damp ||x||^2`` using only ``A v`` and ``A^T u``.

    Started at ``x0 = 0`` with ``damp = 0``, the iterates stay in range(A^T),
    so in exact arithmetic the limit is the minimum-norm least-squares
    solution, also for rank-deficient or inconsistent systems.
    ``jvp``/``vjp`` stand for Jacobian-vector and vector-Jacobian products;
    the number of calls is returned so compute cost can be reported.
    """
    x = [0.0] * n
    r = list(b)
    s = vjp(r)
    n_jvp, n_vjp = 0, 1
    p = list(s)
    gamma = dot(s, s)
    s0 = math.sqrt(gamma)
    it = 0
    if s0 == 0.0:
        return {"x": x, "iters": 0, "n_jvp": n_jvp, "n_vjp": n_vjp, "converged": True}
    converged = False
    for it in range(1, max_iter + 1):
        q = jvp(p)
        n_jvp += 1
        delta = dot(q, q) + damp * dot(p, p)
        if delta <= 0.0:
            converged = True
            break
        alpha = gamma / delta
        x = axpy(alpha, p, x)
        r = axpy(-alpha, q, r)
        s = vjp(r)
        n_vjp += 1
        if damp:
            s = axpy(-damp, x, s)
        gamma_new = dot(s, s)
        if math.sqrt(gamma_new) <= rel_tol * s0:
            converged = True
            break
        p = axpy(gamma_new / gamma, p, s)
        gamma = gamma_new
    return {"x": x, "iters": it, "n_jvp": n_jvp, "n_vjp": n_vjp, "converged": converged}
