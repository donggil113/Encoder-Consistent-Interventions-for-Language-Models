"""Numerical checks of the stdlib linear algebra. Checks, not proofs."""

import math
import random
import unittest

from saeedit import linalg as la


def gmat(r, c, rng):
    return [[rng.gauss(0, 1) for _ in range(c)] for _ in range(r)]


class TestSVD(unittest.TestCase):
    def test_reconstruction_tall_and_wide(self):
        rng = random.Random(0)
        for r, c in [(7, 4), (4, 7), (5, 5), (1, 3), (3, 1)]:
            A = gmat(r, c, rng)
            U, s, V = la.svd(A)
            k = min(r, c)
            self.assertEqual(len(s), k)
            self.assertTrue(all(s[i] >= s[i + 1] for i in range(k - 1)))
            R = [[sum(U[i][t] * s[t] * V[j][t] for t in range(k)) for j in range(c)] for i in range(r)]
            self.assertLess(la.max_abs_diff(A, R), 1e-10)
            # orthonormal columns of U and V
            for M in (U, V):
                G = la.matmul(la.transpose(M), M)
                self.assertLess(la.max_abs_diff(G, la.eye(k)), 1e-10)

    def test_rank_of_product(self):
        rng = random.Random(1)
        A = la.matmul(gmat(9, 3, rng), gmat(3, 6, rng))
        self.assertEqual(la.rank(A), 3)

    def test_penrose_conditions(self):
        rng = random.Random(2)
        for A in (gmat(6, 4, rng), gmat(4, 6, rng), la.matmul(gmat(6, 2, rng), gmat(2, 5, rng))):
            P = la.pinv(A)
            APA = la.matmul(la.matmul(A, P), A)
            PAP = la.matmul(la.matmul(P, A), P)
            AP = la.matmul(A, P)
            PA = la.matmul(P, A)
            self.assertLess(la.max_abs_diff(APA, A), 1e-9)
            self.assertLess(la.max_abs_diff(PAP, P), 1e-9)
            self.assertLess(la.max_abs_diff(AP, la.transpose(AP)), 1e-9)
            self.assertLess(la.max_abs_diff(PA, la.transpose(PA)), 1e-9)


class TestSolvers(unittest.TestCase):
    def test_least_norm_full_row_rank(self):
        rng = random.Random(3)
        M = gmat(4, 9, rng)
        t = [rng.gauss(0, 1) for _ in range(4)]
        sol = la.lstsq_min_norm(M, t)
        self.assertLess(la.norm(sol["residual"]), 1e-10)
        # closed form M^T (M M^T)^-1 t
        MMt = la.matmul(M, la.transpose(M))
        y = la.matvec(la.pinv(MMt), t)
        x_cf = la.rmatvec(M, y)
        self.assertLess(la.rel_diff(sol["x"], x_cf), 1e-9)

    def test_infeasible_certificate(self):
        rng = random.Random(4)
        M = gmat(12, 5, rng)
        t = [rng.gauss(0, 1) for _ in range(12)]
        sol = la.lstsq_min_norm(M, t)
        r = sol["residual"]
        self.assertGreater(la.norm(r), 1e-3)
        self.assertLess(la.norm(la.rmatvec(M, r)) / (la.frob(M) * la.norm(r)), 1e-10)
        self.assertAlmostEqual(la.dot(r, t), la.dot(r, r), places=9)

    def test_cgls_matches_dense_including_rank_deficient(self):
        rng = random.Random(5)
        for M in (gmat(5, 8, rng), gmat(10, 6, rng), la.matmul(gmat(7, 3, rng), gmat(3, 8, rng))):
            t = [rng.gauss(0, 1) for _ in range(len(M))]
            dense = la.lstsq_min_norm(M, t)["x"]
            cg = la.cgls(lambda v: la.matvec(M, v), lambda u: la.rmatvec(M, u), t, len(M[0]),
                         max_iter=200)
            self.assertLess(la.rel_diff(cg["x"], dense), 1e-7)

    def test_trust_region(self):
        rng = random.Random(6)
        M = gmat(3, 7, rng)
        t = [rng.gauss(0, 1) for _ in range(3)]
        x_ln = la.lstsq_min_norm(M, t)["x"]
        n_ln = la.norm(x_ln)
        # inactive budget -> least-norm solution
        sol = la.trust_region_lstsq(M, t, 2 * n_ln)
        self.assertFalse(sol["active"])
        self.assertLess(la.rel_diff(sol["x"], x_ln), 1e-12)
        # active budget -> on the boundary, satisfies ridge KKT
        tau = 0.5 * n_ln
        sol = la.trust_region_lstsq(M, t, tau)
        self.assertTrue(sol["active"])
        self.assertLessEqual(la.norm(sol["x"]), tau * (1 + 1e-12))
        self.assertAlmostEqual(la.norm(sol["x"]), tau, places=8)
        lam = sol["lam"]
        # (M^T M + lam I) x = M^T t
        lhs = la.add(la.rmatvec(M, la.matvec(M, sol["x"])), la.scale(lam, sol["x"]))
        self.assertLess(la.rel_diff(lhs, la.rmatvec(M, t)), 1e-7)
        # no feasible point with norm <= tau does better (random probes)
        best = la.norm(sol["residual"])
        for _ in range(200):
            z = [rng.gauss(0, 1) for _ in range(7)]
            z = la.scale(tau * rng.random() / la.norm(z), z)
            self.assertGreaterEqual(la.norm(la.sub(t, la.matvec(M, z))), best - 1e-9)


if __name__ == "__main__":
    unittest.main()
