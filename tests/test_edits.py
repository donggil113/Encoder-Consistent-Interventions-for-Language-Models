"""Checks of edit constructors against closed forms. Checks, not proofs."""

import math
import random
import unittest

from saeedit import linalg as la
from saeedit.edits import EditProblem, evaluate, run_method
from saeedit.sae_toy import (ToySAE, build_decoder, build_encoder, equiangular_decoder, gaussian_vec,
                             sparse_base_point)


def linear_sae(D, E):
    return ToySAE(E=E, b=[0.0] * len(E), D=D, act="identity")


class TestLinearToy(unittest.TestCase):
    def test_equiangular_gram(self):
        rng = random.Random(0)
        D = equiangular_decoder(10, 6, 0.3, rng)
        G = la.matmul(la.transpose(D), D)
        for i in range(6):
            for j in range(6):
                self.assertAlmostEqual(G[i][j], 1.0 if i == j else 0.3, places=12)

    def test_pinv_encoder_makes_decoder_edit_exact(self):
        rng = random.Random(1)
        D = equiangular_decoder(12, 6, 0.4, rng)
        sae = linear_sae(D, build_encoder("pinv", D, rng))
        pb = EditProblem(sae, gaussian_vec(12, rng), 2, 1.5, [i for i in range(6) if i != 2])
        dec = run_method("decoder", pb, rng)["delta"]
        ln = run_method("jacobian_ln", pb, rng)["delta"]
        ev = evaluate(pb, dec)
        self.assertLess(ev["target_rel_err"], 1e-10)
        self.assertLess(ev["leak_rel_all"], 1e-10)
        self.assertLess(la.rel_diff(dec, ln), 1e-10)  # (D^+)^+ = D

    def test_scaled_orthogonal_rescaling_suffices(self):
        rng = random.Random(2)
        D = build_decoder("orthonormal", 10, 5, rng)
        sae = linear_sae(D, build_encoder("scaled_tied", D, rng, scale_range=(0.5, 2.0)))
        pb = EditProblem(sae, gaussian_vec(10, rng), 1, 1.0, [0, 2, 3, 4])
        dec = evaluate(pb, run_method("decoder", pb, rng)["delta"])
        self.assertGreater(dec["target_rel_err"], 1e-3)       # scale mismatch exists
        self.assertLess(dec["leak_rel_all"], 1e-10)           # but no interference
        resc = run_method("decoder_rescaled", pb, rng)["delta"]
        ln = run_method("jacobian_ln", pb, rng)["delta"]
        self.assertLess(evaluate(pb, resc)["target_rel_err"], 1e-10)
        self.assertLess(la.rel_diff(resc, ln), 1e-10)

    def test_tied_coherent_interference(self):
        rng = random.Random(3)
        c, m = 0.25, 6
        D = equiangular_decoder(12, m, c, rng)
        sae = linear_sae(D, build_encoder("tied", D, rng))
        j = 0
        pb = EditProblem(sae, gaussian_vec(12, rng), j, 1.0, list(range(1, m)))
        for meth in ("decoder", "decoder_rescaled"):
            ev = evaluate(pb, run_method(meth, pb, rng)["delta"])
            self.assertAlmostEqual(ev["leak_rel_all"], c * math.sqrt(m - 1), places=10)
            self.assertLess(ev["target_rel_err"], 1e-10)
        ev = evaluate(pb, run_method("jacobian_ln", pb, rng)["delta"])
        self.assertLess(ev["leak_rel_all"], 1e-10)
        Ginv = la.pinv(la.matmul(la.transpose(D), D))
        self.assertAlmostEqual(ev["norm_ratio"], math.sqrt(Ginv[j][j]), places=10)
        self.assertGreater(ev["norm_ratio"], 1.0)

    def test_cgls_equals_dense_edit(self):
        rng = random.Random(4)
        D = build_decoder("gaussian", 16, 40, rng)
        sae = linear_sae(D, build_encoder("tied", D, rng))
        pb = EditProblem(sae, gaussian_vec(16, rng), 3, 1.0, [i for i in range(40) if i != 3])
        a = run_method("jacobian_ln", pb, rng)
        b = run_method("jacobian_ln_cgls", pb, rng)
        self.assertGreater(a["residual_rel"], 1e-3)  # protect-all with m > d is infeasible
        self.assertLess(la.rel_diff(a["delta"], b["delta"]), 1e-6)


class TestNonlinear(unittest.TestCase):
    def _relu_sae(self, rng):
        D = build_decoder("gaussian", 12, 30, rng)
        E = build_encoder("interp", D, rng, beta=0.5, scale_range=(0.7, 1.4))
        return ToySAE(E=E, b=[-0.25] * 30, D=D, act="relu")

    def test_relu_affine_exactness_without_crossing(self):
        rng = random.Random(5)
        sae = self._relu_sae(rng)
        n_checked = 0
        for _ in range(40):
            x = sparse_base_point(sae.D, 3, (0.5, 1.5), 0.02, rng)
            mask = sae.active_mask(sae.pre(x))
            act = [i for i in range(30) if mask[i]]
            if not act:
                continue
            j = act[0]
            pb = EditProblem(sae, x, j, 0.05, [i for i in act if i != j])
            delta = run_method("jacobian_ln", pb, rng)["delta"]
            ev = evaluate(pb, delta)
            if ev["n_crossings"] == 0 and ev["target_crossed"] == 0:
                n_checked += 1
                self.assertEqual(ev["local_pred_exact"], 1)
                self.assertLess(ev["target_rel_err"], 1e-8)
                self.assertLess(ev["leak_rel_all"], 1e-8)
        self.assertGreater(n_checked, 0)

    def test_naive_local_jacobian_cannot_activate_dead_relu(self):
        rng = random.Random(6)
        sae = self._relu_sae(rng)
        x = sparse_base_point(sae.D, 3, (0.5, 1.5), 0.02, rng)
        mask = sae.active_mask(sae.pre(x))
        j = next(i for i in range(30) if not mask[i])
        pb = EditProblem(sae, x, j, 1.0, [i for i in range(30) if mask[i]])
        naive = evaluate(pb, run_method("jacobian_local_naive", pb, rng)["delta"])
        self.assertAlmostEqual(naive["target_achieved"], 0.0, places=12)
        piece = evaluate(pb, run_method("jacobian_ln", pb, rng)["delta"])
        self.assertLess(piece["target_rel_err"], 1e-8)

    def test_repair_never_adds_target_or_protected(self):
        rng = random.Random(7)
        sae = self._relu_sae(rng)
        x = sparse_base_point(sae.D, 3, (0.5, 1.5), 0.02, rng)
        mask = sae.active_mask(sae.pre(x))
        act = [i for i in range(30) if mask[i]]
        j = next(i for i in range(30) if not mask[i])
        pb = EditProblem(sae, x, j, 4.0, act)
        res = run_method("jacobian_ln_repair", pb, rng, {"repair_max_iters": 3})
        self.assertIn("repair_converged", res)
        self.assertLessEqual(res["repair_iters"], 3)

    def test_topk_active_count_is_fixed(self):
        rng = random.Random(8)
        D = build_decoder("gaussian", 8, 12, rng)
        sae = ToySAE(E=build_encoder("tied", D, rng), b=[0.0] * 12, D=D, act="topk", k=3)
        for _ in range(30):
            x = gaussian_vec(8, rng)
            self.assertLessEqual(sum(sae.active_mask(sae.pre(x))), 3)

    def test_jumprelu_values_below_threshold_unattainable(self):
        rng = random.Random(9)
        D = build_decoder("gaussian", 8, 12, rng)
        sae = ToySAE(E=build_encoder("tied", D, rng), b=[0.0] * 12, D=D, act="jumprelu", theta=0.5)
        x = gaussian_vec(8, rng)
        p = sae.pre(x)
        j = next(i for i in range(12) if p[i] <= 0.5)
        pb = EditProblem(sae, x, j, 0.3, [])
        self.assertFalse(pb.target_attainable())
        a_new = sae.encode(la.add(x, run_method("jacobian_ln", pb, rng)["delta"]))
        self.assertEqual(a_new[j], 0.0)


if __name__ == "__main__":
    unittest.main()
