"""Torch-side checks. Skipped (NOT_RUN) when torch is not installed."""

import importlib.util
import random
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch not installed (installation not approved): NOT_RUN")
class TestDriftTorch(unittest.TestCase):
    def test_batched_matches_reference(self):
        import torch

        from saeedit.real.drift import decompose, decompose_torch

        g = torch.Generator().manual_seed(0)
        B, m = 6, 50
        p0 = torch.randn(B, m, generator=g)
        p1 = p0 + 0.7 * torch.randn(B, m, generator=g)
        a0, a1 = p0.clamp_min(0), p1.clamp_min(0)
        m0, m1 = p0 > 0, p1 > 0
        j = torch.tensor([random.Random(i).randrange(m) for i in range(B)])
        alpha = torch.full((B,), 0.8)
        scale = torch.full((B,), 0.8)
        out = decompose_torch(a0, a1, m0, m1, j, alpha, scale)
        for b in range(B):
            ref = decompose(a0[b].tolist(), a1[b].tolist(), m0[b].tolist(), m1[b].tolist(), int(j[b]), 0.8, 0.8)
            for k, v in ref.items():
                self.assertAlmostEqual(float(out[k][b]), float(v), places=5, msg=k)


if __name__ == "__main__":
    unittest.main()
