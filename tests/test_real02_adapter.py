"""Model-level checks of the REAL-02(-LX) adapter on a fixed synthetic prompt (no WikiText text).

Runs only with P3_MODEL_TESTS=1 and the pinned files already in the HF cache (HF_HUB_OFFLINE=1);
otherwise skipped (NOT_RUN).
"""

import importlib.util
import os
import unittest

RUN = os.environ.get("P3_MODEL_TESTS") == "1" and importlib.util.find_spec("torch") is not None


@unittest.skipUnless(RUN, "set P3_MODEL_TESTS=1 with torch and the cached pinned model/SAE: NOT_RUN")
class TestReal02Adapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        from saeedit.real.backend import Backend
        from saeedit.real.contract import load_contract

        cls.torch = torch
        cls.be = Backend(load_contract("configs/p3_contract_gpt2_res_jb_l8.json"), None, False, 2, "float64")
        text = "The history of the old harbour town begins in the early medieval period, when"
        cls.ids = torch.tensor([[cls.be.bos_id] + cls.be.tokenize(text)])
        cls.j = 1000
        cls.vecs = {"decoder": cls.be.dec_row(cls.j), "encoder_grad": cls.be.enc_col(cls.j),
                    "diffmean": torch.randn(768, generator=torch.Generator().manual_seed(1), dtype=torch.float64)}

    def test_no_edit_generation_matches_hf_greedy(self):
        from saeedit.real.real02 import _generate

        ours = _generate(self.be, self.ids, "no_edit", 0.0, self.j, self.vecs, 8)
        with self.torch.no_grad():
            ref = self.be.model.generate(self.ids, attention_mask=self.torch.ones_like(self.ids), do_sample=False,
                                         max_new_tokens=8, pad_token_id=self.be.bos_id)
        self.assertEqual(ours.tolist(), ref.tolist())

    def test_ln_cache_equals_recomputation(self):
        from saeedit.real.real02 import _deltas

        x = self.be.center(self.be.hook_states(self.ids)[0])
        cache = []
        _deltas(self.be, "jacobian_ln", x[:5], 3.0, self.j, self.vecs, cache)
        inc = _deltas(self.be, "jacobian_ln", x, 3.0, self.j, self.vecs, cache)
        full = _deltas(self.be, "jacobian_ln", x, 3.0, self.j, self.vecs)
        self.assertLess(float((inc - full).abs().max()), 1e-12)

    def test_actor_edits_admissible_and_mean_only_invisible(self):
        from saeedit.real.real02 import _generate, _kl_all_positions, _deltas

        x = self.be.center(self.be.hook_states(self.ids)[0])
        for m in ("decoder", "encoder_grad", "jacobian_ln", "diffmean"):
            dl = _deltas(self.be, m, x, 5.0, self.j, self.vecs)
            self.assertLess(float(dl.mean(-1).abs().max()), 1e-12, m)
            self.assertAlmostEqual(float(dl.norm(dim=-1).max()), 5.0, places=9)
        self.assertLess(_kl_all_positions(self.be, self.ids, "mean_only", 80.0, self.j, self.vecs), 1e-12)
        a = _generate(self.be, self.ids, "mean_only", 80.0, self.j, self.vecs, 8)
        b = _generate(self.be, self.ids, "no_edit", 0.0, self.j, self.vecs, 8)
        self.assertEqual(a.tolist(), b.tolist())


if __name__ == "__main__":
    unittest.main()
