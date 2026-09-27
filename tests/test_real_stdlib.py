"""Standard-library parts of the real-model adapters (run in the approved environment)."""

import json
import os
import random
import struct
import tempfile
import unittest

from saeedit import linalg as la
from saeedit.edits import EditProblem, evaluate, run_method
from saeedit.real import contract as C
from saeedit.real import data as D
from saeedit.real import dose as Q
from saeedit.real.drift import decompose
from saeedit.real.lexicon import WEDDING_ACTADD, bounded_hits, bounded_label, count_hits, label
from saeedit.real.stats import cluster_bootstrap, paired_unit_differences, per_unit_means
from saeedit.sae_toy import ToySAE, build_decoder, build_encoder, sparse_base_point

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = C.load_contract(os.path.join(ROOT, "configs", "p3_contract_gpt2_res_jb_l8.json"))
FIX = os.path.join(ROOT, "tests", "fixtures")


def write_safetensors(path, tensors):
    """Minimal safetensors writer (header + zero bytes) for header tests."""
    header, off = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = 4
        for s in shape:
            n *= s
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * off)


class TestContract(unittest.TestCase):
    def test_real_sae_cfg_matches_contract(self):
        """C1 executed on the verbatim cfg.json from the pinned SAE revision."""
        with open(os.path.join(FIX, "res_jb_blocks.8.hook_resid_pre.cfg.json")) as f:
            cfg = json.load(f)
        self.assertEqual(C.check_sae_cfg(cfg, CONTRACT), [])
        for k in CONTRACT["sae"]["cfg_json_absent_fields"]:
            for name in k.split(" / "):
                self.assertNotIn(name, cfg)  # these really are absent -> SAELens defaults apply

    def test_real_gpt2_config_matches_contract(self):
        with open(os.path.join(FIX, "gpt2_config.json")) as f:
            self.assertEqual(C.check_gpt2_config(json.load(f), CONTRACT), [])

    def test_cfg_violation_detected(self):
        with open(os.path.join(FIX, "res_jb_blocks.8.hook_resid_pre.cfg.json")) as f:
            cfg = json.load(f)
        cfg["hook_point"] = "blocks.9.hook_resid_pre"
        cfg["activation_fn_str"] = "topk"
        errs = C.check_sae_cfg(cfg, CONTRACT)
        self.assertTrue(any("hook_point" in e for e in errs))
        self.assertTrue(any("ReLU" in e for e in errs))

    def test_safetensors_header_expected_and_size(self):
        exp = CONTRACT["sae"]["expected_tensors_INFERRED"]
        tens = {k: ("F32", v) for k, v in exp.items() if isinstance(v, list)}
        n_bytes = sum(4 * (v[0] * (v[1] if len(v) > 1 else 1)) for _, v in tens.values())
        listed = CONTRACT["sae"]["file_sizes_bytes"]["blocks.8.hook_resid_pre/sae_weights.safetensors"]
        self.assertEqual(listed - n_bytes, 320)  # size arithmetic behind the INFERRED shapes
        with tempfile.TemporaryDirectory() as d:
            small = {k: ("F32", [2] * len(v)) for k, (_, v) in tens.items()}
            p = os.path.join(d, "x.safetensors")
            write_safetensors(p, small)
            errs = C.check_safetensors_header(C.read_safetensors_header(p), CONTRACT)
            self.assertTrue(errs and all("shape" in e for e in errs))
            write_safetensors(p, {"W_enc": ("F32", [768, 24576])})
            errs = C.check_safetensors_header(C.read_safetensors_header(p), CONTRACT)
            self.assertTrue(any("tensor names" in e for e in errs))

    def test_environment_status_reports_missing(self):
        env = C.environment_status()
        self.assertIn("all_available", env)


class TestData(unittest.TestCase):
    LINES = [" = Alpha = \n", " text a1 \n", " = = Sub = = \n", " text a2 \n", " = Beta = \n", " text b \n"]

    def test_segment(self):
        docs = D.segment_wikitext(self.LINES)
        self.assertEqual([t for t, _ in docs], ["Alpha", "Beta"])
        self.assertIn("Sub", docs[0][1])  # level-2 headings stay inside the article

    def test_split_before_windows_and_disjoint(self):
        docs = D.build_documents({"calibration": self.LINES[:4], "test": self.LINES[4:]})
        self.assertEqual({d.split for d in docs}, {"calibration", "test"})
        D.tokenize_documents(docs, lambda s: list(range(200)) if "a1" in s else list(range(1000, 1100)), 50256)
        self.assertEqual(D.check_disjoint(docs), [])
        docs[1].windows = [list(docs[0].windows[0])]
        self.assertTrue(D.check_disjoint(docs))

    def test_windows(self):
        w = D.make_windows(list(range(300)), 50256, 128, 64)
        self.assertEqual(len(w), 2)  # 127 + 127, tail of 46 < 64 dropped
        self.assertTrue(all(x[0] == 50256 and len(x) == 128 for x in w))

    def test_positions_exclude_bos(self):
        rng = D.seeded_rng("x", 1)
        self.assertTrue(all(D.sample_position(128, rng) > 0 for _ in range(200)))


class TestDose(unittest.TestCase):
    def test_quantile_linear(self):
        s = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertEqual(Q.quantile(s, 0.5), 2.0)
        self.assertAlmostEqual(Q.quantile(s, 0.9), 3.6)

    def test_target_doses_ignore_nonpositive_and_record_amax(self):
        d = Q.target_doses({3: [0.0, -1.0, 1.0, 2.0, 3.0], 4: [0.0]}, [0.5, 0.99])
        self.assertEqual(d[4]["status"], "NO_POSITIVE_CALIBRATION_ACTIVATIONS")
        self.assertEqual(d[3]["doses"]["q50"], 2.0)
        self.assertAlmostEqual(d[3]["doses"]["2xq99"], 2 * Q.quantile([1.0, 2.0, 3.0], 0.99))
        self.assertEqual(d[3]["a_max_recorded_not_used"], 3.0)

    def test_freeze_detects_change(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            sha = Q.freeze({"a": 1}, p)
            self.assertEqual(Q.load_frozen(p, sha), {"a": 1})
            with open(p, "a") as f:
                f.write(" ")
            with self.assertRaises(RuntimeError):
                Q.load_frozen(p, sha)

    def test_pool_and_sampling(self):
        pool = Q.feature_pool([0, 5, 50, 5000], 10000, 1e-4, 1e-2)
        self.assertEqual(pool, [1, 2])
        self.assertEqual(Q.sample_features(list(range(100)), 5, 0), Q.sample_features(list(range(100)), 5, 0))


class TestDrift(unittest.TestCase):
    def test_hand_example(self):
        a0 = [1.0, 2.0, 0.0, 0.0, 3.0]
        a1 = [1.5, 0.0, 0.4, 0.0, 3.0]
        m0 = [True, True, False, False, True]
        m1 = [True, False, True, False, True]
        r = decompose(a0, a1, m0, m1, j=0, alpha=0.5, scale=0.5)
        self.assertAlmostEqual(r["target_change"], 0.5)
        self.assertAlmostEqual(r["target_err_rel"], 0.0)
        self.assertAlmostEqual(r["drift_deactivated"], 2.0)
        self.assertAlmostEqual(r["drift_newly_active"], 0.4)
        self.assertAlmostEqual(r["drift_orig_active_kept"], 0.0)
        self.assertEqual((r["n_deactivated"], r["n_newly_active"], r["n_orig_active"]), (1, 1, 2))

    def test_matches_toy_protected_split_for_relu(self):
        """With P = active set, the toy's protected/unprotected leakage equals the
        originally-active / newly-active drift (so re-aggregated toy numbers use the same definitions)."""
        rng = random.Random(0)
        D_ = build_decoder("gaussian", 12, 40, rng)
        sae = ToySAE(E=build_encoder("interp", D_, rng, beta=0.5, scale_range=(0.7, 1.4)), b=[-0.25] * 40,
                     D=D_, act="relu")
        checked = 0
        for _ in range(30):
            x = sparse_base_point(D_, 3, (0.5, 1.5), 0.02, rng)
            p = sae.pre(x)
            mask = sae.active_mask(p)
            act = [i for i in range(40) if mask[i]]
            if not act:
                continue
            j = act[0]
            pb = EditProblem(sae, x, j, 1.0, [i for i in act if i != j])
            for meth in ("decoder", "jacobian_ln"):
                delta = run_method(meth, pb, rng)["delta"]
                ev = evaluate(pb, delta)
                p1 = sae.pre(la.add(x, delta))
                r = decompose(sae.activate(p), sae.activate(p1), mask, sae.active_mask(p1), j, 1.0, 1.0)
                self.assertAlmostEqual(r["drift_orig_active"], ev["leak_rel_protected"], places=12)
                self.assertAlmostEqual(r["drift_newly_active"], ev["leak_rel_unprotected"], places=12)
                self.assertAlmostEqual(r["drift_all_nontarget"], ev["leak_rel_all"], places=12)
                checked += 1
        self.assertGreater(checked, 10)


class TestStatsAndLexicon(unittest.TestCase):
    def test_bootstrap_deterministic_and_unit_count(self):
        rows = [{"doc": d, "v": float(d)} for d in range(10) for _ in range(5)]
        units = per_unit_means(rows, "doc", lambda r: r["v"])
        a = cluster_bootstrap(units, n_boot=500, seed=1)
        b = cluster_bootstrap(units, n_boot=500, seed=1)
        self.assertEqual(a, b)
        self.assertEqual(a["n_units"], 10)
        self.assertLessEqual(a["lo"], a["estimate"])
        self.assertLessEqual(a["estimate"], a["hi"])

    def test_paired_by_unit(self):
        rows = []
        for doc in range(4):
            for f in range(3):
                rows.append({"doc": doc, "f": f, "m": "A", "v": 1.0})
                rows.append({"doc": doc, "f": f, "m": "B", "v": 0.5})
        rows.append({"doc": 9, "f": 0, "m": "A", "v": 5.0})  # unmatched: skipped
        d = paired_unit_differences(rows, "doc", lambda r: (r["doc"], r["f"]), "m", "A", "B", lambda r: r["v"])
        self.assertEqual(sorted(d), [0, 1, 2, 3])
        self.assertTrue(all(abs(v - 0.5) < 1e-12 for v in d.values()))

    def test_lexicon(self):
        self.assertEqual(count_hits("The Bride and the groom were married; weddings!", WEDDING_ACTADD), 4)
        self.assertEqual(label("no match here", WEDDING_ACTADD)["success"], 0)
        self.assertEqual(count_hits("wedded bliss", WEDDING_ACTADD), 0)  # exact words only

    def test_bounded_lexicon(self):
        lex = ["space", "spacecraft", "earth", "star", "deep space"]
        self.assertEqual(bounded_hits("Earth's orbit; space-time", lex), 2)       # possessive and hyphen are boundaries
        self.assertEqual(bounded_hits("spacecrafts and stars", lex), 0)           # no inflection, no substring
        self.assertEqual(bounded_hits("aerospace", lex), 0)                        # no substring inside a word
        self.assertEqual(bounded_hits("space and more", lex, before="aero"), 0)   # leading fragment dropped
        self.assertEqual(bounded_hits(" space and more", lex, before="aero"), 1)  # space-prefixed token: whole word
        self.assertEqual(bounded_hits("to the star", lex, after="ship"), 0)       # trailing fragment dropped
        self.assertEqual(bounded_hits("to the star", lex, after=" ship"), 1)
        self.assertEqual(bounded_hits("into deep space now", lex), 1)             # multi-word entry counted once
        lab = bounded_label(" a star", lex, before="the", n_tokens=4)
        self.assertEqual((lab["hits"], lab["success"], lab["hits_per_generated_token"]), (1, 1, 0.25))


if __name__ == "__main__":
    unittest.main()
