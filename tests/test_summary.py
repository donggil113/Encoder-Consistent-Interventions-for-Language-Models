"""Checks of the summary helpers (per-instance averaging, flags, pairing)."""

import unittest

from saeedit.run_first_run import aggregate, paired


class TestSummary(unittest.TestCase):
    def test_aggregate_averages_within_unit_and_counts_flags(self):
        rows = [
            {"seed": 0, "method": "m", "status": "OK", "x": 1.0, "flag": True},
            {"seed": 0, "method": "m", "status": "OK", "x": 3.0, "flag": False},
            {"seed": 1, "method": "m", "status": "OK", "x": 10.0, "flag": True},
            {"seed": 1, "method": "m", "status": "FAIL"},
        ]
        rec = aggregate(rows, ["method"], ["x", "flag"])[0]
        self.assertEqual(rec["n_units"], 2)
        self.assertEqual(rec["status_counts"], {"OK": 3, "FAIL": 1})
        self.assertEqual(rec["x"]["min"], 2.0)   # seed 0 mean, not the raw 1.0
        self.assertEqual(rec["x"]["max"], 10.0)
        self.assertEqual(rec["flag"]["min"], 0.5)
        self.assertEqual(rec["flag"]["max"], 1.0)

    def test_paired_counts_units_not_rows(self):
        rows = []
        for seed, (a, b) in enumerate([(1.0, 2.0), (3.0, 2.0), (0.5, 0.6)]):
            for _ in range(4):  # repeated rows inside one unit must not inflate n
                rows.append({"seed": seed, "g": 0, "method": "A", "status": "OK", "v": a})
                rows.append({"seed": seed, "g": 0, "method": "B", "status": "OK", "v": b})
        rec = paired(rows, ["g"], "A", "B", lambda r: r["v"])[0]
        self.assertEqual(rec["n_units"], 3)
        self.assertEqual(rec["a_better_units"], 2)


if __name__ == "__main__":
    unittest.main()
