"""Unit checks of the R8 delivery runner helpers (standard library only; no PDF, no TeX)."""

import json
import os
import tempfile
import unittest

from saeedit.deliver import build_md, frozen_sha, read_list, scan_text


class TestDeliverHelpers(unittest.TestCase):
    def test_scan_text_reports_patterns_and_commit_ids_without_redacting(self):
        data = b"line one\nsee github.com/someone and commit 0123abc here\n"
        hits = scan_text(data, ["github.com/someone"], ["0123abc"])
        self.assertEqual([(h["kind"], h["line"]) for h in hits], [("identifying", 2), ("commit id", 2)])
        self.assertEqual(scan_text(data, [], []), [])
        self.assertEqual(scan_text(b"0123abcd", [], ["0123abc"]), [])  # word boundary: a longer hex word is not a hit

    def test_read_list_skips_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "l.txt")
            with open(p, "w") as f:
                f.write("# comment\n\nabc def\n ghi \n")
            self.assertEqual(read_list(p), ["abc", "def", "ghi"])
            self.assertEqual(read_list(os.path.join(d, "missing.txt")), [])

    def test_frozen_sha_ignores_evidence_block_only(self):
        a = {"run_id": "x", "inputs": {"p": 1}, "evidence": {}}
        b = {"run_id": "x", "inputs": {"p": 1}, "evidence": {"stage": "done"}}
        c = {"run_id": "x", "inputs": {"p": 2}, "evidence": {}}
        self.assertEqual(frozen_sha(a), frozen_sha(b))
        self.assertNotEqual(frozen_sha(a), frozen_sha(c))

    def test_build_md_states_facts_from_the_contract(self):
        cfg = {"project": "P3", "manuscript_version": "v4.2",
               "source": {"head": "0123456789abcdef", "branch": "b"},
               "deliverables": {"pdf_name": "P3_v4.2_0123456.pdf"},
               "style": {"description": "STYLE"},
               "build": {"recorded": {"description": "REC", "commands": ["pdflatex main"]},
                         "check": {"shell": ["pdflatex -halt-on-error -no-shell-escape main"]},
                         "required_packages": ["booktabs"]},
               "source_zip": {"members": ["paper/main.tex"], "not_included": "NOTINC"},
               "research_status": {"study": "STUDY_SCOPE_FROZEN", "method_utility": "M", "SUBMISSION_READY": False,
                                   "TARGET_YEAR": 2027, "TEMPLATE_YEAR": 2026, "icml_2027_instructions": "not obtained",
                                   "external_read_review": "UNASSIGNED"}}
        md = build_md(cfg, {"sha256": "ab" * 32, "bytes": 416464, "pages": 20, "body_end_page": 7, "appendix_start_page": 11})
        for s in ("P3_v4.2_0123456.pdf", "416,464 bytes", "20 pages", "ends on page 7", "STYLE", "REC",
                  "-no-shell-escape", "NOTINC", "booktabs", "SUBMISSION_READY = false", "UNASSIGNED", "main.tex"):
            self.assertIn(s, md)
        self.assertNotIn("certif", md.split("## Status")[0])  # no certification language before the status block
        json.dumps(cfg)  # the contract stays JSON-serialisable


if __name__ == "__main__":
    unittest.main()
