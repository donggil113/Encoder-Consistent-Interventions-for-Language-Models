"""Build the P3 packages (standard library only).

    PYTHONPATH=src python3 -m saeedit.export_bundle --mode internal  --out export_bundle
    PYTHONPATH=src python3 -m saeedit.export_bundle --mode anonymous --out export_bundle

internal  : evidence package (manuscript sources and PDF, configs, code, tests, small raw files
            and aggregates, STATUS / RESEARCH_PACKET / run manifest).
anonymous : submission package (manuscript sources and PDF plus an anonymised supplement: code,
            configs, tests, small raw files, the claim map and third-party notices). Internal
            notes, run manifests and git metadata are excluded; repository commit identifiers
            are replaced by "<commit>" in shipped text files; every shipped text file is scanned
            for the identifying strings listed in internal/anonymity_patterns.txt and the build
            fails without writing a package if any is found. The scan is a string check, not a
            proof of anonymity.

Neither package holds model/SAE weights, the HF cache or the virtual environment: those are
pinned by revision in configs/p3_contract_gpt2_res_jb_l8.json and by sha256 in
configs/p3_asset_hashes.json. A PDF is included only if paper/main.pdf exists.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time

PAPER = ["paper/main.tex", "paper/references.bib", "paper/generated_numbers.tex", "paper/icml2026.sty",
         "paper/icml2026.bst", "paper/algorithm.sty", "paper/algorithmic.sty", "paper/fancyhdr.sty",
         "paper/tables", "paper/figures", "paper/main.pdf"]
SUPPLEMENT = ["paper/claims.csv", "configs", "src", "tests", "data_external",
              "results/raw", "results/summary.json", "results/reaggregated",
              "results/contract_exec", "results/contract_exec_v2", "results/real01r", "results/real01r_reagg",
              "results/readout_closure", "results/real02", "results/real02_lx", "results/real02_space"]
INTERNAL_ONLY = ["paper/README.md", "run_manifest.json", "STATUS.md", "RESEARCH_PACKET.md", "RELATED_WORK.md"]
EXCLUDE_PARTS = ("__pycache__", ".pyc", ".git/", "internal/")
MAX_FILE_BYTES = 20 * 1024 * 1024
TEXT_EXT = (".json", ".csv", ".md", ".tex", ".py", ".txt", ".bib", ".dat", ".sty", ".bst")
# what the manuscript says the supplement contains -> paths that must be present
CLAIMED = {"code": "src/saeedit", "configs": "configs", "tests": "tests", "raw toy files": "results/raw",
           "real-model raw rows (fidelity pilot)": "results/real01r/raw_run.csv",
           "real-model raw rows (space pilot)": "results/real02_space/raw_run.csv",
           "readout closure report": "results/readout_closure/readout_closure_report.json",
           "claim map": "paper/claims.csv", "space word list and its license": "data_external/pplm_space/LICENSE",
           "asset hashes": "configs/p3_asset_hashes.json"}
SUPPLEMENT_README = """# Anonymous supplement

Contents: manuscript sources and the built PDF (paper/), code (src/), configurations
(configs/), tests (tests/), small raw result files and aggregates (results/), the claim map
(paper/claims.csv) and the third-party space word list with its Apache-2.0 license
(data_external/pplm_space/).

Not included: model/SAE weights and the WikiText files (pinned by revision in
configs/p3_contract_gpt2_res_jb_l8.json and by sha256 in configs/p3_asset_hashes.json), the
HF cache, the virtual environment, git metadata, and the authors' internal notes. Some
configuration and provenance files mention those internal notes (STATUS.md,
RESEARCH_PACKET.md, a run manifest) or approval steps; they are process records and the
referenced files are not part of this supplement. Repository commit identifiers were
replaced by "<commit>" in the shipped text files; the frozen configurations are otherwise
unchanged.

The anonymity scan removes known author strings; it is a string check, not a proof of
anonymity.
"""


def _files(root, items):
    for p in items:
        full = os.path.join(root, p)
        if os.path.isfile(full):
            yield p
        elif os.path.isdir(full):
            for d, _, fs in os.walk(full):
                for f in sorted(fs):
                    rel = os.path.relpath(os.path.join(d, f), root)
                    if not any(x in rel + "/" for x in EXCLUDE_PARTS):
                        yield rel


def _own_commits(root):
    """Repository commit identifiers (short and full), redacted from shipped text in anonymous mode."""
    p = os.path.join(root, "internal", "own_commits.txt")
    out = []
    if os.path.exists(p):
        for l in open(p):
            out += l.split()
    return sorted({x for x in out if x}, key=len, reverse=True)


def _patterns(root):
    pats = []  # every pattern lives in the internal file, so the scanner itself ships clean
    p = os.path.join(root, "internal", "anonymity_patterns.txt")
    if not os.path.exists(p):
        raise SystemExit("internal/anonymity_patterns.txt is required for the anonymous build")
    pats += [l.strip() for l in open(p) if l.strip() and not l.startswith("#")]
    return pats


def _decode(data):
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="export_bundle")
    ap.add_argument("--mode", choices=["internal", "anonymous"], default="internal")
    ap.add_argument("--tag", default="p3_v4_1")
    args = ap.parse_args(argv)
    t0 = time.time()
    items = PAPER + SUPPLEMENT + (INTERNAL_ONLY if args.mode == "internal" else [])
    rels = sorted(set(_files(args.root, items)))
    hits, redacted, contents = [], {}, {}
    if args.mode == "anonymous":
        rx = re.compile("|".join(re.escape(p) for p in _patterns(args.root)), re.IGNORECASE)
        commits = _own_commits(args.root)
        crx = re.compile(r"\b(" + "|".join(re.escape(c) for c in commits) + r")\b") if commits else None
        for rel in rels:
            data = open(os.path.join(args.root, rel), "rb").read()
            if rel.endswith(TEXT_EXT):
                txt = _decode(data)
                if crx is not None:
                    txt, n = crx.subn("<commit>", txt)
                    if n:
                        redacted[rel] = n
                        data = txt.encode("utf-8")
                for m in rx.finditer(txt):
                    hits.append({"path": rel, "line": txt.count("\n", 0, m.start()) + 1, "match": m.group()})
            contents[rel] = data
        if hits:
            print(json.dumps({"error": "identifying strings found; no package written", "hits": hits[:50]}, indent=1))
            return 1
    os.makedirs(args.out, exist_ok=True)
    name = f"{args.tag}_{args.mode}"
    tar_path = os.path.join(args.out, name + ".tar.gz")
    entries, skipped = [], []
    with tarfile.open(tar_path, "w:gz", format=tarfile.PAX_FORMAT) as tar:
        if args.mode == "anonymous":
            data = SUPPLEMENT_README.encode("utf-8")
            info = tarfile.TarInfo(f"{name}/SUPPLEMENT_README.md")
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
            entries.append({"path": "SUPPLEMENT_README.md", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        for rel in rels:
            full = os.path.join(args.root, rel)
            if os.path.getsize(full) > MAX_FILE_BYTES:
                skipped.append({"path": rel, "bytes": os.path.getsize(full), "reason": "larger than 20 MB"})
                continue
            data = contents[rel] if rel in contents else open(full, "rb").read()
            info = tarfile.TarInfo(f"{name}/{rel}")
            info.size, info.mtime, info.mode = len(data), 0, 0o644  # no owner names, fixed mtime
            tar.addfile(info, io.BytesIO(data))
            entries.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    paths = [e["path"] for e in entries]
    claimed = {k: any(p == v or p.startswith(v.rstrip("/") + "/") for p in paths) for k, v in CLAIMED.items()}
    # every file-like evidence pointer in the claim map must resolve inside the package
    missing_evidence = []
    cm = os.path.join(args.root, "paper", "claims.csv")
    if os.path.exists(cm):
        import csv
        for row in csv.DictReader(open(cm)):
            for tok in re.split(r"[;,]\s*", row.get("evidence_files", "")):
                tok = tok.strip().split("::")[0].split(" ")[0]
                if not tok or "/" not in tok or tok.startswith("<") or "*" in tok:
                    continue
                if not any(p == tok or p.startswith(tok.rstrip("/") + "/") for p in paths):
                    missing_evidence.append({"claim": row["claim_id"], "path": tok})
    manifest = {
        "package": tar_path, "mode": args.mode, "sha256": hashlib.sha256(open(tar_path, "rb").read()).hexdigest(),
        "bytes": os.path.getsize(tar_path), "n_files": len(entries), "files": entries, "skipped": skipped,
        "pdf": "included" if "paper/main.pdf" in paths else "NOT_BUILT (no PDF present)",
        "claimed_contents_present": claimed,
        "claim_map_evidence_paths_not_in_package": missing_evidence,
        "anonymity_scan": ("passed: no listed identifying string found (a string check, not a proof of anonymity)"
                           if args.mode == "anonymous" else "not applicable"),
        "commit_ids_redacted": redacted if args.mode == "anonymous" else "not applicable",
        "third_party_notices_kept": ["paper/icml2026.sty and related style files (unmodified)",
                                     "data_external/pplm_space/LICENSE (Apache-2.0, unmodified)"],
        "not_included": ["model and SAE weights and WikiText files (pinned by revision in configs/p3_contract_gpt2_res_jb_l8.json; sha256 in configs/p3_asset_hashes.json)",
                         "HF cache", "virtual environment", "git metadata"]
                        + (["internal notes and run manifests"] if args.mode == "anonymous" else []),
        "build_commands": [f"tar xzf {name}.tar.gz && cd {name}/paper", "latexmk -pdf -interaction=nonstopmode main.tex",
                           "# or: pdflatex main && bibtex main && pdflatex main && pdflatex main"],
        "seconds": round(time.time() - t0, 2)}
    with open(os.path.join(args.out, name + "_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: manifest[k] for k in ("package", "bytes", "n_files", "pdf", "claimed_contents_present",
                                                "claim_map_evidence_paths_not_in_package", "anonymity_scan",
                                                "commit_ids_redacted", "seconds")}))
    return 0 if all(claimed.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
