"""Build the P3 export bundle (standard library only).

    PYTHONPATH=src python3 -m saeedit.export_bundle --out export_bundle

Writes export_bundle/p3_v3_bundle.tar.gz and export_bundle/MANIFEST.json (sha256 and size of
every bundled file). The bundle holds the manuscript sources, configs, code, tests, small raw
files and aggregates, and the run manifest. It never holds model/SAE weights, the HF cache,
the virtual environment or any personal data. A PDF is included only if paper/main.pdf exists.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import time

INCLUDE = [
    "paper/main.tex", "paper/references.bib", "paper/generated_numbers.tex", "paper/claims.csv", "paper/README.md",
    "paper/icml2026.sty", "paper/icml2026.bst", "paper/algorithm.sty", "paper/algorithmic.sty", "paper/fancyhdr.sty",
    "paper/tables", "paper/figures", "paper/main.pdf",
    "configs", "src", "tests",
    "results/raw", "results/summary.json", "results/reaggregated",
    "results/contract_exec", "results/contract_exec_v2", "results/real01r", "results/real01r_reagg",
    "results/readout_closure", "results/real02", "results/real02_lx",
    "run_manifest.json", "STATUS.md", "RESEARCH_PACKET.md", "RELATED_WORK.md",
]
EXCLUDE_PARTS = ("__pycache__", ".pyc", ".git")
MAX_FILE_BYTES = 20 * 1024 * 1024


def files(root: str):
    for p in INCLUDE:
        full = os.path.join(root, p)
        if os.path.isfile(full):
            yield p
        elif os.path.isdir(full):
            for d, _, fs in os.walk(full):
                for f in sorted(fs):
                    rel = os.path.relpath(os.path.join(d, f), root)
                    if not any(x in rel for x in EXCLUDE_PARTS):
                        yield rel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="export_bundle")
    args = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    entries, skipped = [], []
    tar_path = os.path.join(args.out, "p3_v3_bundle.tar.gz")
    with tarfile.open(tar_path, "w:gz", format=tarfile.PAX_FORMAT) as tar:
        for rel in sorted(set(files(args.root))):
            full = os.path.join(args.root, rel)
            size = os.path.getsize(full)
            if size > MAX_FILE_BYTES:
                skipped.append({"path": rel, "bytes": size, "reason": "larger than 20 MB"})
                continue
            data = open(full, "rb").read()
            info = tarfile.TarInfo("p3_v3/" + rel)
            info.size, info.mtime, info.mode = size, 0, 0o644  # fixed mtime: reproducible member metadata
            tar.addfile(info, io.BytesIO(data))
            entries.append({"path": rel, "bytes": size, "sha256": hashlib.sha256(data).hexdigest()})
    has_pdf = any(e["path"] == "paper/main.pdf" for e in entries)
    manifest = {
        "bundle": tar_path, "bundle_sha256": hashlib.sha256(open(tar_path, "rb").read()).hexdigest(),
        "bundle_bytes": os.path.getsize(tar_path), "n_files": len(entries), "files": entries, "skipped": skipped,
        "pdf": "included" if has_pdf else "NOT_BUILT (no LaTeX compiler in the environment; see build_commands)",
        "build_commands": ["cd p3_v3/paper", "latexmk -pdf -interaction=nonstopmode main.tex",
                           "# or: pdflatex main && bibtex main && pdflatex main && pdflatex main"],
        "not_included": ["model and SAE weights", "HF cache", "virtual environment", "WikiText text beyond prompt token ids and generated continuations in results"],
        "seconds": round(time.time() - t0, 2)}
    with open(os.path.join(args.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: manifest[k] for k in ("bundle", "bundle_bytes", "n_files", "pdf", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
