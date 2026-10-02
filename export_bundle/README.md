# P3 packages

## v4.1 (current)

- `p3_v4_1_anonymous.tar.gz` is the **submission package**. It contains:
  - the manuscript sources (`paper/`, with the unmodified ICML 2026 style files);
  - an anonymised supplement: code, configs, tests, small raw files and aggregates, the claim
    map, and the PPLM space word list with its Apache-2.0 license.
  - Internal notes (STATUS, RESEARCH_PACKET, RELATED_WORK, run manifest, paper/README) and git
    metadata are excluded.
  - Every file was scanned against `internal/anonymity_patterns.txt` (not shipped) for
    usernames, local paths, remotes and session metadata; the build fails if any is found.
  - This repository's commit identifiers are replaced by `<commit>` in shipped text files
    (listed in the manifest under `commit_ids_redacted`); third-party pins (revisions, sha256,
    the PPLM commit) are kept.
  - A `SUPPLEMENT_README.md` describes the contents and the exclusions.
  - Tar members carry no owner names and a fixed mtime.
- `p3_v4_1_internal.tar.gz` is the **evidence package**: the same content plus the internal notes.
- `*_MANIFEST.json` gives:
  - sha256 and size per file;
  - the archive sha256;
  - a check that every item the manuscript says the supplement contains is present.
- **PDF included** (`paper/main.pdf`, built 2026-10-02 with a minimal local TeX Live; nothing
  was sent to a web compiler). To rebuild from the sources:

  ```
  tar xzf p3_v4_1_anonymous.tar.gz && cd p3_v4_1_anonymous/paper
  latexmk -pdf -interaction=nonstopmode main.tex
  # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
  ```

  Then check the following:
  - the page that holds the Conclusion's last sentence is page 8 or earlier;
  - there are no undefined references;
  - tables are legible at 100%;
  - the anonymous header is present.
- **Not included:** model/SAE weights and the WikiText files. They are pinned by revision in
  `configs/p3_contract_gpt2_res_jb_l8.json` and by sha256 in `configs/p3_asset_hashes.json`.
  The HF cache and the virtual environment are also not included.
- **Rebuild:**

  ```
  PYTHONPATH=src python3 -m saeedit.export_bundle --mode anonymous --out export_bundle
  PYTHONPATH=src python3 -m saeedit.export_bundle --mode internal --out export_bundle
  ```

## Previous deliverables (kept)

`p3_v4_anonymous.tar.gz` / `p3_v4_internal.tar.gz` (v4, no PDF) and `p3_v3_bundle.tar.gz`
with `MANIFEST.json` (v3).
