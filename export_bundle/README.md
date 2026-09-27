# P3 export bundle (manuscript v3)

- `p3_v3_bundle.tar.gz` contains, under `p3_v3/`:
  - the manuscript sources (`paper/`: main.tex, references.bib, generated numbers, tables,
    figure data, claims.csv, and the unmodified ICML 2026 style files);
  - configs, code and tests;
  - small raw files and aggregates (`results/`);
  - STATUS, RESEARCH_PACKET, RELATED_WORK and run_manifest.
- `MANIFEST.json` gives the sha256 and size of every bundled file and the sha256 of the
  archive.
- **No PDF.** No LaTeX compiler exists in this environment (COMPILE_NOT_RUN), and the
  manuscript was not sent to any web compiler. To build:

  ```
  tar xzf p3_v3_bundle.tar.gz && cd p3_v3/paper
  latexmk -pdf -interaction=nonstopmode main.tex
  # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
  ```

  After building, check: body ≤ 8 pages, no undefined references or citations, no table
  overflow, anonymous header.
- **Not included:** model/SAE weights, the HF cache, the virtual environment, personal data.
- **Rebuild the bundle:** `PYTHONPATH=src python3 -m saeedit.export_bundle --out export_bundle`
