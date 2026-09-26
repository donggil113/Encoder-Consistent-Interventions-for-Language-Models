# Manuscript v1: "From Feature Edits to Behavioral Effects: Auditing Sparse Autoencoder Interventions"

| Field | Value |
|---|---|
| TARGET_YEAR | 2027 (ICML) |
| TEMPLATE_YEAR | 2026 |
| SUBMISSION_READY | **false** |
| Compile status | **COMPILE_NOT_RUN**: no LaTeX compiler in the environment. `src/saeedit/tex_check.py` ran static checks and passed (`results/reaggregated/tex_check.json`). |
| Mode | anonymous review (`\usepackage{icml2026}`) |

## Template

No official ICML 2027 author kit was published on 2026-09-26:
`icml.cc/Conferences/2027/{CallForPapers,AuthorInstructions}` and
`media.icml.cc/Conferences/ICML2027/Styles/icml2027.zip` returned 404.
The **official ICML 2026 style files** are therefore used temporarily and are **unmodified**:
they were not renamed to 2027, and margins and fonts are untouched.

- Source: `https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip`
  (228,368 bytes, sha256 `8b29290f5828e176debb57ea9cc00252502973d55ea561a2f18a7f0a326bfc6c`).
  It was fetched once for verification during this work.
- Files copied into this folder, with sha256 values identical to the zip contents:
  - `icml2026.sty` (`7cdcf90f…`)
  - `icml2026.bst` (`0ec3d5eb…`)
  - `algorithm.sty`
  - `algorithmic.sty`
  - `fancyhdr.sty`
- Rules to re-check once an ICML 2027 kit appears. These are from the ICML 2026 CFP and
  author instructions:
  - 8 pages of main text, with unlimited references, impact statement and appendix;
  - an Impact Statement is required (unnumbered, before References);
  - double-blind review;
  - US letter paper.

## Build (when a TeX distribution is available)

```
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Packages beyond the kit: `pgfplots` (with the `groupplots` library), `booktabs`,
`subcaption`, `cleveref`, `mathtools` and `microtype`. After compiling, check:

1. the main-body page count is at most 8;
2. there are no undefined citations or references;
3. the three wide tables and the figure do not overflow;
4. the anonymous header is shown.

None of this has been checked yet.

## Provenance of content

- **Numbers.** `generated_numbers.tex`, `tables/*.tex` and `figures/*.dat` are written by
  `PYTHONPATH=src python3 -m saeedit.paper_assets`, which reads only
  `results/raw/*.csv`. The input hashes are in `results/reaggregated/paper_numbers.json`.
  Do not edit these files by hand.
- **References.** In `references.bib`, each entry's source is noted above it: an official
  proceedings page, arXiv BibTeX, the transformer-circuits site, or Crossref/catalogue
  records. No venue was added without an official source.
- **Claims.** `claims.csv` maps each claim to its evidence files, experiment IDs,
  assumptions and status.
- **Remaining TODOs.** Two `\todo{...}` markers remain, both for results that do not exist
  yet: `P3-REAL-01R` and `P3-REAL-02`. They must not be filled by hand.
