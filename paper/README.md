# Manuscript v3: "Auditing Sparse Autoencoder Feature Edits in GPT-2 Small: Coordinates, Fidelity, and a Lexical Pilot"

The title was narrowed in v3; the v2 title ("From Feature Edits to Behavioral Effects: …") promised
behavioral results that exist only as a lexical pilot.

Order in v3: (a) raw vs canonical coordinates → (b) admissible feature realization (including
the toy redistribution results) → (c) matched-denominator robustness → (d) lexical change and
quality (P3-REAL-02-LX). v3 was produced by editing v2 in place, not by regenerating it.

| Field | Value |
|---|---|
| TARGET_YEAR | 2027 (ICML) |
| TEMPLATE_YEAR | 2026 |
| SUBMISSION_READY | **false** |
| Compile status | **COMPILE_NOT_RUN**: no LaTeX compiler (`pdflatex`, `latexmk`, `tectonic`, `xelatex`, `lualatex`) is installed, and installing one was not approved. No PDF exists and no page count was measured. `src/saeedit/tex_check.py` ran static checks (`results/reaggregated/tex_check.json`). The manuscript was not uploaded to any web compiler. |
| Mode | anonymous review (`\usepackage{icml2026}`) |

## Template

We did not find an official ICML 2027 author kit on 2026-09-26: the three URLs
`icml.cc/Conferences/2027/{CallForPapers,AuthorInstructions}` and
`media.icml.cc/Conferences/ICML2027/Styles/icml2027.zip` returned 404. That is evidence about
those URLs on that date only, not proof that no 2027 information exists elsewhere; re-check
before submission.
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
latexmk -pdf -interaction=nonstopmode main.tex
# equivalent without latexmk:
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The same sources are in `export_bundle/p3_v3_bundle.tar.gz` under `p3_v3/paper/`.

Packages beyond the kit: `pgfplots` (with the `groupplots` library), `booktabs`,
`subcaption`, `cleveref`, `mathtools` and `microtype`. After compiling, check:

1. the main-body page count is at most 8;
2. there are no undefined citations or references;
3. the three wide tables and the figure do not overflow;
4. the anonymous header is shown.

None of this has been checked yet. In particular the 8-page body limit is unverified: the
static word count of the body (`main_body_words_approx` in `results/reaggregated/tex_check.json`)
is only a proxy.

## Provenance of content

- **Numbers.** `generated_numbers.tex`, `tables/*.tex` and `figures/*.dat` are written by
  `PYTHONPATH=src python3 -m saeedit.paper_assets`, which reads only `results/raw/*.csv`
  (toy) and the real-model result files in `results/contract_exec*/` and
  `results/real01r/`. The input hashes are in `results/reaggregated/paper_numbers.json`.
  Do not edit these files by hand.
- **References.** In `references.bib`, each entry's source is noted above it: an official
  proceedings page, arXiv BibTeX, the transformer-circuits site, or Crossref/catalogue
  records. No venue was added without an official source.
- **Claims.** `claims.csv` maps each claim to its evidence files, experiment IDs,
  assumptions and status.
- **Remaining TODOs.** `\todo{...}` markers remain only for results that do not exist:
  `P3-REAL-01R-FULL` (all 64 features) and `P3-REAL-02-SEM` (semantic evaluation). They must
  not be filled by hand.
- **Real-model numbers.** They come from `results/contract_exec*/`, `results/real01r/`,
  `results/readout_closure/`, `results/real01r_reagg/` and `results/real02_lx/`, through the
  same script.
