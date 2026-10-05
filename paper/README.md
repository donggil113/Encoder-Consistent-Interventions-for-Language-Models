# Manuscript v4.2: "Auditing Sparse Autoencoder Feature Edits in GPT-2 Small: Coordinates, Fidelity, and a Lexical Pilot"

v4.2 (2026-10-05) changes only wording: the two `\todo` markers in the body were replaced by a
"Not evaluated" limitation, and the primary estimand is stated in Section 6.2. No number,
table or result changed. v4 condenses v3 (edited in place, not regenerated). It keeps three core claims (see
`claims.csv`, column `core`):
1. the readout convention, a measurement caveat resting on a standard LayerNorm property;
2. conditional same-layer fidelity with its full denominator;
3. one lexical pilot, P3-REAL-02-SPACE.

The body has three core tables: coordinates, target-matched fidelity with denominators, and
the lexical result. Operational records moved to the appendix, among them the closed
wedding branch and the denominator status table.

| Field | Value |
|---|---|
| TARGET_YEAR | 2027 (ICML) |
| TEMPLATE_YEAR | 2026 |
| SUBMISSION_READY | **false** |
| Compile status | **BUILT (v4.2, 2026-10-05; wording-only change from the v4.1 build of 2026-10-02)** with a minimal TeX Live (scheme-basic + the packages the manuscript needs, installed into a session scratch folder under the Round-6 limited build approval; pdfTeX 1.40.29, TL 2026 tlmgr r79639) and `pdflatex`/`bibtex` (4 passes). 20 pages: body (through the last sentence of the Conclusion) ends on **page 7**; Impact Statement and References start on page 7; the one-column appendix starts on page 11. No undefined references or citations; 0 overfull boxes and 0 font-shape warnings after the layout fixes listed in STATUS.md (URL line breaks, figure width, table column spacing and sizes, a split appendix table, number formatting). Pages were rendered locally at 110 dpi and inspected. PDF metadata: title set, author "Anonymous Authors" (from the style), no personal metadata. Nothing was sent to a web compiler. |
| Mode | anonymous review (`\usepackage{icml2026}`) |

## Template

We did not find an official ICML 2027 author kit on 2026-09-26: the three URLs
`icml.cc/Conferences/2027/{CallForPapers,AuthorInstructions}` and
`media.icml.cc/Conferences/ICML2027/Styles/icml2027.zip` returned 404. That is evidence about
those URLs on that date only, not proof that no 2027 information exists elsewhere. A status-only
re-check on 2026-09-27 09:15 UTC again returned 404 for all three. Re-check before submission.
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

## Build (reproduces the shipped PDF)

```
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
# equivalent without latexmk:
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The same sources and the built PDF are in both v4.1 packages:
- `export_bundle/p3_v4_1_anonymous.tar.gz`: the submission package. It is scanned for listed
  identifying strings (a string check, not a proof of anonymity), redacts this repository's
  commit identifiers, and excludes internal notes.
- `export_bundle/p3_v4_1_internal.tar.gz`: the evidence package.

Packages beyond the kit: `pgfplots` (with the `groupplots` library), `booktabs`,
`subcaption`, `cleveref`, `mathtools` and `microtype`. After compiling, check:

1. the main-body page count is at most 8;
2. there are no undefined citations or references;
3. the three wide tables and the figure do not overflow;
4. the anonymous header is shown.

Checked on the 2026-10-02 build: body ends on page 7 of 20 (limit 8); no undefined references;
no overfull boxes; tables legible at 100%; anonymous header and line numbers present.
`main_body_words_approx` in `results/reaggregated/tex_check.json` is only a proxy and is not
the page check.

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
- **Follow-ups outside the frozen scope.** `P3-REAL-01R-FULL` (all 64 calibration
  features) and `P3-REAL-02-SEM` (semantic evaluation) are recorded as internal TODOs
  (STATUS.md, RESEARCH_PACKET.md, claims.csv notes). In the manuscript they appear only as the
  "Not evaluated" limitation; they are not required by the three core claims, and they must
  not be filled in by hand.
- **Real-model numbers.** They come from `results/contract_exec*/`, `results/real01r/`,
  `results/readout_closure/`, `results/real01r_reagg/` and `results/real02_lx/`, through the
  same script.

## External read review (UNASSIGNED)

The results are frozen (STUDY_SCOPE_FROZEN; METHOD_UTILITY_NOT_SUPPORTED_IN_THIS_PILOT).
An external read of the manuscript by a non-author is recommended quality control; it is not
a separate certification that ICML requires of every author, and it does not by itself make
the manuscript submission-ready (SUBMISSION_READY stays false). No reviewer has been
contacted; status **UNASSIGNED**. Target: `paper/main.pdf` (v4.2; sha256 in STATUS.md).

Three questions for the reader:

1. Is the coordinate-handling caveat (canonical readout $E(P(h+\delta))$ versus the raw
   $E(x+\delta)$) worth generalising beyond the prior implementations and literature it rests
   on, or is it adequately stated as a measurement caveat?
2. Does the conditional same-layer fidelity measurement (8 features, 54 documents, full
   common denominator) provide a meaningful ML insight?
3. Does the paper make any behavioral claim beyond the sparse lexical null (0 of 32
   discordant articles at one KL cap)?
