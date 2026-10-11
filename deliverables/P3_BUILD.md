# P3 manuscript source (v4.2, source commit 0f5b60e)

Delivered PDF: `P3_v4.2_0f5b60e.pdf` (byte-identical to `paper/main.pdf` at commit `0f5b60e539f56e74942e1170e42c95ba9d092e9d` on branch `claude/magical-bell-8vyrpd`): sha256 `24ced5541de774356502b3b93e8d7c6a3dcac5d5388b288cb1b2dfc6966352e8`, 416,464 bytes, 20 pages (US letter), body (last sentence of the Conclusion) ends on page 7; the one-column appendix starts on page 11.

## Style

Official ICML 2026 style files (icml2026.sty, header 'version of 2025-10-29'; icml2026.bst; algorithm.sty; algorithmic.sty; fancyhdr.sty), unmodified and not renamed, loaded as \usepackage{icml2026} (anonymous review mode). They serve as the temporary template while the ICML 2027 instructions are not obtained (TARGET_YEAR = 2027, TEMPLATE_YEAR = 2026). Fonts, margins and spacing are the style defaults.

## Build that produced the delivered PDF (recorded)

The delivered PDF was built once on 2026-10-05 from these sources with the session TeX Live 2026 (scheme-basic plus the packages listed below; pdfTeX 3.141592653-2.6-1.40.29, BibTeX 0.99e) by four passes in paper/. Flags as recorded: -interaction=nonstopmode; the log shows 'restricted \write18 enabled' (the pdfTeX default; -no-shell-escape was not passed) and -halt-on-error was not passed. Nothing was sent to a web compiler.

```
cd paper
pdflatex -interaction=nonstopmode main
bibtex main
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode main
```

## Reproduction command (used for the R8 self-containment check of this zip)

```
cd paper
pdflatex -halt-on-error -no-shell-escape -interaction=nonstopmode main
bibtex main
pdflatex -halt-on-error -no-shell-escape -interaction=nonstopmode main
pdflatex -halt-on-error -no-shell-escape -interaction=nonstopmode main
```

A byte-identical PDF is not expected from a rebuild (pdfTeX embeds the creation date and a document id); the comparison is the page count and the extracted text of every page.

## Contents

`paper/`: algorithm.sty, algorithmic.sty, fancyhdr.sty, figures/alpha_curves.dat, figures/extproxy.dat, figures/norm_budget.dat, figures/pilot_dose.dat, generated_numbers.tex, icml2026.bst, icml2026.sty, main.tex, references.bib, tables/tab_activeset_all.tex, tables/tab_activeset_relu.tex, tables/tab_contract_exec.tex, tables/tab_coord.tex, tables/tab_extproxy.tex, tables/tab_feasibility.tex, tables/tab_linear.tex, tables/tab_matched_denominators.tex, tables/tab_norm_budget.tex, tables/tab_pilot_equalnorm.tex, tables/tab_pilot_matched.tex, tables/tab_pilot_matched_median.tex, tables/tab_pilot_matched_quality.tex, tables/tab_pilot_paired.tex, tables/tab_space.tex, tables/tab_space_paired.tex, tables/tab_sparsity_exploratory.tex.

Not included: paper/README.md and paper/claims.csv (internal records; in the repository and the internal package), raw results and code (export_bundle packages / repository), the PDF (delivered separately as the .pdf file).

## Required LaTeX packages

microtype, graphicx, subcaption, booktabs, array, hyperref, amsmath, amssymb, mathtools, amsthm, cleveref, pgfplots, icml2026 (shipped), fancyhdr/algorithm/algorithmic (shipped), times/courier/helvetic fonts (Type 1).

## Status

- Study: STUDY_SCOPE_FROZEN; METHOD_UTILITY_NOT_SUPPORTED_IN_THIS_PILOT.
- SUBMISSION_READY = false; TARGET_YEAR = 2027, TEMPLATE_YEAR = 2026 (not obtained: the official 2027 URLs returned 404 on 2026-09-26/27; the 2026 anonymous review style is the temporary template).
- External read review: UNASSIGNED (three questions in paper/README.md, left as written; nobody contacted; no acceptance or verification claimed).
- Compile status, visual check, scientific validity, novelty and submission readiness are separate records (see P3_delivery.json); this file certifies none of them beyond the build facts above.
