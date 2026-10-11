# STATUS — P3: Do SAE Edits Change What They Claim?

Last updated: 2026-10-11 (eighth session, R8 delivery). Branch: `claude/magical-bell-8vyrpd`.

## Eighth session summary (R8 delivery, 2026-10-11; read first)

Checked at start: HEAD `0f5b60e` equal to origin; clean tree; `paper/main.pdf` header `%PDF-1.7`,
416,464 bytes, sha256 `24ced5541de77435…` equal to the v4.2 record (seventh session below,
`run_manifest.json` seventh_session, README compile row); git blob `f948c007…` equal to the blob
GitHub serves for commit 0f5b60e (contents API, size 416464); the session TeX install, the venv
(pymupdf) and the v4.2 page renders still present; `/nvmedata` absent (not created); no CLAUDE.md.
Verdicts kept: **STUDY_SCOPE_FROZEN / METHOD_UTILITY_NOT_SUPPORTED_IN_THIS_PILOT**.
SUBMISSION_READY stays false; the external reader stays **UNASSIGNED** and the three README
questions are untouched.

**Delivery (the PDF is unchanged and was not rebuilt).** The contract was frozen before
execution in `configs/auto_run_r8.json` (run id `P3-R8-DELIVERY-2026-10-11`: delivery stages
only, no research stage, input hashes, limits, stop conditions; the runner appends only the
`evidence` block). Runner `src/saeedit/deliver.py` (standard library; pymupdf for page facts),
helpers covered by `tests/test_deliver.py` (4 tests, pass).
- `deliverables/P3_v4.2_0f5b60e.pdf`: byte copy of `paper/main.pdf` (`cmp` identical; sha256
  `24ced5541de77435…`; 20 pages; body ends on page 7; appendix starts on page 11; author
  metadata "Anonymous Authors"). Attached in the session reply.
- `deliverables/P3_source.zip` (225,353 bytes; sha256 `20e499d3b2b469d5…`): the 29 tracked
  manuscript sources (main.tex, references.bib, generated_numbers.tex, 16 tables, 4 `.dat`
  figure files, the unmodified ICML 2026 style files) plus a generated `BUILD.md` (also at
  `deliverables/P3_BUILD.md`); deterministic (sorted members, fixed timestamps). Not included:
  README.md, claims.csv, raw results, code, the PDF. Identifying-string scan (15 patterns,
  38 commit ids): 0 hits; the zip is a user deliverable and is not redacted.
- `deliverables/P3_delivery.json`: source commit, hashes, pages, style version, the recorded
  build command, the check command, pages viewed, study status, file list with hashes.
- Self-containment check (a check in the scratch folder, not a rebuild of `paper/main.pdf`):
  the zip unpacked and built with `pdflatex -halt-on-error -no-shell-escape
  -interaction=nonstopmode` ×3 + `bibtex`; all four exit 0; 20 pages, body end page 7,
  appendix page 11, 0 overfull boxes, 0 undefined references, 0 font warnings; extracted text
  identical to the delivered PDF on all 20 pages. The log has no `\write18` line and epstopdf
  reports "Shell escape feature is not enabled" (shell escape off). The delivered PDF itself
  was built on 2026-10-05 without those two flags (`restricted \write18 enabled`, the pdfTeX
  default); BUILD.md and the record say so.
- One engineering bug during the run: the runner read `cfg["build_check"]` while the contract
  stores the commands under `cfg["build"]`; the first invocation did verify/copy/zip and
  crashed before the check. One-line fix, recorded in the contract's `amendments` (runner
  hashes before/after, frozen hash before), one rerun (copy and zip skipped by hash, the
  check ran). No stage, limit, member list or input changed.
- Pages viewed in R8: none re-rendered (unchanged PDF; the receipts of 2026-10-02 (v4.1, all
  20 pages) and 2026-10-05 (v4.2, pages 5–7) are reused and listed in the record). The runner
  re-derived the page count and the body-end page from the file.
- Existing packages untouched (`export_bundle/p3_v4_1_*.tar.gz` keep the v4_1 file name and
  hold v4.2 content). No reviewer agent or workflow. Nothing uploaded; no PR; no submission.

**Correction to the seventh-session record:** the README hash listed there (`95557a9d…`) was
taken before that file's final edit; the committed `paper/README.md` at 0f5b60e has sha256
`9780bc7dc93f698c…`. The manuscript and PDF hashes there are correct.

**Not done (by instruction):** no topic, dose, KL, feature, model, semantic judge, generation,
activation check or readout-closure rerun; no rebuild of `paper/main.pdf`; no install, download,
GPU or paid API.

### Eighth session: cost ledger

| step | wall | process CPU (incl. children) | notes |
|---|---|---|---|
| state checks, 2 GitHub API reads | interactive | – | no agents |
| runner, first invocation (verify, copy, zip; crashed before the check) | 0.5 s | 0.35 s | |
| runner, rerun incl. the self-containment build | 6.2 s | 0.3 s self + 4.5 s children | pdflatex ×3 + bibtex |
| unit tests | < 0.1 s | – | |
| network / installs / GPU | 0 | – | git push only, by the session |

## Seventh session summary (Round 7; previous)

Checked at start: HEAD `9956c8f` equal to origin; clean tree; `paper/main.pdf` sha256
`2303c528…` as reported; the session TeX install and the page renders still present; no
CLAUDE.md in the repository; `/nvmedata` absent (not created). Verdicts kept:
**STUDY_SCOPE_FROZEN / METHOD_UTILITY_NOT_SUPPORTED_IN_THIS_PILOT**. Freezing the results and
asking for an external read are not alternatives: the results are frozen, and the manuscript
can be read externally; neither makes it submission-ready.

**Not done (by instruction):** no generation, topic, feature, dose/KL, semantic judge, model,
data, installation, retraining, statistics recomputation, fourth full manuscript review or
per-page subagent pass. No reviewer agents were spawned; this round used my own read plus the
existing automatic checks (tex_check, build log, render script).

**One editing pass (v4.2; wording only, no number, table or result changed).**
1. Core claims re-read against the frozen scope: C41 (coordinate caveat resting on a known
   LayerNorm property; the text says "a measurement caveat, not a new property"), C38
   (conditional fidelity of 8 selected features on a full common denominator), C48 (sparse
   lexical proxy null). They match; no claim text changed.
2. The two remaining `\todo` markers (P3-REAL-01R-FULL, P3-REAL-02-SEM) are follow-ups
   outside the frozen scope, not evidence the three core claims need: C38 is stated as
   conditional on the 8-feature pilot and C48 as lexical. They stay as internal TODOs
   (STATUS, RESEARCH_PACKET, claims.csv notes) and appear in the manuscript only as a new
   "Not evaluated" limitation; nothing was filled in.
3. Unchanged: 0/32 discordance, 8.94% / 10.89% model-based bounds, 28/32 reused articles,
   3 DiffMean discordant articles (net +1), the "All-Star" hit.
4. Checked: no sentence says LN is a better steering method or that steering is ineffective.
   Internal drift reduction and the external lexical null both stay. The words "beats", "won"
   and "better" occur only as quantified instance counts in the toy and synthetic-proxy
   results, with the proxy caption stating they are fixed by construction.
5. The opening sentences describe the observation (steering reports an edit as a feature
   change; we check the report in three steps) and separate theory (Prop. 4.1 with its GPT-2
   assumptions), internal measurement (Sec. 5) and behavior (Sec. 6). One sentence was added
   to Sec. 6.2 stating the unit, the primary estimand and the secondary status of everything
   else.

**Build (once, because sentences changed):** `paper/main.pdf` v4.2, sha256
`24ced5541de77435…`, 20 pages, Conclusion ends on page 7, appendix starts on page 11, 0
overfull boxes, 0 undefined references, 0 font warnings, author metadata "Anonymous Authors".
Changed pages 5–7 were rendered and read; other pages were not re-inspected (their content
did not change; the earlier full inspection applies to the v4.1 hash only). Body word count
≈ 4,400 (proxy, not the page check).

**Packages (existing generator, re-run):** `export_bundle/p3_v4_1_anonymous.tar.gz` and
`p3_v4_1_internal.tar.gz` now hold the v4.2 sources and PDF (the file names keep the v4_1
tag; the manifests carry the hashes). The anonymous package was unpacked and compared with
the README list: every listed item present, internal files absent, no repository commit id
or identifying string found by the scan, and it rebuilds standalone to the same 20 pages.

**External read review:** three questions and the status **UNASSIGNED** are in
`paper/README.md`. No one has been contacted; no acceptance or independent verification is
claimed.

**File hashes (v4.2):** main.tex `46fa4fc929a24b9e…`, claims.csv
`3ad8938e27bee75f…`, paper/README.md `95557a9d8ae37dd4…`. Executable code
unchanged this round (`src/`, `configs/`, `tests/` untouched); only documents changed.

### Seventh session: cost ledger

| step | wall | notes |
|---|---|---|
| state check, reads, edits | interactive | no agents |
| pdflatex ×3 + bibtex (v4.2) | 7 s | reused the session TeX install; no network |
| render 20 pages | ≈ 1.5 s | pymupdf, local |
| packages (anonymous + internal) | ≈ 6 s | existing generator |
| anonymous package unpack + standalone rebuild | ≈ 12 s | |
| network / installs / GPU | 0 | |

CPU-seconds were not measured separately (unmeasured).

## Sixth session summary (previous)

Checked at start: HEAD `09db0f8` equal to origin; clean tree; venv and HF cache present; no
TeX; `/nvmedata` absent (not created). Decision A: no behavioural extension (B not approved),
no new topic/feature/lexicon/dose/model/SAE/generation/semantic judge. No model test or
generation was rerun. Study status: **METHOD_UTILITY_NOT_SUPPORTED_IN_THIS_PILOT /
STUDY_SCOPE_FROZEN** (an operational label for this study, not a verdict on SAE steering).

**Statistics relabelled (derived file only; `results/real02_space/sensitivity_bounds.json`;
`summary.json` unchanged).**
- Primary LN − decoder any-hit: 0, discordance 0 of 32, as observed. The [0, 0] bootstrap is
  degenerate (every per-article difference is 0) and is not equivalence evidence.
- Under an iid Bernoulli discordance model (unverified; 28/32 articles reused from the
  internal pilot; one fixed seeded order): one-sided 95% upper bound 1 − 0.05^(1/32) =
  8.94%; two-sided 95% Clopper–Pearson upper 1 − 0.025^(1/32) = 10.89%. The 10.9% reported
  in v4 is the latter and is now labelled so. Both bound the discordance probability; they
  are not an observed difference. |E[hit_LN − hit_dec]| ≤ Pr(discordance) is used only under
  the same model.
- DiffMean vs no edit: 3 discordant articles (2 gained, 1 lost), net +1; CP 95% on its
  discordance probability 2.0–25.0% (descriptive).
- Text-change counts 24–31/32 are post-hoc description, not a substitute outcome.
- "All-Star" → "star" stays a legitimate whole-word hit; no post-hoc filter.
- mean-only canonical ≈ 0 kept; raw off-slice drift stays withdrawn.

**Manuscript v4.1** (`paper/main.tex`, edited in place; body ≈ 4,235 words).
- Central question narrowed; invariance stated as a known property; title/abstract state
  the pilot was run, with no improvement/equivalence/semantic claim.
- Limitations added: dose not extended on the same topic; sensitivity bounds model-based
  with reused articles.
- Study-status paragraph in the appendix; claims C48/C50/C31 updated, C51 added.

**PDF built locally** (limited build approval): `paper/main.pdf`, 20 pages, the last
sentence of the Conclusion is on page 7 (Impact Statement and References start on page 7;
the one-column appendix starts on page 11), no undefined references or citations, 0 overfull
boxes, 0 font-shape warnings. Layout fixes applied to reach that (no font, margin or scaling change): URL line
breaks after letters/punctuation in the preamble (bibliography hash URLs); the appendix dose
figure resized for one-column width; Table 1 `\tabcolsep` 4 pt; the appendix interval
table split into a drift part and a quality part (KL in units of 10⁻³ nats; rounding-level
target errors shown as <10⁻⁷); contract, proxy and denominator appendix tables set in
`\small` instead of `\scriptsize`; number formatting without "e-04" strings; a sentence
under the empty "Model–SAE Contract" heading; two appendix floats moved after their text;
acronyms in bibliography titles brace-protected (SAE, GPT-2, Gemma 2, …); the paired-difference
appendix table drops the redundant ΔD_orig column so it fits at a readable size.
Pages rendered at 110 dpi (pymupdf) and inspected by me and by per-page agents. PDF metadata: title set, author "Anonymous Authors" (style default), no personal
metadata. Not sent to any web compiler. The anonymous package was unpacked in a clean
folder and rebuilt with the same TeX install: same page count, 0 overfull, 0 undefined.

**Packages** (`export_bundle/p3_v4_1_anonymous.tar.gz`, 113 files, 4.0 MB;
`export_bundle/p3_v4_1_internal.tar.gz`, 117 files). Both include `paper/main.pdf`. The
anonymous build now also replaces this repository's commit identifiers with `<commit>` in
shipped text files (4 files; the frozen configs are byte-identical otherwise and remain
valid JSON), ships a `SUPPLEMENT_README.md`, and checks that every item the manuscript says
the supplement contains is present. The identifying-string scan is a string check, not a
proof of anonymity. Third-party notices (ICML style files, PPLM Apache-2.0 license) and all
reproducibility pins (revisions, sha256, PPLM commit) are kept.

**Verification (adversarial, agent-based; results read and acted on).**
- Manuscript lenses (statistics wording, numbers vs sources, claim map, anonymity/packaging,
  LaTeX structure), each finding challenged by two skeptics: the numbers lens found no
  mismatch; the other lenses' findings were fixed (post-hoc text-change count out of the
  abstract and stated per method; "upper end of a two-sided CP interval" wording; "with the
  decoder, encoder-row and corrected edits matched" instead of "every method"; closure
  agreement stated for the 760 edits at 16 test points; claim-map locations refreshed to v4.1
  numbering; internal-only evidence pointers marked; repository commit ids redacted in the
  anonymous package; paragraph-level labels removed; small caps no longer nested in italics).
- Per-page PDF checks on the final render (three passes; the last one caught a verdict row
  in Table 11 that my own generator had broken, fixed before packaging): body ends on page 7;
  no margin overflow, clipping, "??" or identifying strings. Remaining accepted cosmetics: loose justified
  spacing next to unbreakable inline intervals; blank space before the trailing appendix
  float pages; the two red TODO markers (P3-REAL-01R-FULL, P3-REAL-02-SEM) kept on purpose.

### Sixth session: build environment and cost ledger

| step | wall | network (RX counter) | disk | notes |
|---|---|---|---|---|
| TeX Live scheme-basic (install-tl from the CTAN mirror network, user folder in the session scratchpad, no sudo) | 118 s | ≈ 111.1 MB | 187 MB | official TeX Live distribution; docs/sources off |
| tlmgr: 15 packages (booktabs, caption, cleveref, eso-pic, float, forloop, fp, latexmk, mathtools, microtype, pgf, pgfplots, xcolor, xkeyval, everyshi) + fonts (times, courier, helvetic) | 8 s + 5 s + 5 s | ≈ 2.5 MB | 4 MB | two package names were wrong in the first list (`ms`, `urw-base35`); corrected by file search |
| pymupdf 1.28.2 into the scratch venv (PyPI, `--no-deps`) | 5.6 s | pip cache / ≤ 25 MB | 67 MB | local page rendering only |
| pdflatex/bibtex passes (7 pdflatex + 1 bibtex, incl. 3 rebuilds after layout fixes) | ≈ 45 s total | 0 | 1.3 MB | first pass failed on missing Courier metrics; fixed by installing `courier` |
| page rendering (3 × 19 pages) | ≈ 1.5 s each | 0 | 2 MB | 110 dpi PNGs in the scratchpad |
| verification workflows (manuscript rules/numbers/claims/anonymity/LaTeX; per-page PDF checks) | parallel agents | 0 | – | results below |

Totals: TeX install + first PDF 16:08–16:13 UTC (≈ 5 min of the 30-min cap); all later
rebuilds (≈ 20 pdflatex passes, 5 bibtex runs, 3 standalone rebuilds from the anonymous
package) were each 5–12 s. RX counter delta over the whole 16:08–16:53 window ≈ 137.5 MB,
an upper bound on the build-related network use (cap 3 GiB); scratch disk ≈ 258 MB for TeX
Live (160 packages installed) + pymupdf, plus ≈ 28 MB of renders and the unpacked test copy
(cap 6 GiB). CPU-seconds were not recorded for the TeX steps. Agent verification: 4
workflows, 131 subagents, ≈ 7.3 M tokens.

## Fifth session summary (previous)

Checked at start: HEAD `78d95ac` equal to origin; clean tree; venv and HF cache (672 MB) present;
no LaTeX compiler; `/nvmedata` does not exist in this sandbox, so it was not created.

**Kept verdicts (unchanged).**
- Raw mean-only drift ≈ 125 and canonical drift ≈ 0 measure different things.
- The v2 interpretation stays withdrawn, and the float32 CONTRACT_FAIL stays.
- Wedding LX v1/v1.1: calibration BLOCKED; test and generation NOT_RUN. The branch is
  closed; 13/1943 is lexical support, not a steering failure.

**External read (only what the instruction named).**
- `git ls-remote` of uber-research/PPLM gave commit `e236b898…`.
- Fetched `paper_code/wordlists/space.txt` (142 B, 18 words, sha256 `de1a50a4…`), LICENSE
  (Apache-2.0) and the arXiv abstract page (metadata).
- No weights and no corpus.
- Files are in `data_external/pplm_space/` with PROVENANCE.json.
- Separately, a status-only check of the three ICML 2027 URLs again returned 404.

**P3-REAL-02-SPACE** (`configs/p3_real02_space.json`, frozen in commit caf43c8 before any
count).
- Matching rules: bounded whole-word (`lexicon.bounded_hits`, unit-tested).
- Support (text-only): 58/1943 calibration windows in 22/60 articles (max 9 per article;
  "star" 28) → SUPPORTED.
- Calibrate: RAN, 961 s; feature 4671; every actor at 0.1× the median norm under κ = 0.1
  (cal KL 0.021–0.054).
- Run: RAN, 1334 s; 32/32 articles attempted, none skipped, 0 failed rows; 28 of them were
  edit sites in the REAL-01R pilot.
- **Primary LN − decoder any-hit: 0 with 0/0 discordant articles.** The bootstrap interval
  degenerates to [0, 0]; the exact 95% bound is 10.9%. Verdict: NOT_SUPPORTED, which is not
  equivalence.
- Decoder, encoder row and LN never changed occurrence relative to no edit. The single hit
  is "All-Star" → "star" (polysemy). DiffMean gained 2 articles and lost 1.
- Post hoc, descriptive: edits changed 24–31/32 continuations; mean-only changed 0/32.

**Manuscript v4** (edited in place; body ≈ 3960 words, down from 5170).
- Three core claims: readout caveat (C41), fidelity with full denominator (C38), null
  lexical pilot (C48).
- Three core tables: coordinates, fidelity + denominators, space pilot.
- Wedding records moved to the appendix.
- **COMPILE_NOT_RUN** (no TeX).

**Packages.**
- `export_bundle/p3_v4_anonymous.tar.gz`: anonymity scan passed; claimed contents present;
  no internal notes.
- `export_bundle/p3_v4_internal.tar.gz`.
- Weights are not included; they are pinned by revision + sha256 in
  `configs/p3_asset_hashes.json`.

### Fifth session: time and cost ledger

wall = elapsed; CPU-s = process CPU over all threads; threads = torch intra-op setting.

| step | wall | CPU-s | threads | result read? |
|---|---|---|---|---|
| PPLM ls-remote + 2 files + arXiv page | < 5 s | – | – | yes (list, license, metadata) |
| ICML 2027 URL status check | < 5 s | – | – | status codes only |
| unit tests (stdlib) | < 1 s | – | 1 | PASS |
| adapter model tests (3) | 93.8 s (cold model load) | not recorded | 2 | PASS (not hypothesis support) |
| asset sha256 (local cache) | 0.7 s | – | 1 | – |
| SPACE support (tokenizer only) | 11.7 s | 6.1 | 1 | yes (support counts) |
| SPACE calibrate (scan 742, KL sweep 189, cost probe 28) | 961 s | not recorded | 2 | calibration only |
| SPACE run (32 × 6 generations) | 1334 s | not recorded | 2 | not read until summary |
| SPACE summarize | 1.2 s | – | 1 | yes |
| post-hoc continuation-change count | < 1 s | – | 1 | yes, after summary (labelled post hoc) |
| paper assets / tex_check / packages | 1–3 s each | – | 1 | – |
| LaTeX build | – | – | – | COMPILE_NOT_RUN |

Cumulative SPACE compute: calibrate + run = 2295 s wall, within the fixed 60-minute budget.

**Next decision (one, needs the user):** see RESEARCH_PACKET §14.

## Fourth session summary (previous)

Scope, as instructed: one readout-closure check, one re-aggregation, one limited lexical pilot
(P3-REAL-02-LX), and manuscript v3. No new project, gate, toy or feature sweep. No new
installation or download: everything reused the scratch venv and the HF cache (672 MB)
from the third session, run offline (`HF_HUB_OFFLINE=1`).

**Approval provenance.** The third session's `approval_interpretation` in
`run_manifest.json` was Claude's reading of an instruction to execute. It was not a user
sentence approving installs or downloads. This session installed and downloaded nothing.
The user's server path `/nvmedata/dgkang/papers/<project_id>/` was not used: this checkout
is a separate cloud sandbox, and nothing was moved.

- **P3-READOUT-CLOSURE (new, executed).** The K9 "drift 125×" and every pilot mean-only row
  are *raw off-slice* readouts, `E(x + δ) − E(x)`. We recomputed the stored numbers exactly
  (rel. 4.6e-16; `raw_run.csv` columns exact). The *canonical reprojected* readout
  `E(P(h + δ)) − E(Ph)` of the mean-only edit is ≤ 3.8e-15, and its KL is ≤ 1.1e-15. For
  admissible edits the two readouts agree to 3.6e-12, so no admissible result changes.
  - **Corrected interpretation:** the third session's "real-model instance of internal
    change without external effect" and "internal efficiency maximized by a model-invisible
    direction" (mean-only gain 1.85) are withdrawn. The numbers are an off-training-subspace
    stress test of the encoder.
  - `results/readout_closure/`; 843 s wall.
- **Math.** Mean invariance is proved for GPT-2 only (mean-subtracting LN readers, exact
  arithmetic) and is UNPROVED elsewhere. `range(P)` is a chosen representative, not the
  unique valid space. `J_eff = J_E(Ph) P`.
- **Matched-denominator re-aggregation (new derived files only;
  `results/real01r_reagg/`).**
  - Decoder, encoder row and LN are matched at 100% of test points at every dose, so the
    main pilot comparison already used the full common denominator.
  - Random directions are infeasible at 52–55% of points.
  - The rescaled decoder is matched for 256/256 active-target rows and 0/256
    inactive-target rows.
  - The 512 mean-only target-matched rows exist only under the raw readout and are excluded.
  - G1 as run (0.426) pooled matched and unmatched rows. Split, it is 0.583 active and
    0.342 inactive, both on the same side.
  - The G1–G3 min-unit rule and target_gain are labelled **in-run amendments**.
  - q99 rests on 26–134 positive calibration activations. For 3 of 8 pilot features it lies
    between the two largest values.
- **P3-REAL-02-LX: BLOCKED at calibration** (`results/real02_lx/`).
  - The config (`configs/p3_real02_lx.json`, commit 656b6f0) was frozen before any data read:
    ≤ 32 test docs, one prompt each, greedy 32 tokens, edit at every position, KL cap κ = 0.1
    on the common prompt; projected decoder, projected encoder row, LN (`J_E P`), projected
    DiffMean, and no edit. mean_only is a fixed-amplitude control outside every ranking;
    random is NOT_RUN.
  - v1 calibrate: **BLOCKED**, 5 calibration windows with a lexicon word against the
    pre-set minimum of 20 (cap of 8 windows per article); 158 s.
  - A single-shot amendment v1.1 (`configs/p3_real02_lx_v1_1.json`, commit 876e77c) was
    committed before rerunning: no window cap, everything else unchanged, and a second
    BLOCKED ends the experiment.
  - v1.1 calibrate: **BLOCKED**, 13 of 1943 windows; 630 s.
  - The run stage recorded BLOCKED and summarize recorded NOT_RUN. **No test document was
    read, and no text was generated.** min_hit_windows was not lowered.
  - Adapter checks on a synthetic prompt pass (`tests/test_real02_adapter.py`, 9 s). Passing
    tests do not support any hypothesis.
  - Exact blocker: the only approved calibration text (WikiText-103 validation at the
    pinned revision) lacks wedding-lexicon support. Unblocking needs a new frozen protocol:
    more topic text (a new data download, which needs approval) or a topic that WikiText
    supports.
- **Manuscript v3** (`paper/main.tex`, edited in place).
  - Order: (a) raw/canonical coordinates, (b) admissible realization with the toy
    redistribution results kept in the body, (c) matched denominators, (d) lexical change
    and quality.
  - The title was narrowed.
  - **COMPILE_NOT_RUN**: no TeX installed. Sources and exact build commands are in
    `paper/README.md` and `export_bundle/`.

**Next research decision (one, needs the user).** To reopen the behavioural test, either:
- (A, recommended) pre-register a published steering-topic lexicon that the approved
  WikiText calibration text supports (≥ 20 windows, checked on calibration only); or
- (B) approve new calibration text for the wedding lexicon.

See RESEARCH_PACKET §12.

### Fourth session: time and cost ledger

Terms used below:
- **wall** is elapsed time.
- **CPU-s** is process CPU time summed over all threads, so it can exceed wall.
- **threads** is the torch intra-op thread setting.

| step | wall | CPU-s | threads | outcome |
|---|---|---|---|---|
| installs / downloads | 0 | – | – | none (reused venv + HF cache, offline) |
| readout closure (`readout_closure.py`) | 843 s (load 86, part A 22, part B 734) | 1481 | 2 | RAN |
| matched-denominator re-aggregation, first attempt | < 5 s | – | 1 | FAILED (CSV fieldnames), fixed |
| matched-denominator re-aggregation | 5.2 s | 5.0 | 1 | RAN |
| unit tests, stdlib (41) and torch (1) | 0.5 s + 1.1 s | – | 1–2 | PASS (4 skipped = model tests) |
| REAL-02 adapter model tests (3) | 9.0 s | not recorded | 2 | PASS |
| REAL-02-LX v1 calibrate | 158.4 s | not recorded | 2 | BLOCKED (5 < 20) |
| REAL-02-LX v1.1 calibrate | 629.6 s | not recorded | 2 | BLOCKED (13 < 20) |
| REAL-02-LX run / summarize | 2.0 s / 0 s | – | – | BLOCKED / NOT_RUN |
| paper_assets and tex_check (several runs) | 1–2 s each | – | 1 | RAN |
| LaTeX build | – | – | – | COMPILE_NOT_RUN (no compiler) |

Total heavy compute this session was about 1650 s wall on 2 threads.

### Fourth session: NOT_RUN and blockers
- P3-REAL-02-LX lexical metrics, KL/NLL tradeoff: **BLOCKED** at calibration (above).
- P3-REAL-02-SEM (semantic evaluation): NOT_RUN.
- P3-REAL-01R-FULL (all 64 features): NOT_RUN, not requested this session.
- PDF build and page count: COMPILE_NOT_RUN. No TeX distribution is installed, and
  installing one was not approved.

## Third session summary (previous)

- **Order.** P3-CONTRACT-EXEC was run before any REAL-02 work, as instructed. Model/SAE
  revisions, layer 8 and hook site are unchanged.
- **Resources.**
  - The instruction to execute was read as approval for the resources listed in the second
    session: CPU packages and the pinned files.
  - The packages are installed in a scratch venv outside the repo: torch 2.14.0+cpu,
    transformers 4.57.6, transformer-lens 2.18.0, sae-lens 6.51.3 and pyarrow.
    `results/contract_exec/venv_freeze.txt` lists all 105 packages.
  - The pinned files come to 703,545,978 B, exactly the contract total.
  - Everything runs on CPU with 2 threads. No GPU, paid API or upload was used.
- **Contract v1 (float32, pre-registered): `CONTRACT_FAIL_BLOCKED`. Kept.**
  - K0–K5 pass.
  - K6 and K7 fail the probability tolerance: 2.8e-5 against 1e-5. KL (≤ 7.7e-7) and the
    logit residual after removing the center_unembed shift (4.3e-4) pass.
  - K9 fails the mean-only KL bound: 6e-7 against 1e-8.
- **Contract v2 (float64 diagnostic, same tolerances, written after v1):
  `CONTRACT_PASS_FLOAT64`.** Every backend difference is 1e-13–1e-15. The HF shortcut
  (`x = h − mean(h)`) matches canonical TransformerLens/SAELens for activations, features,
  post-edit features and post-edit next-token distributions. The no-processing loading
  path recommended by SAELens (K10) also matches. The float32 failure is rounding on top
  of a common logit shift of up to 279.
- **Coordinates.**
  - The uncentred HF state is wrong for this SAE: FVE 0.003 against 0.83 centred.
  - Admissible edits are `range(P)`, `P = I − 11ᵀ/d`; the model is invariant to the
    1-direction (proved).
  - The LN correction now uses `J_E P`.
  - Raw decoder edits include a mean component. For them, the TransformerLens and
    HF-shortcut conventions give SAE readouts that differ by 51–114% of the edit's effect.
- **Mean-only diagnostic control.** ~~A real-model instance of internal change with no
  external effect~~ *(corrected in the fourth session: raw off-slice readout; canonically 0)*:
  model KL ≤ 1.3e-15, while the raw SAE readout moves by a drift/ρ of 125 (median).
- **REAL-01R pilot: RAN.** Config `configs/p3_real01r_pilot.json`, frozen before the test
  run.
  - Setup: float64; 8 features, a seeded subsample of 64 calibration features; 54 test
    documents; 6144 rows (272 INFEASIBLE, all random-direction target matching, kept);
    run 1042 s.
  - Target-matched, q99, active targets: decoder total drift 0.656α; LN (`J_E P`) 0.114α.
    Per-document paired ΔD_all = −0.50 [−0.58, −0.42] (bootstrap over documents),
    ΔD_new = −0.17, ΔKL = −0.0038 nats. The LN edit norm is 12.7, against 21.6 for the
    decoder.
  - Against the encoder row, LN has lower drift and nearly the same KL.
  - ~~At equal norm, the mean-only control has the highest target gain (1.85) with model KL
    of about 0.~~ *(fourth session: raw off-slice readout; the canonical gain is 0.)*
  - G1–G3 read "continue side" but are descriptive only (pilot).
  - The real SAE (L0 ≈ 72 of d = 768) behaves like the toy's sparse regime, not the dense
    one. This is consistent with the toy, but not a test of it.
- **Manuscript v2.** `paper/main.tex` is restructured as coordinate validity → fidelity →
  redistribution → external evaluation. **COMPILE_NOT_RUN** (no LaTeX). Static checks
  pass. Three TODOs remain: REAL-01R-FULL, REAL-02 (lexical proxy) and REAL-02-SEM
  (semantic).
- **Deviations recorded.**
  - Contract v2 (float64) was designed after the v1 failure; tolerances were unchanged.
  - The smoke-stage cost projection double-counted fixed costs, so the pre-registered
    N = 8 was kept on a corrected estimate.
  - The gate min-unit rule (≥ 10 documents) and the target-gain metric were added during
    the run, before results were read.
  - A contract-stage window-batching bug was fixed before the stage passed.
- **REAL-02.** NOT_RUN. The wedding keyword metric is a lexical proxy only; semantic
  success is NOT_RUN (no independent semantic evaluation).


## Second session summary (previous)

| Layer | Status |
|---|---|
| Existing toy raw/config/labels | **Preserved.** No experiment was rerun. Raw sha256 is unchanged (`run_manifest.json → second_session`). T4b stays EXPLORATORY. The dense-T4 failure, target abandonment, leakage moving to unprotected features and the internal/external mismatch are all kept. |
| Aggregation correction | Repair convergence is now reported as row counts: 72–80/80 per ReLU/JumpReLU cell and 2–66/80 per TopK cell. The first session's "100%" / "0–88%" were medians of per-instance rates. This was re-aggregated from raw by `src/saeedit/paper_assets.py`. |
| Model–SAE contract | `configs/p3_contract_gpt2_res_jb_l8.json`: revisions, hook, width, dtype, context, licences and file sizes come from source. The activation, input-bias handling and normalisation are SAELens defaults because `cfg.json` omits them. Coordinates follow `center_writing_weights=True`: SAE input = HF hidden state − per-token mean (**derived**, not executed). |
| Contract checks | C1 **executed: PASS** on the verbatim configs (`tests/fixtures/`). C2 is inferred from file size (header not read). C3 (no-op hook logits) and C4 (L0/FVE) are **NOT_RUN**. |
| P3-REAL-01R adapter + CLI | Written (`src/saeedit/real/`). Stdlib parts are tested. The torch backend is **NOT_RUN**. `check-env` → `BLOCKED_DEPENDENCIES`; `contract` → `BLOCKED` (`results/real01r/stage_status.json`). |
| P3-REAL-02 adapter + CLI | Written: wedding task, ActAdd lexicon label, DiffMean baseline, calibration-selected KL budget. **NOT_RUN** (`results/real02/stage_status.json`). |
| Manuscript v1 | `paper/main.tex`: all sections written. **COMPILE_NOT_RUN** (no LaTeX compiler). Static checks pass (`results/reaggregated/tex_check.json`). TARGET_YEAR=2027, TEMPLATE_YEAR=2026 (official ICML 2026 style, unmodified), `SUBMISSION_READY=false`. Two `\todo` markers remain: REAL-01R and REAL-02 results. |
| Claims map | `paper/claims.csv` (32 claims → evidence, experiment IDs, assumptions, status). |
| H_MAIN / R1–R3 | **NOT_RUN** |
| New-method claim | **Not made.** The prior evidence is against it: SAE-TS pseudo-inverse, Cui et al., toy TX. |

**Next decision experiment (third session):** the `P3-REAL-02` lexical-proxy pilot, with a
mean-only negative control and the semantic evaluation still NOT_RUN. It asks the first
external question the internal pilot cannot answer. The earlier text below is kept.

**Next decision experiment (second session, superseded):** `P3-REAL-01R` (§7 below). **Needs approval** for:
- installing Python packages: torch (CPU), transformers, safetensors, huggingface_hub,
  pyarrow;
- downloads of 703,545,978 bytes listed in the contract (GPT-2 files 550,959,861 B, SAE L8
  151,196,298 B, WikiText-103 raw val/test 1,389,819 B), plus package sizes, which have not
  been verified.

CPU time is unknown until the timing-smoke stage runs.

---

# First session (preserved)

## 0. Pre-existing state (checked, not assumed)

- The repository had **no commits and no files** when this session started: the remote
  had no branches, and the local checkout had only `.git`.
- There was no `CLAUDE.md`, earlier `STATUS.md`, `RESEARCH_PACKET.md`, config or result.
- So there were **no earlier STOP/ARCHIVE verdicts and no earlier Work IDs**. "P3" is the
  user's label for this project. Experiment IDs below (`P3-T*`, `P3-TX`, `P3-REAL-*`)
  are new and are not mapped onto any other scheme.
- Environment: Python 3.11.15 with the standard library only. numpy, torch and pytest
  are **not installed**, and installing them was not approved. Everything here is pure
  Python with `unittest`.

## 1. Engineering status vs scientific status

| Layer | Status |
|---|---|
| Toy code (`src/saeedit/`), 19 unit tests | RAN; 19/19 pass (`PYTHONPATH=src python3 -m unittest discover -s tests`) |
| First run, report seeds 1000–1019 | RAN; 76 s wall clock on 1 thread (budget 120 s, 2 threads) |
| Determinism | The run was repeated after a summary-only bug fix. All 5 raw CSVs are **bit-identical** (sha256 in `run_manifest.json`). |
| Verification identities V1–V5, V7, T3 (8 checks) | PASS (these are checks of the code, not findings) |
| V6 (ReLU piece exactness) | PASS (550 rows, max error 3.3e-15) |
| V8 (synthetic external proxy) | DESCRIPTIVE (see `RESEARCH_PACKET.md` §4.4) |
| **H_MAIN / R1–R3 on language models** | **NOT_RUN** |
| Scientific support for a new steering method | **None.** Toy results are mostly fixed by linear algebra, and the literature already covers the solver (`RELATED_WORK.md` §6). |

## 2. Experiments

| ID | What | Pre-registered? | Status | Raw | Wall |
|---|---|---|---|---|---|
| P3-T2-MISMATCH / P3-T5-RESCALE-VS-INTERF | Linear regimes R0–R5 × 8 methods | yes (v1) | RAN | `results/raw/linear_regimes.csv` | 9.1 s |
| P3-T3-FEASIBILITY | Least-norm, Fredholm certificates, duplicate rows, range and TopK infeasibility | yes (v1; tolerance made explicit in v2 before the report run) | RAN, 260/260 checks pass | `results/raw/feasibility.csv` | 1.2 s |
| P3-T4-ACTIVESET-NORM | ReLU/JumpReLU/TopK, active vs inactive targets, α ∈ {0.25, 1, 4}, norm budgets ×{0.5, 1, 2} | yes (v1) | RAN | `results/raw/activeset_norm.csv` | 50.4 s |
| P3-T4b-SPARSITY-EXPLORATORY | ReLU bias −0.5 / −0.75 (sparser active sets) | **no**: added after inspecting dev seeds 0–2 | RAN (32 SKIP rows: no active feature at the base point) | `results/raw/sparsity_exploratory.csv` | 14.4 s |
| P3-TX-EXTPROXY | Synthetic external readout; internal vs external transfer | yes (v1) | RAN | `results/raw/external_proxy.csv` | 0.6 s |
| P3-REAL-01 | Same-layer fidelity gate on a public SAE | specified below | **NOT_RUN** (needs approval) | — | — |
| P3-REAL-02 | Held-out behaviour, quality, norm–quality, cost | protocol in `RESEARCH_PACKET.md` §5 | **NOT_RUN** | — | — |

Summary: `results/summary.json`. Manifest: `run_manifest.json`. Config:
`configs/p3_first_run.json` (`p3_first_run_v2`; the changelog is inside the file).

## 3. What the first run shows (toy only)

1. **Step 2 (mismatch).** With a linear encoder, the re-encoded change of a decoder edit
   is `αEDe_j` exactly. The toy reproduces the closed forms: tied, equiangular c=0.3,
   m=8 gives leakage 0.794 = 0.3·√7.
2. **Step 3 (feasibility).**
   - Full rank: least-norm solution with Pythagorean minimality, 80/80.
   - Protect-all with m > d: infeasible, with a valid certificate (R4a relative residual ≥ 0.69).
     The least-squares solve then gives up the target (error 0.675).
   - Duplicate encoder rows with conflicting targets: residual exactly α/√2.
3. **Step 4 (active sets and norm).**
   - ReLU: local prediction is exact when nothing crosses (proved).
   - The naive post-activation Jacobian cannot move an inactive target.
   - JumpReLU values in (0, θ] are unattainable. TopK must evict a feature.
   - In the pre-registered dense regime (median 13 active features for d=16), the
     correction has **more** total leakage than the rescaled decoder in almost every
     cell, because of threshold crossings.
   - "Repair" converges only by giving up the target.
4. **Step 5 (rescaling vs interference).**
   - The regimes separate into NO_MISMATCH (R0), RESCALE_SUFFICIENT (R1),
     INTERFERENCE_REMOVED_BY_CORRECTION (R2, R3; norm cost ×1.14–1.24),
     INFEASIBLE (R4a) and INTERFERENCE_MOVED_TO_UNPROTECTED (R4b, R5).
   - In R4b and R5, protecting 7 features *increased* total leakage.
5. **Exploratory.** With sparser active sets, the correction beats rescaling on internal
   leakage in most instances, but only by small margins, and the advantage disappears
   at large α when the active set is moderate.
6. **External proxy.** The internal metric becomes perfect in every case. The external
   error improves only if the encoder is the accurate side: it gets worse in 20/20
   instances when the decoder is accurate, and there is no change (11/20) when both are
   noisy.

## 4. Known issues and deviations (disclosed)

- **The config changed after dev seeds, before the report run** (v1→v2): the T3 CGLS
  tolerance was made explicit, `low_rank_rows` was added, and T4b was added as
  exploratory. v1 sections were not modified.
- **Summary bug found after the first report run:** boolean flags were dropped from the
  aggregates, and V8 had a misleading median-only "external_improvement" flag. It was
  fixed in commit `46dbb25`. The rerun reproduced bit-identical raw files, so only the
  summary changed.
- `mismatch_diag_share` in the raw CSVs is numerical noise wherever the total mismatch is
  ≤ 1e-8. Use `P3-T5-RESCALE-VS-INTERF.decoder_diag_share_median` instead.
- For TopK, active-set changes are counted by comparing endpoints, which is a lower bound
  on changes along the path (`RESEARCH_PACKET.md` P7).
- Trust region vs decoder at matched norm is **circular** (it optimises the scored
  internal metric), so it is not evidence for the method.
- The T4 v1 regime is denser than public LM SAEs. The claim that public SAEs are sparser
  is an order-of-magnitude expectation that has not been verified here.
- Related-work reading was done through a WebFetch extraction model, not by reading raw
  full text. Three decisive claims were spot-checked (`RELATED_WORK.md`).

## 5. NOT_RUN (explicit)

- Any language model, any public SAE, and any held-out behavioural or quality metric.
- The R2 predictive test against decoder-geometry predictors.
- The SAE-TS, FGAA, DiffMean and global pseudo-inverse baselines.
- GPU anything.

## 6. [SUPERSEDED by §7] Next decision experiment: `P3-REAL-01` (first session design; kept verbatim)

**Purpose.** A cheap gate that can only **STOP** the method claim or permit `P3-REAL-02`.
It measures, on real SAE weights and real activations, what the toy says decides the
outcome: how much post-rescale leakage lands on *active* features, which a local
correction can address, and how much comes from threshold crossings, which it cannot.
It also produces the descriptive same-layer fidelity decomposition (gap #1 in
`RELATED_WORK.md` §6).

**Fixed before running (to be frozen in `configs/p3_real01.json`):**
- **Model:** GPT-2 small (HF `openai-community/gpt2`; record the commit hash at download).
- **SAE:** `jbloom/GPT2-Small-SAEs-Reformatted`, `blocks.8.hook_resid_pre`, d_sae 24576
  (record the commit hash). Verify the activation from the loaded config; if it is not
  ReLU, record that and use the matching activation, without silent substitution.
- **Data:** WikiText-103 validation (license to be confirmed at download). There are
  two disjoint sequence sets:
  - a calibration set (300 sequences), used for the feature pool and `a_max(j)`;
  - a test set (300 sequences, one random non-BOS position per sequence).
- **Features:** 100 features sampled with seed 0 from those with calibration firing
  density in [1e-4, 1e-2]. **Independent unit = feature**; base points are nested
  inside a feature.
- **α grid:** {0.5, 1, 2, 4} × `a_max(j)`.
- **Methods:** `decoder`, `decoder_rescaled`, `encoder_grad`, `jacobian_ln` (P = active
  set), `jacobian_ln_repair`, `trust_region_normmatched`, `random_normmatched`.
- **Metrics:**
  - target error;
  - protected leakage;
  - unprotected (crossing) leakage;
  - number of crossings;
  - |active|/d;
  - norm ratio;
  - diagonal share.
- **STOP the method claim** if any of the following holds at α ≥ 1·a_max (median over
  test features):
  - **G1:** the rescaled decoder's protected leakage is < 0.1·α;
  - **G2:** crossing leakage ≥ protected leakage for the rescaled decoder;
  - **G3:** `jacobian_ln` total leakage ≥ the rescaled decoder's in ≥ 50% of features.

  Otherwise **PROCEED** to `P3-REAL-02`. The thresholds are proposals for the user to
  confirm or change *before* the run.
- **Budget:** CPU only, 2 threads, ≤ 30 min wall clock, about 1 GB of downloads.

**Approvals needed:**
- installing numpy, torch-CPU and safetensors (or SAELens);
- downloading GPT-2 small, one SAE file and WikiText-103 validation from Hugging Face.

## 7. Next decision experiment (second session): `P3-REAL-01R`

The full specification is `configs/p3_real01r.json`. The changes from §6 are:
- **Unit.** The independent unit is the test **document**, not the feature. Documents are
  split before windows are made (WikiText-103 validation → calibration, test → test), and
  an overlap check runs before any evaluation.
- **Doses.** They come from calibration data only and are frozen with sha256 before any
  test window is encoded:
  - activation quantiles q50/q90/q99 and 2×q99 (`a_max` is recorded, not used);
  - norm budgets of {0.025, 0.05, 0.1, 0.2} × the median centred residual norm.
- **Features.** 64, drawn from a calibration density band with seed 0 before any test data
  is read.
- **Comparisons.** Equal-norm and target-matched are kept separate. Infeasible rows are
  kept.
- **Metrics.**
  - target change and error;
  - originally active drift, split into kept and deactivated;
  - newly active drift;
  - total non-target drift;
  - edit norm and mean(δ);
  - KL and ΔNLL;
  - time and peak RSS.
- **Targets.** Active and inactive targets are reported separately.
- **G1–G3.** The original text is preserved verbatim in the config. Each is read at q99 as
  a regime curve plus a paired difference with a document-cluster bootstrap interval, with
  the outcome STOP_SIDE, CONTINUE_SIDE or INCONCLUSIVE. These are operational rules, not
  significance tests. A large crossing share ends the method claim, not the measurement
  study.
- **Stages.** `check-env` → `contract` (C1–C4; BLOCKED on failure) → `calibrate` → `smoke`
  (calibration documents only; picks N) → `run` → `summarize`.
