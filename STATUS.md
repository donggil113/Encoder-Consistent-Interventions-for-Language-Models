# STATUS — P3: Do SAE Edits Change What They Claim?

Last updated: 2026-09-26 (second session). Branch: `claude/magical-bell-8vyrpd`.

## Second session summary (read first)

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

**Next decision experiment:** `P3-REAL-01R` (§7 below). **Needs approval** for:
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
