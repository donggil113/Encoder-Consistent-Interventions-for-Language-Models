# RESEARCH_PACKET — P3: Do SAE Edits Change What They Claim?

Last updated: 2026-09-26 (second session: real-model adapters and protocols, manuscript v1).
Toy config: `configs/p3_first_run.json` (`p3_first_run_v2`).
Real-model configs: `configs/p3_contract_gpt2_res_jb_l8.json`, `configs/p3_real01r.json`,
`configs/p3_real02.json`.
Evidence: `results/summary.json`, `results/raw/*.csv`, `run_manifest.json`.

Status labels: **PROVED** (a complete argument is written here or is textbook with a
reference), **CHECKED** (numerical or unit-test agreement only; not a proof),
**OBSERVED_IN_TOY** (a descriptive toy result), **CONJECTURE / UNPROVED**,
**NOT_RUN**.

---

## 0. Where things stand

| Item | Status |
|---|---|
| H_MAIN: the SAE-re-encoded mismatch of decoder edits predicts external steering failure on held-out prompts, and the local correction beats norm-matched rescaling on external behaviour at equal quality | **NOT_RUN** |
| Toy implementation (stdlib only), 19 unit tests | engineering: RAN, all pass |
| Toy verification identities V1–V5, V7, T3 (8 checks) | CHECKED, all PASS (`run_manifest.json → verification`) |
| V6 (ReLU piece exactness) | PROVED (§3 P5) and CHECKED (550 no-crossing rows, max error 3.3e-15) |
| V8 (internal vs external transfer, synthetic proxy) | OBSERVED_IN_TOY (descriptive) |
| Model–SAE contract (GPT-2 small + res-jb L8) | Verified from source (pinned configs, SAELens@ef4c208, TransformerLens@24b039f). C1 was **executed** on the verbatim configs and passes. C2 is inferred from file size (header not read). C3/C4 are **NOT_RUN**. |
| REAL-01R / REAL-02 adapters and CLIs | Written. The stdlib parts are unit-tested (38 tests: 37 pass, 1 torch test skipped). The torch parts are **NOT_RUN**. The `check-env` stage recorded `BLOCKED_DEPENDENCIES`. |
| Manuscript v1 (`paper/main.tex`) | All sections written. **COMPILE_NOT_RUN** (no LaTeX). Static checks pass. `SUBMISSION_READY=false`. |
| Novelty of the correction as a solver | **None claimed.** In the linear case it is a standard least-norm / minimum-norm least-squares solve. Its protected-set projector `I − Jᵀ(JJᵀ)†J` on SAE encoder readouts is already published (Cui et al. 2026, arXiv:2606.18322). SAE-TS reports that a pseudo-inverse did worse than its heuristic. "Encoder–decoder misalignment" has already been named as a cause of SAE steering failure (GLP, arXiv:2602.06964). See `RELATED_WORK.md` §6. |

---

## 1. Question and falsifiable claims

**Question.** A decoder edit `x + α D_j` is described as "+α on feature j". Re-encoding
with the same SAE gives the achieved change. (a) How much of the gap is a scale error
and how much is interference between features? (b) Does the gap explain failures in
external behaviour? (c) Does a local correction on target and protected features reduce
those failures beyond simple rescaling?

**Claims written so they can be refuted:**

- **R1 (prevalence).** In a fixed public SAE, at steering-relevant α, the post-rescaling
  leakage onto *active* features is a non-negligible fraction of α, and threshold
  crossings of inactive features do not dominate it. *Refuted if* post-rescale protected
  leakage is small, or crossings dominate. → **NOT_RUN** (`P3-REAL-01`)
- **R2 (explanation).** Across held-out features and prompts, larger same-layer
  re-encoded interference predicts lower external steering success at matched norm,
  **beyond** decoder-geometry predictors (decoder crowding and max cosine; Duan 2026,
  Khan 2026) and edit norm. *Refuted if* the incremental association is absent in the
  pre-registered test. → **NOT_RUN** (`P3-REAL-02`)
- **R3 (method).** The local correction improves held-out external concept success at
  equal or lower quality loss than norm-matched decoder rescaling, encoder-row steering
  and DiffMean. *Refuted if* it does not, or if it improves only internal SAE metrics.
  → **NOT_RUN** (`P3-REAL-02`). *Prior evidence points against R3:* SAE-TS's
  pseudo-inverse result, Cui et al.'s behaviour change with readouts held fixed, and toy
  TX (§4.4).

Stop rule (fixed in the config before the run): if only internal metrics improve, or
the correction cannot be distinguished from norm-matched rescaling on held-out external
behaviour and quality, stop claiming a new steering method.

---

## 2. Setup and notation

- Activation `x ∈ R^d`. SAE pre-activation `p = E x + b` (`E ∈ R^{m×d}`), features
  `a = σ(p)` with σ ∈ {identity, ReLU, JumpReLU(θ), TopK(k)}. Decoder `D ∈ R^{d×m}`
  with unit-norm columns in the toy.
- Edit problem: target `j`, intended post-activation change `+α`, protected set `P`
  (in nonlinear toys, the features active at `x` other than `j`).
- Required target pre-activation change: `Δp_j^req = (a_j + α) − p_j`. For an active
  target this equals `α`.
- Constraint system: `M = E_{S∪P}`, `t = (Δp_j^req, 0, …, 0)`.

Methods (`src/saeedit/edits.py`):

| name | definition |
|---|---|
| `decoder` | `δ = α D_j` |
| `decoder_rescaled` | `δ = c D_j` with `c = Δp_j^req / (E_j·D_j)` (the strongest one-dimensional rescaling) |
| `encoder_grad` | `δ = Δp_j^req E_j^T / ‖E_j‖²` (least norm for the target row alone) |
| `jacobian_ln` | `δ = M^+ t` (min-norm least squares, dense SVD) |
| `jacobian_ln_cgls` | same target, matrix-free CGLS using only `Mv` (JVP) and `M^T u` (VJP) |
| `jacobian_local_naive` | the a.e. post-activation Jacobian at `x` in place of pre-activation rows |
| `jacobian_ln_repair` | `jacobian_ln`, then up to 3 rounds that add crossed unprotected features as equality constraints (conservative) |
| `jacobian_ln_normmatched` | `jacobian_ln` rescaled to `‖α D_j‖` |
| `trust_region_x{b}` | `argmin ‖Mδ − t‖` s.t. `‖δ‖ ≤ b·‖α D_j‖` |
| `random_normmatched` | random direction with norm `‖α D_j‖` |

Metrics (`evaluate`): `target_rel_err = |Δa_j − α|/|α|`,
`leak_rel_all = ‖Δa_{−j}‖/|α|` (also split into protected and unprotected), the number of
active-set changes (`n_crossings`), the local-prediction error, and
`norm_ratio = ‖δ‖/‖αD_j‖`.

---

## 3. Propositions and proof status

**P1 (linear mismatch).** For σ = identity, `Δa = E δ`, so a decoder edit achieves
`α E D e_j`. It realises the intended change exactly iff `E D e_j = e_j`.
*Proof:* substitute. **PROVED.**

**P2 (what rescaling can do).** `decoder_rescaled` achieves `c α E D e_j`. It hits the
target iff `(ED)_{jj} ≠ 0`. After that, leakage on `P` is
`|α|·‖(ED)_{Pj}‖/|(ED)_{jj}|`, which is zero iff `(ED)_{Pj} = 0`. So rescaling is
sufficient iff column j of `ED`, restricted to `S∪P`, is a multiple of `e_j`.
*Corollary:* for a tied encoder `E = D^T` with an equiangular decoder (pairwise cosine c),
leakage is `c·√(m−1)·|α|` both before and after rescaling, because `(ED)_{jj} = 1`.
**PROVED** (substitute). CHECKED as V3: max abs error 8.9e-16.

**P3 (least-norm solution).** If `M` has full row rank, `δ* = M^T(MM^T)^{-1}t = M^+t`
is the unique minimum-norm solution. Every other solution is `δ* + n` with `n ∈ ker M`,
and `‖δ*+n‖² = ‖δ*‖² + ‖n‖²`. **PROVED** (textbook, e.g. Golub & Van Loan,
*Matrix Computations*, least-squares chapter).
Special cases (all **PROVED** by substitution; CHECKED as V1–V3):
- `E = D^+` with `D` of full column rank and `P` = all features: `δ* = α D e_j`, so the
  decoder edit is already least-norm, because `(D^+)^+ = D`.
- `E = diag(s) D^T` with orthonormal `D`: `δ* = (α/s_j) D e_j`, so `jacobian_ln` equals
  `decoder_rescaled`.
- `E = D^T` with Gram matrix `G = D^T D` (positive definite): `δ* = α D G^{-1} e_j` and
  `‖δ*‖/|α| = √((G^{-1})_{jj}) ≥ 1`. Equality holds iff `G e_j ∝ e_j`. The bound
  follows from Cauchy–Schwarz in the `G`-inner product:
  `1 = (e_j^T e_j)² ≤ (e_j^T G e_j)(e_j^T G^{-1} e_j)` with `G_jj = 1`.

**P4 (feasibility; Fredholm alternative).** `Mδ = t` is solvable iff `t ⊥ ker M^T`. If
`rank M < |S∪P|`, in particular whenever `|S∪P| > d`, the feasible targets form a
proper subspace, so a generic target is infeasible. The least-squares residual
`r = t − MM^+t` is a certificate: `M^T r = 0` and `r·t = ‖r‖² > 0`. **PROVED**
(textbook).
*Duplicate rows:* if rows `j` and `i` of `E` are identical, the targets `(+α, 0)` are
infeasible for every `d`. When `e` and the remaining rows are linearly independent, the
minimum residual is exactly `α/√2`: minimise `(u−α)² + u²` over `u = e·δ`. **PROVED.**
CHECKED in T3: 20/20.

**P5 (ReLU piece exactness).** Along `x + sδ` with `s ∈ [0,1]`, each `p_i(s)` is affine
in `s`. So feature i changes activity somewhere on the segment iff it has different
activity at the two endpoints. If no index (target included) changes activity, then
`Δa = diag(1[p>0]) E δ` exactly. With protected pre-activations held fixed, protected
activations are exactly unchanged. Remaining leakage can only come from inactive
features with `(Eδ)_i > −p_i`, i.e. features that cross their margin. **PROVED.**
CHECKED as V6.
*Corollary:* for an inactive target, row j of the a.e. post-activation Jacobian is zero,
so `jacobian_local_naive` cannot move the target. Observed `target_rel_err = 1` in every
inactive-target cell. **PROVED.**

**P6 (JumpReLU).** The attainable post-activation values are `{0} ∪ (θ, ∞)`, so targets
in `(0, θ]` are infeasible. Crossing the threshold changes the activation by at least θ,
while the a.e. derivative predicts 0. **PROVED** (from the definition). The local
derivative is **not** a global inverse.

**P7 (TopK).** If exactly k features are active (all positive) and an inactive target
enters the top k, at least one other feature must leave. Zero leakage is therefore
impossible for such an edit. **PROVED** (counting). CHECKED in T3: 20/20.
*Caveat:* for TopK, comparing the active sets at the two endpoints only gives a lower
bound on the number of changes along the path. The k-th order statistic of affine
functions can cross an index more than once.

**P8 (norm budget).** `min ‖Mδ−t‖ s.t. ‖δ‖ ≤ τ` is convex. The solution is `M^+t` if
`‖M^+t‖ ≤ τ`. Otherwise it is `(M^TM+λI)^{-1}M^Tt` with the unique `λ > 0` such that
`‖δ(λ)‖ = τ`. **PROVED** (KKT for a convex trust-region subproblem; convexity rules out
the "hard case"). CHECKED as V7: 4320 rows, 0 budget violations.

**P9 (CGLS).** Started from 0, the CGLS iterates lie in `range(M^T)`. In exact
arithmetic they reach the minimum-norm least-squares solution in at most
`rank(M)` steps. **PROVED** (standard Krylov argument). The floating-point behaviour is
only CHECKED: max relative difference to dense 7.5e-13 (V5).

**Unproved or open:**
- **OBSERVED_IN_TOY, no general statement:** protecting a subset `P` can *increase*
  total leakage by moving it onto unprotected coordinates (R4b, R5 below). There is
  no general theorem here about when this happens.
- **UNPROVED:** any statement about language models, including R1–R3.

---

## 4. First-run toy results (seeds 1000–1019; 20 instances per cell)

The independent unit is a toy instance, i.e. one draw of (D, E, b). Numbers are medians,
across instances, of within-instance means. All of this is a synthetic illustration.
Most linear outcomes are fixed by P1–P4.

### 4.1 Linear regimes: steps 2, 3 and 5 (`P3-T2-MISMATCH`, `P3-T5-RESCALE-VS-INTERF`)

| regime | class (holds in all 20 instances) | decoder tgt err / leak | rescaled leak (norm) | jacobian_ln leak (norm) | residual |
|---|---|---|---|---|---|
| R0 `E=D^+`, m=8<d=16 | NO_MISMATCH | ~1e-16 / ~1e-15 | ~1e-15 (1.00) | ~1e-15 (1.00) | ~1e-15 |
| R1 orthonormal D, `E=diag(s)D^T` | RESCALE_SUFFICIENT | 0.365 / ~1e-15 | ~1e-15 (1.13) | = rescaled | ~1e-16 |
| R2 tied, equiangular c=0.3 | INTERFERENCE_REMOVED_BY_CORRECTION | ~0 / 0.794 = 0.3√7 | 0.794 (1.00) | ~1e-15 (1.14 = √G⁻¹_jj) | ~1e-16 |
| R3 scaled tied, c=0.3 | INTERFERENCE_REMOVED_BY_CORRECTION | 0.340 / 0.884 | 0.989 (1.09) | ~1e-15 (1.24) | ~1e-15 |
| R4a tied, m=48>d=16, protect all | INFEASIBLE_ON_S_UNION_P | ~0 / 1.73 | 1.73 (1.00) | 0.466; **target err 0.675** (0.40) | 0.821 |
| R4b tied, m=48, protect 7 | INTERFERENCE_MOVED_TO_UNPROTECTED | ~0 / 1.73 (P: 0.638, U: 1.59) | 1.73 | **2.16** (P: ~0, U: 2.16) (1.35) | ~1e-15 |
| R5 `E=D^+`, m=48, protect 7 | INTERFERENCE_MOVED_TO_UNPROTECTED | 0.669 / 0.467 | 1.44 (3.10) | **2.21** (P: ~0) (3.60) | ~1e-15 |

Notes:
- `encoder_grad` leaks 0.283 in R0, where the decoder edit is exact. Steering along the
  encoder row is not "encoder-consistent" once other features are counted.
- At matched norm in R2, `trust_region_normmatched` gives target error 0.111 and leakage
  0.031: it trades a little target for most of the leakage.
- Infeasible systems (R4a) are solved in the least-squares sense, which gives up on the
  target (0.675 error). Any real use needs an explicit priority or weighting between
  the target and protected rows. None was pre-registered here.
- `mismatch_diag_share` in the raw CSV is numerical noise wherever the total mismatch is
  ≤ 1e-8. Use `P3-T5-RESCALE-VS-INTERF.decoder_diag_share_median`, which is `None` in
  those cases.

### 4.2 ReLU / JumpReLU / TopK active sets and the norm budget (step 4; `P3-T4-ACTIVESET-NORM`, pre-registered)

Setting: d=16, m=64, interpolated encoder. The median active count is 13 for ReLU
(range 5–22), 11 for JumpReLU (3–23), and 6 for TopK. This is dense: `|S∪P|/d ≈ 0.4–1.4`.
- Paired, per instance: `jacobian_ln` has *higher* total leakage than `decoder_rescaled`
  in almost every cell. Both hit the target when feasible. For ReLU active targets,
  `jacobian_ln` has lower leakage in 10/20 instances at α=0.25, 3/20 at α=1 and 0/20 at
  α=4; for inactive targets, 0/20 at every α. The cause is threshold crossings of
  unprotected features: median 13–23 crossings for `jacobian_ln` against 9.5–26 for
  rescaled. `jacobian_ln` also has a larger norm: median norm ratio 3.0–7.6, against
  1.5–4.1 for rescaled.
- `jacobian_ln_repair` reaches a crossing-free edit in **72–80 of 80 rows** per ReLU or
  JumpReLU cell, but it does so by giving up the target (median target error 0.23–0.84,
  leaving out the range-infeasible JumpReLU cell). For TopK it reaches one in only
  **2–66 of 80 rows** per cell, because eviction is rank-based and equality constraints
  cannot prevent it (P7).
  - *Correction (second session):* the first-session text said "100% of ReLU/JumpReLU
    cells" and "0–88%". Those were medians of per-instance rates, not row counts. They
    were re-aggregated from the unchanged raw CSV by `src/saeedit/paper_assets.py`.
- `jacobian_local_naive` never moves an inactive target (target error 1.0; P5
  corollary).
- JumpReLU, inactive target, α=0.25 < θ=0.3: the target is range-infeasible in all
  instances (P6).
- Norm budget: `trust_region_normmatched` has a lower total error than `decoder` at equal
  norm in 15–20/20 instances for ReLU/JumpReLU (except ReLU inactive α=4: 10/20). This
  comparison is **circular**: the trust region minimises exactly the internal metric
  being scored.

- **Re-aggregated with the drift decomposition** (`paper/tables/tab_activeset_relu.tex`).
  With P = the active set, the toy's protected and unprotected leakage are exactly
  `D_orig` and `D_new` (unit test `TestDrift`).
  - At ReLU, active target, α=1, the rescaled decoder has `D_orig` 0.786 and `D_new` 0.51.
  - `jacobian_ln` has `D_orig` 0.051, which is nonzero only because some rows are
    infeasible, and `D_new` **1.73**.
  - So the correction moves drift onto newly activated features. This is the same pattern
    as linear R4b/R5.

### 4.3 Exploratory sparsity axis (`P3-T4b-SPARSITY-EXPLORATORY`; added after dev inspection, not pre-registered)

Only the ReLU bias was changed.
- Bias −0.75 (median 1 active, range 0–6): `jacobian_ln` has lower leakage than
  `decoder_rescaled` in 15–19/20 instances in every (target kind, α) cell. The median
  paired differences are small, from −0.04 to −0.29 (in units of α).
- Bias −0.5 (median 4 active, range 0–11): lower in 19–20/20 at α ≤ 1 for active targets,
  but 4/20 at α=4. For inactive targets: 16/20, 11/20 and 3/20.
- With repair, at bias −0.75 and α=0.25 for active targets, the target error and the
  leakage are both about 1e-16.
- **Reading:** in this toy, the correction beats rescaling only when the protected set is
  small relative to d *and* the edit is small enough to avoid crossings. That is a
  hypothesis about real SAEs (`P3-REAL-01`), not a finding about them.

### 4.4 Synthetic external proxy (`P3-TX-EXTPROXY`; V8)

A ground-truth dictionary `A` defines the external readout `z = A^+ x`. The SAE is a noisy
estimate of it.

| case | internal err: decoder → jacobian_ln | external err: decoder → jacobian_ln | ln better externally |
|---|---|---|---|
| encoder accurate (`E = A^+`) | 0.221 → ~0 | 0.221 → ~0 | 20/20 |
| decoder accurate (`D = A`) | 0.242 → ~0 | ~0 → **0.239** | **0/20** |
| both noisy | 0.313 → ~0 | 0.223 → 0.223 (paired median diff −0.0014) | 11/20 |

In every case the internal metric becomes perfect, but that says nothing about the
external effect. This is a constructed counterexample, **PROVED by construction**, for
"internal consistency implies external correctness". It motivates the stop rule. It
says nothing about which case real SAEs are in.

### 4.5 Compute

- Same-layer edits: the Jacobian of the pre-activations with respect to x is just the
  gathered rows `E_{S∪P}` (`|S∪P| × d`), and materialising it is cheap. JVP/VJP (CGLS)
  matter only when the target is read at a later layer or at the output.
- CGLS cost in the toy: the median number of JVPs per edit is 1–16 depending on regime
  (max 17), each with one extra VJP. Wall time for the whole suite: 76–77 s on one
  thread. Per-experiment times are in `run_manifest.json`.

---

## 5. Protocols for the real evaluation (specified, NOT_RUN)

### 5.1 Current: `P3-REAL-01R` (fidelity) — `configs/p3_real01r.json`

- **Units and splits.** The unit is the **test document**. WikiText-103 validation
  articles are calibration and test articles are test. Documents are split *before*
  windows (BOS + 127 tokens) are made. An overlap check aborts the run if a title or a
  window appears in both splits. The document count is not verified; below 40 the run is
  BLOCKED.
- **Calibration only, frozen with sha256.**
  - Feature pool: firing density in [1e-4, 1e-2]. 64 features are sampled with seed 0.
  - Target-change doses: activation quantiles q50/q90/q99 and 2×q99. `a_max` is recorded
    but not used.
  - Norm budgets: {0.025, 0.05, 0.1, 0.2} × the median centred residual norm.
  - The timing smoke uses calibration documents only and chooses N ∈ {8, 16, 32} by
    projected budget.
- **Methods.**
  - no_edit;
  - decoder;
  - weight-only calibration-rescaled decoder (unmatched, deployable);
  - encoder_grad;
  - `jacobian_ln` (P = active set);
  - random.
- **Modes.** Equal-norm and target-matched. For ReLU the target-matched scale has a closed
  form, and rows without a solution are recorded as `INFEASIBLE` rather than dropped.
- **Metrics.**
  - target change and error;
  - `D_kept`, `D_deact`, `D_orig`, `D_new`, `D_all` (relative);
  - counts;
  - edit norm and the mean(δ) component;
  - KL(P_clean‖P_edit) and ΔNLL over 16 positions;
  - solve and forward seconds;
  - peak RSS.
- **Uncertainty.** Document-cluster bootstrap (2000 resamples).
- **Gates.** The G1–G3 text is preserved verbatim. Each is read at q99 as an interval
  with the outcome STOP_SIDE, CONTINUE_SIDE or INCONCLUSIVE. These are operational rules,
  not significance tests. A large crossing share ends the *method claim*, not the
  measurement study.
- **Boundary conventions.**
  - JumpReLU (not used by this SAE): `a = p·1[p>θ]`, strict.
  - TopK: stable sort by (−p, index); positive values only; displacement counted.
- **SAE-TS / FGAA.** NOT_RUN baselines: they need a fitted effect model built from many
  steering runs.

### 5.2 Current: `P3-REAL-02` (behaviour) — `configs/p3_real02.json`

- **Task.** Benign wedding-topic steering. The label is ActAdd's keyword list (verified in
  the source text), counted only in the continuation. The SAE is never used for the label.
- **Chosen on calibration data only:**
  - the feature, as the largest activation difference between windows with and without
    keyword hits (BLOCKED if fewer than 20 hit windows);
  - the DiffMean vector;
  - each method's budget, as the largest budget with KL ≤ 0.1 nats.
- **Test.** Neutral prompts (zero hits), greedy decoding of 32 tokens, the edit at all
  positions. Success is paired by document, with bootstrap intervals.
- **Support rule.** The method is supported only if `jacobian_ln` beats decoder,
  encoder_grad and DiffMean at matched calibration KL with intervals excluding 0.

### 5.3 Superseded first-session protocol (kept for the record)

**Units and splits.** The independent unit for behaviour is a (feature/concept,
prompt-template) pair. Split *features* into dev and test, and split prompt templates
into dev and test. Tune α grids and any priority weights on dev features and prompts
only. Report test once. Paraphrases of one prompt count as a single unit.

**Comparators.** These are decided in advance:
- `decoder` with a dev-tuned coefficient;
- `decoder_rescaled`;
- `encoder_grad`;
- `jacobian_ln` and `trust_region` at matched norm;
- `random_normmatched`;
- the simplest non-SAE alternative, a difference-in-means (DiffMean) steering vector
  for the same concept, computed on dev prompts;
- a global pseudo-inverse of a fitted effect map, plus SAE-TS where feasible (prior
  work, `RELATED_WORK.md` #1).

Predictors to control for in R2: decoder crowding and max cosine (Duan 2026;
Khan 2026), and edit norm. Feature-selection confound (Arad et al. 2025): report results
stratified by output score, or fix the feature set before looking at outcomes.

**Metrics.**
- *External:* concept success on test prompts, scored by a scorer that does **not**
  use the SAE (a fixed, versioned classifier or judge, recorded in the manifest).
- *Quality:* next-token KL, or the perplexity change, on neutral held-out text.
- *Internal (secondary only):* SAE re-encoded target error and leakage.
- *Cost:* JVP/VJP count and wall clock.
- Report the norm–quality curve.

**Primary test.** Paired by unit: external success of `jacobian_ln` minus that of
`decoder_rescaled` at matched norm and at matched quality loss, with a pre-registered
margin.

**Stop.** As in §1.

---

## 6. Next decision experiment (proposed, needs approval)

**`P3-REAL-01R`** runs the stages `contract → calibrate → smoke → run → summarize` from
`configs/p3_real01r.json`. Its outcome decides whether P3-REAL-02 is run as a method test,
so it can only STOP or PROCEED; it cannot support the method. It is also the measurement
study on its own terms. The superseded P3-REAL-01 design is in `STATUS.md` §6 of the first
session (git history, commit 7c46116).

---

## 7. Coordinate validity: `P3-CONTRACT-EXEC` (third session, executed)

### 7.1 Setup
- **Configs.** `configs/p3_contract_exec.json` (v1, float32, pre-registered) and
  `configs/p3_contract_exec_v2.json` (v2, float64 diagnostic).
- **What is compared.** The canonical path, TransformerLens 2.18.0 `HookedTransformer`
  with explicit options (fold_ln, center_writing_weights, center_unembed and
  fold_value_biases all on; refactor off; BOS prepended) plus SAELens 6.51.3, against the
  HF shortcut used by the adapter.
- **Pinned inputs.** Model and SAE revisions, layer 8 and hook site are unchanged. A pinned
  tokenizer and model object are passed into TransformerLens, so library defaults cannot
  replace the old contract.

### 7.2 Proposition (admissible subspace), **PROVED** (elementary)

In GPT-2, every read of the residual stream at or after layer 8 goes through a LayerNorm
that subtracts the mean over d_model: ln_1, ln_2 of later blocks, and ln_f. So adding
`c·1` at the layer-8 residual leaves all logits unchanged.

*Proof:* `LN(h + c1) = LN(h)`. Later blocks add to the residual without reading the `1`
component, and ln_f removes it at the end.

Consequences:
- The admissible perturbations are `range(P)`, with `P = I − 11ᵀ/d`.
- Split any edit as `δ = Pδ + (1ᵀδ/d)·1`. The second term changes SAE readouts
  (through `W_encᵀ1`) but not the model.
- The SAE's coordinates are `x = P h`, i.e. `center_writing_weights` gives exactly the
  centred HF state (checked: K2 below).
- For `δ ∈ range(P)`, `P(h + δ) = x + δ`, so the TransformerLens convention and the HF
  shortcut coincide. Outside `range(P)` they differ by the mean component.
- The LN correction is therefore posed as `min ‖δ‖ s.t. (M P) δ = t`, i.e. with the
  encoder Jacobian `J_E P`. Its solution lies in `range(P M^T) ⊆ range(P)`.

### 7.3 Results (`results/contract_exec/`, `results/contract_exec_v2/`)

| Check | float32 (v1, pre-registered) | float64 (v2, same tolerances) |
|---|---|---|
| K0 weights and config (SAELens vs pinned file) | bitwise equal; cfg ok | same |
| K1 tokenizer | 16/16 | 16/16 |
| K2 activation, `x = h − mean(h)` | 4.2e-6 | 1.0e-14 |
| K2, uncentred `h` (negative control) | 3.2e-2 | 3.2e-2 |
| K3 SAE features | 7.7e-6 | 1.6e-14 |
| K4 L0 / FVE (canonical) | 72.4 / 0.827 | same |
| K4, uncentred input | L0 176 / FVE 0.003 | same |
| K5 no-op hooks | 0 | 0 |
| K6 common logit shift (center_unembed) | max 279 | 279 |
| K6 residual after shift | 4.3e-4 | 7.4e-13 |
| K6 max \|Δp\| | **2.8e-5 (> 1e-5: FAIL)** | 5.1e-14 |
| K6 KL | 7.5e-7 | 1.5e-15 |
| K7 post-edit feature change | 7.6e-5 | 1.9e-13 |
| K7 max \|Δp\| | **2.8e-5 (FAIL)** | 5.3e-14 |
| K7 KL | 7.7e-7 | 1.5e-15 |
| K9 mean-only KL | **5.1e-7 / 6.0e-7 (> 1e-8: FAIL)** | 7.8e-16 / 1.3e-15 |
| K9 SAE drift/ρ, median | 125 | 125 |
| K10 no-processing path | not run | KL 1.9e-15, residual 7.8e-15 |
| **Verdict** | **CONTRACT_FAIL_BLOCKED** (kept) | **CONTRACT_PASS_FLOAT64** |

Reading:
- The mapping is exact: the float64 differences are rounding-level.
- The float32 failure comes from rounding on top of a common logit shift of up to 279.
- Any float32 KL below about 1e-6 is therefore at the numerical noise floor.
- The real-model pilot runs in float64.

### 7.4 Diagnostic findings (descriptive)
- **K8.** Raw decoder rows have a mean component of 3–6% of their norm. Even so, the SAE
  readout differs between the two conventions by 51–114% (median 79%) of the edit's own
  readout change. For non-admissible edits, the implementation convention changes the
  internal measurement.
- **K9.** A mean-only edit with ρ = 10.1 (10% of the median residual norm) leaves the model
  invariant, yet moves the SAE by a drift/ρ of 125 (median), with a single feature
  changing by up to 37.5.
  - The encoder is unconstrained along a direction the training data never occupies.
  - This is a real-model instance of internal change with no external effect. It is used
    as a diagnostic control in the pilot and is never counted as a method.
