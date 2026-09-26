# RELATED_WORK — P3: Do SAE Edits Change What They Claim?

Checked on 2026-09-26 with WebSearch/WebFetch only. Nothing was downloaded, installed or
uploaded. The full working notes (66 fetched URLs, including failures) were kept in the
session scratchpad. This file keeps only what matters for the claims.

## Access levels (read this first)

WebFetch returns an extraction model's answer about a page, not the raw page. So nobody
here read the raw text of these papers.

| label | meaning |
|---|---|
| `FT-X` | The full HTML was fetched, and the method and experiment sections were queried through the extraction model. Not a human full-text read. |
| `PARTIAL-X` | As `FT-X`, but only the named sections. |
| `ABSTRACT_ONLY` | Only the abstract or landing page was seen. |
| `SPOT-CHECKED` | The main agent re-fetched the page and confirmed the quoted claim. |
| `UNVERIFIED` | Stated in a search snippet or inferred; not confirmed from a source. |

| `SOURCE-TEXT` | (second session) The arXiv HTML was fetched with curl and converted to text, and the equations and experiment sections were read in that text. For the 10 papers in §0 this supersedes `FT-X`/`PARTIAL-X`. |

"(derivation)" marks our own linear algebra, not a claim made by a paper.

## 0. Source-text verification of the closest papers (second session, 2026-09-26)

All ten were read as `SOURCE-TEXT`. The HTML version equals the latest version listed on
each abs page.

| Paper | Key equations / settings as written | Still not found |
|---|---|---|
| SAE-TS, arXiv:2411.02193v2 | Eq. 2: `y = E_steered[f(x)] − E_unsteered[f(x)]` over SAE activations of rollouts passed "back through the model up to layer l". Effect model `ŷ = xM + b` (M: d_model × d_sae), learned from 50k decoder vectors of a 65k SAE with α set so loss rises by 0.5. Steering vector `s = M_j/‖M_j‖ − λ Mb/‖Mb‖`, λ=1, then unit-normalised. App. A.1 / Table 3: the pseudoinverse of **M** (the learned effect map, not the SAE encoder) scores below SAE-TS on **all 9** tasks (e.g. Wedding 0.5432 vs 0.0567). Evaluation: gpt-4o-mini judge (behavioural × coherence), prompt "<BOS>I think", 256 × 32-token completions. | Whether the steering vector is present during the re-pass; the target vector used in the pseudoinverse solve. |
| Cui, Shen, Yang, arXiv:2606.18322v1 | Constraints C1 (encoder orthogonality `A^T δ = 0`), C2 (activation stability), C3 (decode stability), C4 (budget). Only C1 (projection) and C4 (clipping) are enforced; C2/C3 are checked afterwards. Within a layer: `P = I − A(A^T A)^† A^T`. Across layers: `P = I − J_t^T (J_t J_t^T)^† J_t`, where J_t is the Jacobian of the defended-feature map. Refusal recovery 95.8% at drift 0.131. | Exact model names for some tasks ("Gemma-scale model"). |
| GLP, arXiv:2602.06964v1 §4.1 | Llama-3.1-8B-Base + LlamaScope, 500 random directions, judge 0–2 on concept and fluency. Quote: "feature descriptions are derived from the SAE encoder, while concept directions for steering are derived from the SAE decoder". Conclusion: "off-manifold artifacts, not just encoder-decoder misalignment, contribute to SAE steering failures". | SAE layer and width; numbers beyond the figure. |
| Duan, arXiv:2606.08365v1 | `h' = h + α d_f`, α = 1.0, at the final token. Collateral is counted on a **downstream** SAE (e.g. Gemma layer 12 → 16) and normalised by the logit-effect norm. | Any same-layer measurement of the target's own activation. |
| Khan et al., arXiv:2609.22782v1 | α* = the smallest α on {0, 0.25, 0.5, 1, 2, 3, 5, 10, 20} that raises a keyword-based refusal rate by ≥ 0.10. Predictors: neighbour density, max cosine, co-activation. Off-target effects are deferred to future work. | — |
| AxBench, arXiv:2501.17148v3 | SAE steering `h + α w_dec` with a unit-norm direction; α = steering factor × Neuronpedia max activation. Table 2 means: DiffMean 0.239, SAE 0.165. The body text's win rates (88.0% / 61.6%) disagree with Table 3 (81.8% / 58.7%); cite the table. | — |
| Arad et al., arXiv:2505.20063v2 | Encode, add s·a_max to feature i, decode `W_dec ã + b_dec`. Input and output scores are Eqs. 8–10; Gen Success@k (Eq. 7) is a keyword-style label. | Whether the error term is added back. |
| ReFT, arXiv:2404.03592v3 | DII Eq. 1: `b + R^T(Rs − Rb)`. LoReFT Eq. 2: `h + R^T(Wh + b − Rh)`, R with orthonormal rows. | — |
| AlphaEdit, arXiv:2410.02355v4 | `P = Û Û^T` with an eigenvalue cut at 1e-2; closed form Eq. 14. | App. B.5 heading "+αI" vs Eq. 14 "+I" (not reconciled). |
| ActAdd, arXiv:2308.10248v5 | Sec. 4.1.3 and App. H: wedding steering is scored by the "average number of wedding words per completion" with the keywords wedding, weddings, wed, marry, married, marriage, bride, groom, honeymoon. This is the label used in P3-REAL-02. | — |

**Bibliography.** `paper/references.bib` has 33 entries.
- 18 are venue-verified from proceedings pages (PMLR, NeurIPS, ACL Anthology, OpenReview).
- 10 are arXiv BibTeX copied verbatim; their year is that of the latest arXiv version.
- One (Scaling Monosemanticity) is the site-provided citation.
- Four are numerics references checked through Crossref/DOI or library-catalogue records.

Not venue-verified, so cited as arXiv: SAE-TS, Cui et al., Duan, Chaudhary & Geiger,
ActAdd, Bhalla et al., Mayne et al., O'Brien et al., FGAA and Khan et al.

## 1. Closest prior work: what is already known

| # | Work | Access | What it establishes | Consequence for P3 |
|---|---|---|---|---|
| 1 | Chalnev, Siu, Conmy. *Improving Steering Vectors by Targeting SAE Features* (SAE-TS), arXiv:2411.02193 | FT-X, **SPOT-CHECKED** | Fits a global linear effect map `ŷ = xM + b` from 50k steering vectors (Gemma-2-2B, Gemma Scope L12). Steers with `s = M_j/‖M_j‖ − Mb/‖Mb‖`. Quote: SAE-TS "significantly outperforms the pseudoinverse approach" (App. A.1, Table 3). Shows qualitatively that decoder steering often mainly moves *other* features. Effects are measured on generated text re-encoded at layer ℓ; whether the vector is absent in that second pass is `UNVERIFIED` (not found in the spot check). | Targeting SAE features while suppressing side effects is **published**, and a pseudo-inverse was tried and lost. P3 must include a pseudo-inverse or least-norm baseline and cannot claim the idea. |
| 2 | Soo et al. *FGAA*, arXiv:2501.09929 | FT-X | Multi-feature targets through the SAE-TS effect map; beats CAA, SAE and SAE-TS on 8 of 9 tasks (2B). No fidelity check. | Strong external baseline for P3-REAL-02. |
| 3 | Luo, Feng, Darrell, Radford, Steinhardt. *Learning a Generative Meta-Model of LLM Activations* (GLP), arXiv:2602.06964 | PARTIAL-X (SAE-steering section), **SPOT-CHECKED** | Quote: "feature descriptions are derived from the SAE encoder, while concept directions for steering are derived from the SAE decoder". Quote: GLP "suggest[s] that off-manifold artifacts, not just encoder-decoder misalignment, contribute to SAE steering failures." Setup: Llama-3.1-8B-Base with LlamaScope. | Encoder–decoder misalignment as a *cause* of steering failure has **already been stated**, together with a competing cause (off-manifold effects). P3 can only add a quantification and a test of how much it explains. |
| 4 | Cui, Shen, Yang. *SAE Interventions are Unreliable*, arXiv:2606.18322 | PARTIAL-X, **SPOT-CHECKED** | Uses `P = I − Jᵀ(JJᵀ)†J`, where `J` is the Jacobian of the defended SAE features, so perturbations keep SAE readouts fixed. These perturbations recover suppressed behaviour: 95.8% refusal recovery on AdvBench, 100% on IOI (Gemma Scope, GPT-2 small). | (a) The **same projector** as P3's protected-set correction is published. (b) **Holding SAE readouts fixed does not pin down behaviour.** This is direct prior evidence against the premise that encoder consistency implies behavioural control (compare toy TX). |
| 5 | Duan. *Pre-Intervention Prediction of SAE Steering Side Effects*, arXiv:2606.08365 | FT-X (method and metrics) | Predicts "collateral spread" in a **downstream-layer** SAE from decoder crowding, activation statistics, co-activation and logit footprint. Covers ReLU, JumpReLU and TopK SAEs on 4 models. Does not measure the target's own re-encoded change. Proposes no correction. | P3-R2 must show that same-layer re-encoded mismatch predicts failure **beyond** these decoder-geometry predictors. |
| 6 | Khan et al. *Look Before You Steer*, arXiv:2609.22782 | PARTIAL-X | Decoder neighbour density and max cosine predict the smallest effective coefficient. | Same as 5: a geometry-only predictor to control for. |
| 7 | Karvonen et al. *SAEBench*, arXiv:2503.09532 (ICML 2025) | FT-X | Eight metrics (loss recovered, AutoInterp, sparse probing, absorption, RAVEL, SCR, TPP, unlearning). None re-encodes an edited activation or measures intended vs achieved latent change, and there is no steering metric. | A same-layer "edit fidelity" metric would be new *relative to SAEBench*. Low bar; not a contribution by itself. |
| 8 | Farnik et al. *Jacobian SAEs*, arXiv:2502.18147 (ICML 2025) | FT-X | Sparsifies input-latent → output-latent Jacobians through an MLP. Not about edits. (derivation) With identity in place of the MLP, the object reduces to `E_active D_active`. | Tooling precedent only. |
| 9 | Arad, Mueller, Belinkov. *SAEs Are Good for Steering – If You Select the Right Features*, arXiv:2505.20063 (EMNLP 2025) | PARTIAL-X | Input scores and output scores rarely coincide. Filtering by output score gives 2–3× better steering. | Behavioural evidence that detection and causation come apart. Feature selection is a confound P3 must control. |
| 10 | Wu et al. *AxBench*, arXiv:2501.17148 (ICML 2025) | FT-X | SAE decoder steering (mean score 0.165) loses to DiffMean (0.239) and to prompting, LoReFT and SFT. No encoder–decoder analysis. | DiffMean is the simplest-alternative baseline P3 must include. |
| 11 | Durmus et al. *Evaluating feature steering* (Anthropic, 2024) | PARTIAL-X (post; appendix unreadable) | Off-target behavioural effects, e.g. a gender-bias feature raising age bias by 13%. | Behavioural precedent for leakage. |
| 12 | Templeton et al. *Scaling Monosemanticity* (2024) | PARTIAL-X (App. D.2 not reached) | "Clamp" features during the forward pass. That the error term is kept is `UNVERIFIED`. (derivation) Clamping inside the reconstruction with the error kept equals adding `(c − f_j(x)) D_j` to `x`, i.e. a decoder edit. | Clamping inherits the same mismatch. |

## 2. Encoder ≠ decoder is documented

| Work | Access | Point |
|---|---|---|
| Chanin et al. *A is for Absorption*, arXiv:2409.14507 | FT-X | Toy and LM evidence: the encoder has recall holes that the decoder lacks. |
| O'Neill, Gumran, Klindt. *Amortisation Gap*, arXiv:2411.13117 | FT-X | Theorem 3.1: a linear-nonlinear encoder has a non-zero amortisation gap in the stated regime. |
| Karvonen, blog (2024) | FT-X | Median encoder–decoder cosine ≈ 0.5 (one SAE family; `UNVERIFIED` for others). |
| Mayne, Yang, Mahdi, arXiv:2411.08790 | FT-X | `f(v)` of a bare steering vector is dominated by the encoder bias. Suggests a difference of encodings instead. |

**Consequence:** "`ED ≠ I`" is not a finding. At most, P3 can measure how large the mismatch
is and how it splits into diagonal, off-diagonal and active-set parts in public SAEs.

## 3. The linear core of the correction is standard

(derivation) With readout rows `A = E_{S∪P}`, the least-norm edit is `A⁺t`. With `P = ∅` in
the linear regime, this is **steering along the encoder row**, `Δp_j E_jᵀ/‖E_j‖²`
(`encoder_grad`, a mandatory baseline). Adding `P` projects onto `ker A_P`.

| Work | Access | Relation |
|---|---|---|
| Geiger et al. *DAS*, arXiv:2303.02536 (CLeaR 2024) | PARTIAL-X | With orthonormal `R`, the interchange `h + Rᵀ(Rs − Rh)` is the least-norm edit that sets the readouts `Rh` (derivation). |
| Wu et al. *ReFT/LoReFT*, arXiv:2404.03592 (NeurIPS 2024) | FT-X | `h + Rᵀ(Wh + b − Rh)`: the same form with learned orthonormal `R`. P3 ≈ LoReFT with `R` replaced by non-orthogonal SAE encoder rows, `Rᵀ` by `A⁺`, and a fixed target. |
| Fang et al. *AlphaEdit*, arXiv:2410.02355 (ICLR 2025) | FT-X | Null-space projection protects preserved outputs (in weight space). |
| Sheng et al. *AlphaSteer*, arXiv:2506.07022 | ABSTRACT_ONLY | Null-space-constrained activation steering. |
| Belrose et al. *LEACE*, arXiv:2306.03819 (NeurIPS 2023) | FT-X | Least-change edit under a readout constraint (for erasure). |
| Singh et al. *MiMiC*, arXiv:2402.09631 (ICML 2024) | FT-X | Least-squares-optimal affine steering; distribution-level. |
| Park, Choe, Veitch, arXiv:2311.03658 (ICML 2024) | FT-X | Readout and intervention directions are related by a metric (`Cov⁻¹`). The Euclidean norm used for P3's least-norm edit is a choice, not a given. |
| Makelov, Lange, Nanda, arXiv:2405.08366 | PARTIAL-X | Least-squares fits of decoder-space edits to counterfactual activations. |
| Bhalla et al., arXiv:2411.04430 | PARTIAL-X | Encoder–decoder intervention framework; success measured on outputs. |

## 4. External evaluation standards

- RAVEL (Huang et al., arXiv:2402.17700, ACL 2024; FT-X): Cause and Isolation. These are
  the standard *external* measures of whether an edit did what it claims and left the
  rest alone.
- Chaudhary & Geiger, arXiv:2409.04478 (FT-X): SAEs underperform DAS on RAVEL-style
  disentanglement in GPT-2 small.

## 5. Candidate public model/SAE resources (nothing downloaded; licenses as displayed on the cards)

| Resource | License on card | Options | Caveats |
|---|---|---|---|
| GPT-2 small + `jbloom/GPT2-Small-SAEs-Reformatted` (SAELens `gpt2-small-res-jb`) | MIT (SAEs); GPT-2 weights: MIT (`UNVERIFIED` here) | `blocks.{0..11}.hook_resid_pre` + `blocks.11.hook_resid_post`, d_sae 24576, d_model 768 | Model card is empty; the ReLU architecture is `UNVERIFIED` on the card and must be checked from the loaded config. CPU-feasible. |
| Gemma-2-2B + Gemma Scope `google/gemma-scope-2b-pt-res` | cc-by-4.0 (SAEs); Gemma Terms of Use apply to the base model | Layers 0–25; 16k/65k widths on all layers; `canonical` IDs with L0 ≈ 100 | JumpReLU: local derivative ≠ global inverse (P6). Larger. |
| Llama-3.1-8B + Llama Scope `fnlp/Llama-Scope` | apache-2.0 (card); Meta license for the base model (`UNVERIFIED` terms) | TopK; 32k/128k | TopK eviction (P7). GPU-scale. |
| SAELens | MIT | loader | — |

## 6. Novelty boundary (what P3 may and may not claim)

**Must not claim:**
- that decoder steering leaks;
- that the encoder differs from the decoder;
- that misalignment causes steering failure (stated in GLP);
- a least-norm or null-space-protected correction as a new method (DAS/LoReFT form,
  AlphaEdit/AlphaSteer, Cui et al.'s exact projector);
- JVP/CG solves (routine numerics).

**Remaining, testable gaps (bounded search, so absence is weak evidence):**
1. A same-layer fidelity decomposition of decoder edits in public SAEs: diagonal (rescalable)
   vs off-diagonal on the active set vs threshold/active-set flips, across SAE
   activation types. → `P3-REAL-01`
2. Whether that re-encoded mismatch predicts external steering failure **beyond**
   decoder-geometry predictors (Duan 2026, Khan 2026) and edit norm. → `P3-REAL-02`
3. Whether an active-set-protected local correction beats rescaling, encoder-row steering,
   DiffMean, SAE-TS/FGAA and a pseudo-inverse **on held-out behaviour at matched
   quality**. The prior evidence points the other way (SAE-TS pseudo-inverse result;
   Cui et al.; toy TX). The expected outcome is negative, and it should be reported as
   such if it happens.
