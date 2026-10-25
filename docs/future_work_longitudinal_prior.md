# Future Work: Longitudinal Patient Priors for Underdetermined Fault Disambiguation

## Context & Clinical Rationale
In multi-field laboratory extraction errors ($K \ge 2$), algebraic conservation laws (e.g., Friedewald, Sampson) can become mathematically underdetermined when only a single conservation identity is available. While BioTrace addresses this using two legitimate contemporary independent-evidence sources:
1. **Extra-Field Equation Augmentation:** Integrating reported subfractions ($\text{Non-HDL} = \text{TC} - \text{HDL}$, $\text{VLDL} = \text{TG} / 5$) to render the linear system exactly determined or overdetermined.
2. **Independent Dual Blind Re-Read:** Running the vision gate (GrabCut + Real-ESRGAN 4x) independently on both suspect crops and requiring consensus agreement with a cost-ranked candidate pair.

A complementary clinical mechanism for fault localization in the medical literature is the **longitudinal historical prior** (delta-check against prior patient lab results).

## Deferred Mechanism: Bayesian Delta-Checks & Patient Trajectories
- **Concept:** Under steady clinical state, serum lipid parameters exhibit biological autocorrelation across longitudinal encounters (e.g., baseline HDL and LDL rarely shift by $> 40\%$ within 30 days in the absence of statin titration or acute illness).
- **Proposed Architecture:** Formulate a Bayesian prior $P(\theta \mid \text{History})$ parameterized by Gaussian process or autoregressive state estimation over historical lab values, using Mahalanobis distance to weight competing candidate inversion pairs.

## Grounded Exclusion Rationale
This longitudinal prior mechanism is **deliberately deferred** from active implementation based on data provenance constraints:
- **Requirement for Real Longitudinal Trajectories:** Bayesian delta-checks require authentic, multi-encounter patient longitudinal time series (with realistic clinical intervals, treatment changes, and biological intra-individual variance).
- **Synthetic Contamination Risk:** Seeding patient histories with synthetic noise or simulated trajectories risks fabricating artificial priors that overestimate disambiguation performance while misrepresenting true clinical delta distributions.
- **Data Availability Status:** CDC Continuous NHANES is a cross-sectional examination survey, providing single-point measurements without repeated longitudinal patient encounters. Real multi-encounter longitudinal EHR datasets (e.g., MIMIC-IV or local health system cohorts) with verified multi-field extraction corruptions are not currently staged in this repository.

## Current System Behavior
When neither extra-field equations nor dual blind re-reads are available, BioTrace deterministically emits `ABSTAIN_ALGEBRAIC_INDETERMINATE`. This is the medically safe and correct behavior under clinical non-maleficence, strictly avoiding ungrounded speculative edits.
