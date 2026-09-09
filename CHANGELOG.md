# Changelog

## 0.8.0 - 2026-09-08

- Replace the TWFE confidence-ball upper decision with covariance-aware
  projected-Wald noncentral-chi-square inversion. An upper certificate is now
  issued only when the estimated score covariance spans the full prespecified
  heterogeneity class; rank-deficient classes are structurally
  `INCONCLUSIVE` unless the one-sided lower test already flags them.
- Add `heterogeneity_class` so the report verdict can use a prespecified
  saturated group-time, additive, cohort, or event-time class. Reports expose
  separate lower bounds, upper bounds, ranks, and verdicts for all four.
- Retain the regular multiplier/reverse-triangle lower test without the former
  denominator buffer under the bounded local experiment. `q_band` exposes the
  more general, less powerful full confidence-ball fallback separately and it
  never determines the main verdict.
- Update the castle-doctrine lock: the additive homicide class certifies with
  an upper bound near 0.236, while the saturated class is rank-inconclusive at
  18/19. The divorce saturated class is rank-inconclusive at 49/167.
- Evaluate covariance traces and projected scores in group-time coordinates,
  eliminating the quadratic treated-cell covariance allocation. Bundle the
  reproducible 2,840-municipality Brazil property-tax extract and lock the
  finding that every class is flagged by its lower test.

## 0.7.1 - 2026-09-05

- Use the exact leverage-weighted noise trace
  `sum((1 - h_ii) * sigma_nu_i^2)` when observation-specific measurement-error
  standard deviations are supplied. Both this trace and the common-SD formula
  now use the full nuisance projection (fixed effects plus supplied controls).
- Add nuisance-control support to `eiv_adequacy`, including correct residual
  degrees of freedom and control-aware projection-compatibility diagnostics.
- Replace the bundled V-Dem interval-half-width proxy with V-Dem's direct
  posterior-standard-deviation fields and refresh the article reproduction
  locks. Clarify that `reliability_from_interval` is valid only when an interval
  half-width is substantively calibrated as one error standard deviation.

## 0.7.0 - 2026-09-01

- Add `certified_breakdown_reliability` and expose the paper's closed-form
  certified threshold in every measurement-error report.
- Separate coefficient and reliability uncertainty: `reliability_lower` and
  `gamma_lambda` implement the lower-bound rule, while the existing `gamma`
  keyword remains the coefficient budget. Reports now state when certification
  is conditional on treating a reliability input as known/consistent.
- Align the TWFE module with the current *Positive Weights Do Not Certify TWFE Inference* theory:
  direct CR1 cluster-score normalization is now the default, while AR(1),
  i.i.d., and user-supplied scales remain explicit sensitivity routes.
- Add the saturated post-treatment group-time class and verify the balanced-
  panel identity `Gamma_gt = Gamma`, alongside the cohort, event-time, and
  additive cohort-plus-event-time ladder.
- Replace point-envelope certification by a one-sided confidence construction
  for the projected group-time vector. HC2 projected-norm bounds remain valid
  at zero heterogeneity; HC3 bounds are reported as a sensitivity check.
  Trace-debiased quadratic pilots and percentile summaries remain descriptive.
- Separate every worst-case size envelope from the signed directional plug-in.
  Reports label envelopes as uniform upper bounds, never as realized rejection
  probabilities, and expose directional alignment.
- Replace the dense fixed-effect bootstrap regression with an absorbed,
  cluster-score implementation, making the 2,284-county application practical.
- Bundle the exact 15,988-row Callaway--Sant'Anna minimum-wage analysis extract.
  The reported target is now the equally weighted treated-cell ATT (about
  -5.2%), matching the theory rather than the differently weighted group
  aggregate. Its saturated-class point envelope is about 31.6%, and the
  projected-norm lower bound formally withholds certification.
- Add never-treated versus not-yet-treated comparison-group selection, direct
  and AR(1) scale outputs, CR1 standard errors, sign-reversal RMS, and a
  fixed-panel-length warning when `T` is large relative to the cluster count.
- Distinguish Gaussian mean-shift calculations from procedure-specific finite-
  cluster inference; alternative standard errors or bootstrap tests are not
  treated as drop-in changes to the Gaussian rejection map.
- Retain the old realized-size and bootstrap names only as deprecated
  compatibility aliases to the newly explicit directional and envelope fields.

## 0.6.0 - 2026-08-25

- Add `reliability_from_repeats` with covariance and equal-variance methods.
- Bundle the public Ashenfelter--Krueger twins extract and reproduce the current
  twins table, including the Rouse correlated-report sensitivity.
- Replace obsolete PSID article locks while retaining the dataset for backward
  compatibility; lock the Design 4/5 exact-normal endpoints at 3.58 and 3.67.
- Correct stale cluster-direction guidance and refresh data provenance,
  documentation, and the current manuscript snapshot.

## 0.5.1 - 2026-08-04

- Keep `cycle_report` output concise by hiding its detailed diagnostic notes in default REPL rendering.
- Preserve all notes in `report.notes` and add `show_notes(report)` for an explicit, numbered display; structural `INCONCLUSIVE` reasons remain visible in the default report.
- Retain the Julia 1.6 compatibility and CI fixes merged after the v0.5.0 tag.

## 0.5.0 - 2026-08-02

- Bundle the canonical public-domain 11-firm Grunfeld panel used by Paper A's concentrated-regime showcase, with pinned source and version provenance.
- Add `lambda_n` and `n_eff` to `DesignSummary` and share their computation with `leverage_report`.
- Add `score_concentration`, multiway `design_summary`, and the flat `adequacy_row` screen.
- Add the outcome-free `PanelAdequacy.applicable` pre-flight check and expose the binary-treatment granularity floor through `cycle_report` and `adequacy_row`.
- Add `eiv_adequacy_summary` for API parity with R.
- Canonicalize cycle packing and use deterministic multi-start traversal so row permutations cannot change the selected design.
- Lock the stable v0.5.0 captures for the public fixtures: KSS match `0.5110`, KSS wage `0.6112`, F-score `0.8274`, and calibrated dense `0.8341`; each meets or improves on the article's former lower-bound construction.
- Add cross-language fixtures, parity checks, property tests, and published-result regression locks.
- Add weak-dependency model adapters, CI, public documentation, licensing, and release metadata.
- Add explicit numeric nuisance-control support to design, score, leverage, screen, and exact-cycle APIs. For one continuous control, dense exact inference uses locally projected 2-by-3 supports; model adapters no longer ignore additional regressors.
- Lock the canonical 11-firm Grunfeld capital specification at `lambda_score = 0.7388` and valid controlled capture `kappa_C = 0.6270` over 32 supports.

## 0.4.2 - 2026-07-31

- Package the corrected Paper A, Paper B, Paper C, and diffuse-regime diagnostics with bundled replication panels.
