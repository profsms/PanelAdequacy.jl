# Changelog

## 0.7.0 - 2026-09-01

- Align the TWFE module with the current *Is Bias Correction Enough?* theory:
  direct CR1 cluster-score normalization is now the default, while AR(1),
  i.i.d., and user-supplied scales remain explicit sensitivity routes.
- Separate the combined-class worst-case size envelope from the signed
  directional plug-in. Reports now label the former as a uniform upper bound,
  never as realized rejection probability, and expose directional alignment.
- Replace the dense fixed-effect bootstrap regression with an absorbed,
  cluster-score implementation, making the 2,284-county application practical.
- Bundle the exact 15,988-row Callaway--Sant'Anna minimum-wage analysis extract
  and lock its 29.3% envelope, 27.8% directional diagnostic, and sign-reversal
  calibration.
- Add never-treated versus not-yet-treated comparison-group selection, direct
  and AR(1) scale outputs, CR1 standard errors, sign-reversal RMS, and a
  fixed-panel-length warning when `T` is large relative to the cluster count.
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
