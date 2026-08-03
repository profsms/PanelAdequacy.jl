# Changelog

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
