# PanelAdequacy.jl

PanelAdequacy.jl screens fixed-effect panel designs for concentrated identifying variation, leverage, measurement error, and TWFE heterogeneity, and supplies exact contrast inference where Gaussian approximations are fragile.

[![CI](https://github.com/profsms/PanelAdequacy.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/profsms/PanelAdequacy.jl/actions/workflows/ci.yml)
[![Cross-language parity](https://github.com/profsms/PanelAdequacy.jl/actions/workflows/parity.yml/badge.svg)](https://github.com/profsms/PanelAdequacy.jl/actions/workflows/parity.yml)

## Install

Install the current release from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/profsms/PanelAdequacy.jl", rev="v0.5.1")
```

After registration in Julia General, `Pkg.add("PanelAdequacy")` will be sufficient.

## Quickstart

```julia
using PanelAdequacy; d = load_dataset("grunfeld_panel")
PanelAdequacy.applicable(d.capital, d.firm, d.year; controls=d.value)
report = cycle_report(d.invest, d.capital, d.firm, d.year; controls=d.value, interval=false)
(report.verdict, report.statistic.lambda_score, report.statistic.kappa)
```

## Entry Points

- `leverage_report(y, x, unit, time)` is the diffuse-regime diagnostic. It checks treatment concentration and leverage balance and compares df-corrected and HC0-HC3 inference.
- `cycle_report(y, x, unit, time)` is the concentrated-regime workflow from Paper A. It reports capture, granularity, an exact sign-flip test, and an exact confidence set.
- `eiv_adequacy(y, x, unit, time; ...)` screens continuous-regressor measurement error under Paper B's exact-normal mapping and conservative certificate.
- `twfe_adequacy(y, unit, time, first_treat)` screens staggered-DiD/TWFE designs for heterogeneous-effect exposure.

Default `cycle_report` rendering is concise. Detailed caveats remain available in `report.notes` and as a numbered display through `show_notes(report)`.

`adequacy_row(x, unit, time; y=nothing)` composes the design and capture diagnostics into one flat record for prevalence screens.

The regression diagnostics accept `controls=` for numeric nuisance covariates. Exact cycle packing supports one linearly independent control and constructs locally annihilating 2-by-3 supports; it never substitutes globally residualized outcomes for valid disjoint contrasts.

Julia's `Base.applicable` is a builtin, so the pre-flight check uses the qualified spelling `PanelAdequacy.applicable(x, unit, time)`; it reports structural binary-treatment granularity failures before packing.

On Julia 1.9+, package extensions add these methods for `GLM.jl` linear models and `FixedEffectModels.jl`. GLM methods take the fitted model plus `unit` and `time`; because a `FixedEffectModel` intentionally does not retain its source table, its methods also require the exact estimation-sample table and explicit `y`, `x`, `unit`, and `time` column symbols.

## Packing Methods

- `:structured` exploits four-cycle structure in dense panels and 2-by-3 local projections when one continuous control is supplied.
- `:sparse` is the scalable method for AKM-style mobility networks.
- `:greedy` is a cheap lower bound that keeps the best of four deterministic DFS traversals.

## Verdicts

- `POINT_PASS` is a descriptive pass at the requested finite-design threshold; it is not a theorem-level certificate.
- `FLAGGED` marks a concentration or adequacy failure for which conventional inference is not licensed.
- `INCONCLUSIVE` is returned when the enumeration floor `2^(1-effective_C)` exceeds `alpha`. Its `reason` distinguishes the coding-invariant binary-treatment floor based on `min(n_treated, n_untreated)`, which repacking cannot fix, from a selected packing that may be improved.

Measurement-error reports can additionally return `CERTIFIED` when the formal upper-bound pilot passes.

## Design-Only Use

`design_summary`, `cycle_capture`, and `adequacy_row(...; y=nothing)` require only `(x, D, controls)`, not an outcome. In particular, `rho`, `V_n`, `lambda_n`, `n_eff`, and `kappa_C` are available before estimation.

## R Package And Data

The sibling R package is [panelcert](https://github.com/profsms/panelcert). Fixed CSV designs and CI enforce agreement to `1e-12` on numerical outputs and exact agreement on integer fields, verdicts, and packed supports.

Dataset access follows language conventions: Julia exposes `datasets()`, `datapath()`, and `load_dataset()`; R exposes the same bundled panels through `LazyData`. Redistributable paper panels, including the public-domain canonical 11-firm Grunfeld showcase, are bundled for offline replication. `DATA_SOURCES.md` records pinned provenance and explains why the KSS test extract uses a checksum-pinned direct download instead.

## Citation

Please cite both working papers when the corresponding diagnostics are used:

- Halkiewicz, Stanislaw M. S. (2026). *Exact Inference in Fixed-Effect Regressions with Concentrated Identifying Variation*.
- Halkiewicz, Stanislaw M. S. (2026). *Fixed-Effect Saturation Is Not Weak Identification: Certifying Inference under Measurement Error*.

See `CITATION.cff` for machine-readable metadata.
