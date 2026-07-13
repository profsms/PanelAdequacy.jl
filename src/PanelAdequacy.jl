"""
    PanelAdequacy

Panel-data inference-adequacy diagnostics. Certifies whether naive inference
on an ALREADY-ESTIMATED fixed-effect design is size-controlled, and if not,
by how much (spec §0). Diagnostic only: it consumes fitted output, implements
no remedies, and points to them.

Modules (one per source paper):
- Module A — leverage / variance diagnostics (Paper A)
- Module B — measurement-error adequacy (Paper B)
- Module C — staggered-DiD / TWFE-heterogeneity adequacy (Paper C)
"""
module PanelAdequacy

using Printf
using Statistics
using LinearAlgebra
using SpecialFunctions: erfc, erfcinv

export DesignSummary, design_summary, twoway_demean, fe_dimension
export AdequacyReport
export leverage_report, fe_leverage
export eiv_adequacy, reliability_from_interval, reliability_from_ratio,
       breakdown_reliability
export twfe_design, twfe_adequacy
export datasets, datapath, load_dataset

include("design.jl")
include("normal.jl")
include("leverage.jl")
include("measurement_error.jl")
include("staggered_weights.jl")
include("datasets.jl")

# =============================================================================
# The unified result object (spec §2.2)
# =============================================================================

const PATHOLOGY_TITLES = Dict(
    :leverage            => "Leverage / Variance (Paper A)",
    :measurement_error   => "Measurement Error (Paper B)",
    :twfe_heterogeneity  => "TWFE Heterogeneity (Paper C)",
)

"""
    AdequacyReport

Unified result object returned by every diagnostic (spec §2.2).

Fields:
- `pathology`    : `:leverage` | `:measurement_error` | `:twfe_heterogeneity`
- `design`       : the [`DesignSummary`](@ref)
- `statistic`    : pathology-specific statistic(s) as a `NamedTuple`
                   (e.g. `(lambda_hat=..., )`, `(Gamma=..., neg_share=...)`)
- `eta`          : feasible non-centrality (`nothing` if not applicable)
- `threshold`    : critical value at the user's `(alpha, delta)`
- `breakdown`    : breakdown reliability / threshold
- `implied_size` : implied size of the nominal-`alpha` test
- `verdict`      : `:CERTIFIED` | `:FLAGGED` | `:INCONCLUSIVE`
- `alpha`, `delta` : tolerances used
- `notes`        : honesty caveats triggered (e.g. "conservative pilot used")
"""
struct AdequacyReport
    pathology::Symbol
    design::DesignSummary
    statistic::NamedTuple
    eta::Union{Nothing,Float64}
    threshold::Union{Nothing,Float64}
    breakdown::Union{Nothing,Float64}
    implied_size::Union{Nothing,Float64}
    verdict::Symbol
    alpha::Float64
    delta::Float64
    notes::Vector{String}

    function AdequacyReport(pathology, design, statistic, eta, threshold,
                            breakdown, implied_size, verdict, alpha, delta, notes)
        haskey(PATHOLOGY_TITLES, pathology) ||
            throw(ArgumentError("unknown pathology :$pathology"))
        verdict in (:CERTIFIED, :FLAGGED, :INCONCLUSIVE) ||
            throw(ArgumentError("unknown verdict :$verdict"))
        new(pathology, design, statistic, eta, threshold, breakdown,
            implied_size, verdict, alpha, delta, notes)
    end
end

# Module-specific rendering of the statistic line(s). Each module adds a
# specialised branch as it is built; the fallback prints raw key = value pairs.
function _statistic_lines(pathology::Symbol, s::NamedTuple)
    lines = String[]
    if pathology === :measurement_error && haskey(s, :lambda_hat)
        line = @sprintf("Within reliability lambda_hat = %.3f", s.lambda_hat)
        haskey(s, :noise_ratio) && isfinite(s.noise_ratio) &&
            (line *= @sprintf("   ((1-lambda)/lambda = %.3f)", s.noise_ratio))
        push!(lines, line)
        haskey(s, :beta_corr) &&
            push!(lines, @sprintf("Pilot: beta* = %.4g -> corrected beta0 = %.4g (se %.3g)",
                                  s.beta_star, s.beta_corr, s.se_beta_corr))
        haskey(s, :psi_hat) && s.psi_hat != 1.0 &&
            push!(lines, @sprintf("Cluster-robust standardization: psi_hat = %.3f (|eta|, breakdown deflated by 1/sqrt(psi) = %.3f)",
                                  s.psi_hat, 1 / sqrt(s.psi_hat)))
        haskey(s, :eta_upper) &&
            push!(lines, @sprintf("Conservative |eta| (upper-bound pilot) = %.3f",
                                  s.eta_upper))
    elseif pathology === :twfe_heterogeneity && haskey(s, :Gamma)
        line = @sprintf("Design statistic Gamma = %.3f", s.Gamma)
        haskey(s, :neg_share) &&
            (line *= @sprintf("   negative-weight share = %.1f%%", 100 * s.neg_share))
        push!(lines, line)
        haskey(s, :Gamma_CR) &&
            push!(lines, @sprintf("Cluster-robust Gamma_CR = %.3f (psi_hat = %.3f)",
                                  s.Gamma_CR, s.psi_hat))
        haskey(s, :beta) &&
            push!(lines, @sprintf("TWFE beta_hat = %.4g   sigma = %.4g", s.beta, s.sigma))
        haskey(s, :cohort_sd_raw) &&
            push!(lines, @sprintf("Cohort dispersion: raw sd %.4g -> shrunk %.4g (factor %.3f)",
                                  s.cohort_sd_raw, s.cohort_sd_shrunk, s.shrink_factor))
        haskey(s, :eta_worst) &&
            push!(lines, @sprintf("Worst-case |eta| (shrunk pilot) = %.3g", s.eta_worst))
    elseif pathology === :leverage && haskey(s, :max_leverage)
        line = @sprintf("Max leverage max_i H_ii = %.3f", s.max_leverage)
        haskey(s, :leverage_spread) &&
            (line *= @sprintf(" | spread hmax/hmin = %.2f", s.leverage_spread))
        push!(lines, line)
        if haskey(s, :se_cjn)
            push!(lines, @sprintf("SE(beta): CJN %.4g | HC0 %.4g | HC2/LO %.4g | HC3 %.4g",
                                  s.se_cjn, s.se_hc0, s.se_hc2, s.se_hc3))
            push!(lines, @sprintf("beta_hat = %.4g   t (HC2/LO) = %.2f",
                                  s.beta, s.t_hc2))
        end
    else
        for k in keys(s)
            push!(lines, "$(k) = $(s[k])")
        end
    end
    return lines
end

function Base.show(io::IO, ::MIME"text/plain", r::AdequacyReport)
    println(io, "Panel Adequacy Report — ", PATHOLOGY_TITLES[r.pathology])
    d = r.design
    if d.N > 0
        @printf(io, "Design: n=%d, N=%d, T=%d, d_K=%d, rho=%.4f\n",
                d.n, d.N, d.T, d.d_K, d.rho)
    else  # summary-form input: unit/time structure not supplied
        @printf(io, "Design: n=%d, d_K=%d, rho=%.4f (from summary input)\n",
                d.n, d.d_K, d.rho)
    end
    for line in _statistic_lines(r.pathology, r.statistic)
        println(io, line)
    end
    if r.eta !== nothing
        @printf(io, "Non-centrality |eta| = %.3f", abs(r.eta))
        r.threshold !== nothing &&
            @printf(io, "   Threshold (delta=%.2g) = %.3f", r.delta, r.threshold)
        println(io)
    end
    r.breakdown !== nothing &&
        @printf(io, "Breakdown threshold = %.3f\n", r.breakdown)
    r.implied_size !== nothing &&
        @printf(io, "Implied size of nominal %.0f%% test: %.1f%%\n",
                100 * r.alpha, 100 * r.implied_size)
    if r.verdict === :CERTIFIED
        @printf(io, "VERDICT: CERTIFIED at delta=%.2g", r.delta)
    elseif r.verdict === :FLAGGED
        @printf(io, "VERDICT: FLAGGED at delta=%.2g", r.delta)
    else
        print(io, "VERDICT: INCONCLUSIVE")
    end
    for note in r.notes
        print(io, "\nNote: ", note)
    end
end

Base.show(io::IO, r::AdequacyReport) = print(io,
    "AdequacyReport(:", r.pathology, ", verdict=:", r.verdict, ")")

end # module
