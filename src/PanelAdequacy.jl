"""
    PanelAdequacy

Panel-data inference-adequacy diagnostics and exact contrast inference for
fixed-effect designs. Paper A's engine constructs nuisance-annihilating
contrasts and sign-flip confidence sets; the remaining modules diagnose
variance-estimator, measurement-error, and TWFE-heterogeneity failures.

Current source map:
- Paper A — exact contrast inference under concentrated identifying variation
- Paper B — measurement-error adequacy
- Paper C — staggered-DiD / TWFE-heterogeneity adequacy
- diffuse companion — leverage / variance diagnostics
"""
module PanelAdequacy

using Printf
using Statistics
using LinearAlgebra
using Random
using SpecialFunctions: erfc, erfcinv

export DesignSummary, design_summary, twoway_demean, multiway_demean, fe_dimension
export AdequacyReport
export leverage_report, fe_leverage
export eiv_adequacy, reliability_from_interval, reliability_from_ratio,
       breakdown_reliability, cluster_diagnostics, projection_compatibility,
       tau2_crit, eta_finite_n
export twfe_design, twfe_adequacy, twfe_gammas
export cycle_report, cycle_capture, cycle_contrasts, contrast_system,
       support_compatibility, signflip_test, signflip_interval
export datasets, datapath, load_dataset

include("design.jl")
include("normal.jl")
include("leverage.jl")
include("measurement_error.jl")
include("staggered_weights.jl")
include("cycle.jl")
include("datasets.jl")

# =============================================================================
# The unified result object (spec §2.2)
# =============================================================================

const PATHOLOGY_TITLES = Dict(
    :leverage            => "Leverage / Variance (diffuse-regime companion)",
    :measurement_error   => "Measurement Error (Paper B)",
    :twfe_heterogeneity  => "TWFE Heterogeneity (Paper C)",
    :cycle_inference     => "Concentrated Identifying Variation (Paper A)",
)

"""
    AdequacyReport

Unified result object returned by every diagnostic (spec §2.2).

Fields:
- `pathology`    : `:cycle_inference` | `:leverage` | `:measurement_error` |
                   `:twfe_heterogeneity`
- `design`       : the [`DesignSummary`](@ref)
- `statistic`    : pathology-specific statistic(s) as a `NamedTuple`
                   (e.g. `(lambda_hat=..., )`, `(Gamma=..., neg_share=...)`)
- `eta`          : feasible non-centrality (`nothing` if not applicable)
- `threshold`    : critical value at the user's `(alpha, delta)`
- `breakdown`    : breakdown reliability / threshold
- `implied_size` : implied size of the nominal-`alpha` test
- `verdict`      : `:CERTIFIED` | `:POINT_PASS` | `:FLAGGED` | `:INCONCLUSIVE`
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
        verdict in (:CERTIFIED, :POINT_PASS, :FLAGGED, :INCONCLUSIVE) ||
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
        if haskey(s, :psi_hat) && s.psi_hat != 1.0
            push!(lines, @sprintf("Cluster-robust: psi_hat = %.3f, s_CR = %.4g, t^CR = %.3f (|eta|, breakdown deflated by 1/sqrt(psi) = %.3f)",
                                  s.psi_hat, s.s_CR, s.t_CR, 1 / sqrt(s.psi_hat)))
            if haskey(s, :cluster)
                c = s.cluster
                push!(lines, @sprintf("  cluster design: G = %d, max size = %d, max_g A_g/tau*2 = %.3f, d_ne/G = %.3f%s",
                                      c.G, c.max_size, c.max_energy, c.ratio_ne,
                                      isempty(c.nested) ? "" : " (nested: " * join(c.nested, ", ") * ")"))
                haskey(c, :projection_ratio) &&
                    push!(lines, isfinite(c.projection_ratio) ?
                          @sprintf("  direct projection compatibility chi_proj = %.5f", c.projection_ratio) :
                          "  direct projection compatibility chi_proj = not computed (allocation guard)")
            end
        end
        haskey(s, :eta_upper) &&
            push!(lines, @sprintf("Conservative |eta| (upper-bound pilot) = %.3f",
                                  s.eta_upper))
    elseif pathology === :twfe_heterogeneity && haskey(s, :Gamma)
        line = @sprintf("Design statistic Gamma = %.3f", s.Gamma)
        haskey(s, :neg_share) &&
            (line *= @sprintf("   negative-weight share = %.1f%%", 100 * s.neg_share))
        push!(lines, line)
        haskey(s, :Gamma_cmb) &&
            push!(lines, @sprintf("  restricted ladder: Gamma_c+e = %.3f | Gamma_evt = %.3f | Gamma_coh = %.3f",
                                  s.Gamma_cmb, s.Gamma_evt, s.Gamma_coh))
        haskey(s, :Gamma_CR) &&
            push!(lines, @sprintf("Cluster-robust (psi_hat = %.3f): Gamma_c+e,CR = %.3f | Gamma_CR = %.3f",
                                  s.psi_hat, s.Gamma_cmb_CR, s.Gamma_CR))
        haskey(s, :beta) &&
            push!(lines, @sprintf("TWFE beta_hat = %.4g   sigma = %.4g", s.beta, s.sigma))
        haskey(s, :pilot_cmb) &&
            push!(lines, @sprintf("Covariance-corrected pilots c_S/sigma: c+e = %.3g | evt = %.3g | coh = %.3g",
                                  s.pilot_cmb, s.pilot_evt, s.pilot_coh))
        if haskey(s, :size_cmb)
            l = @sprintf("Worst-case size: combined-class = %.1f%% (headline) | cohort %.1f%% | event %.1f%%",
                         100*s.size_cmb, 100*s.size_coh, 100*s.size_evt)
            push!(lines, l)
        end
        if haskey(s, :boot) && s.boot !== nothing
            b = s.boot
            push!(lines, @sprintf("  wild bootstrap (B=%d): combined median %.1f%%, 95%% [%.1f, %.1f]; psi in [%.2f, %.2f]",
                                  b.n, 100*b.cmb_med, 100*b.cmb_lo, 100*b.cmb_hi, b.psi_lo, b.psi_hi))
        end
        haskey(s, :size_realized) &&
            push!(lines, @sprintf("Realized-profile size (CR) = %.1f%%", 100*s.size_realized))
    elseif pathology === :leverage && haskey(s, :max_leverage)
        line = @sprintf("Max leverage max_i H_ii = %.3f", s.max_leverage)
        haskey(s, :leverage_spread) &&
            (line *= @sprintf(" | spread hmax/hmin = %.2f", s.leverage_spread))
        push!(lines, line)
        haskey(s, :lambda_n) &&
            push!(lines, @sprintf("Design conditions: lambda_n = %.4f (N_eff = %.1f) | max|H_ii - rho| = %.3f",
                                  s.lambda_n, s.n_eff, s.uniform_leverage_gap))
        haskey(s, :score_lambda_n) && isfinite(s.score_lambda_n) &&
            push!(lines, @sprintf("Realized score concentration: lambda_score = %.4f (N_eff,score = %.1f)",
                                  s.score_lambda_n, s.score_n_eff))
        if haskey(s, :se_df)
            push!(lines, @sprintf("SE(beta): df-corrected %.4g | HC0 %.4g | HC2 %.4g | HC3 %.4g",
                                  s.se_df, s.se_hc0, s.se_hc2, s.se_hc3))
            push!(lines, @sprintf("beta_hat = %.4g   t (HC2) = %.2f",
                                  s.beta, s.t_hc2))
        end
    elseif pathology === :cycle_inference && haskey(s, :kappa)
        push!(lines, @sprintf("Concentration: lambda_n = %.4f (N_eff = %.1f)",
                              s.lambda_n, s.n_eff))
        haskey(s, :score_lambda_n) &&
            push!(lines, @sprintf("Realized score concentration: lambda_score = %.4f (N_eff,score = %.1f)",
                                  s.score_lambda_n, s.score_n_eff))
        ctext = haskey(s, :effective_C) && s.effective_C != s.C ?
                @sprintf("%d supports (%d treatment-loaded)", s.C, s.effective_C) :
                @sprintf("%d supports", s.C)
        push!(lines, @sprintf("Capture kappa_C = %.4f over %s (cycle-space dim %d) | capture-implied SE ratio %.3fx | max share %.3f",
                              s.kappa, ctext, s.cycle_dim, s.se_price, s.max_share))
        if s.beta_tilde !== nothing
            l = @sprintf("Contrast estimate beta~ = %.4g", s.beta_tilde)
            if s.ci_lo !== nothing
                level = haskey(s, :ci_level) ? s.ci_level : 0.95
                l *= @sprintf("   exact %.0f%% set: [%.4g, %.4g]%s",
                              100 * level, s.ci_lo, s.ci_hi,
                              haskey(s, :ci_grid_truncated) && s.ci_grid_truncated ?
                              " (conservative: grid boundary reached)" : "")
            else
                l *= "   exact set: EMPTY at this level"
            end
            push!(lines, l)
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
        g = haskey(r.statistic, :gamma) ? r.statistic.gamma : nothing
        g === nothing ? @printf(io, "VERDICT: CERTIFIED at delta=%.2g", r.delta) :
            @printf(io, "VERDICT: FORMALLY CERTIFIED at (alpha, delta, gamma) = (%.2g, %.2g, %.2g)",
                    r.alpha, r.delta, g)
    elseif r.verdict === :POINT_PASS
        @printf(io, "VERDICT: POINT PASS at delta=%.2g (descriptive — not a certificate)", r.delta)
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
