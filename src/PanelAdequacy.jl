"""
    PanelAdequacy

Panel-data inference-adequacy diagnostics and exact contrast inference for
fixed-effect designs. Paper A's engine constructs nuisance-annihilating
contrasts and sign-flip confidence sets; the remaining modules diagnose
variance-estimator, measurement-error, and TWFE-heterogeneity failures.

Current source map:
- Paper A — exact contrast inference under concentrated identifying variation
- *Breakdown Reliability for Saturated Fixed-Effect Inference* — measurement-error adequacy
- staggered-DiD / TWFE-heterogeneity adequacy
- diffuse companion — leverage / variance diagnostics
"""
module PanelAdequacy

using Printf
using Statistics
using LinearAlgebra
using Random
using SparseArrays
using SpecialFunctions: erfc, erfcinv, gamma_inc, loggamma

export DesignSummary, design_summary, twoway_demean, multiway_demean, fe_dimension
export AdequacyReport, show_notes
export leverage_report, fe_leverage
export score_concentration, applicable, adequacy_row
export eiv_adequacy, reliability_from_interval, reliability_from_ratio,
       reliability_from_repeats,
       breakdown_reliability, certified_breakdown_reliability,
       cluster_diagnostics, projection_compatibility,
       tau2_crit, eta_finite_n, eiv_adequacy_summary
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
include("screen.jl")
include("datasets.jl")

# =============================================================================
# The unified result object (spec §2.2)
# =============================================================================

const PATHOLOGY_TITLES = Dict(
    :leverage            => "Leverage / Variance (diffuse-regime companion)",
    :measurement_error   => "Measurement Error",
    :twfe_heterogeneity  => "TWFE Heterogeneity",
    :cycle_inference     => "Concentrated Identifying Variation",
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
- `notes`        : honesty caveats triggered (e.g. "conservative pilot used").
                   Cycle-inference notes are retained here but hidden in the
                   default display; call [`show_notes`](@ref) to print them.
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
            push!(lines, @sprintf("Certified breakdown = %.3f   reliability lower bound = %.3f   |eta| upper bound = %.3f",
                                  s.breakdown_certified, s.reliability_lower,
                                  s.eta_upper))
    elseif pathology === :twfe_heterogeneity && haskey(s, :Gamma)
        line = @sprintf("Design statistic Gamma = %.3f", s.Gamma)
        haskey(s, :neg_share) &&
            (line *= @sprintf("   negative-weight share = %.1f%%", 100 * s.neg_share))
        push!(lines, line)
        haskey(s, :Gamma_cmb) &&
            push!(lines, @sprintf("  restricted ladder: Gamma_gt = %.3f | Gamma_c+e = %.3f | Gamma_evt = %.3f | Gamma_coh = %.3f",
                                  haskey(s, :Gamma_gt) ? s.Gamma_gt : s.Gamma,
                                  s.Gamma_cmb, s.Gamma_evt, s.Gamma_coh))
        haskey(s, :Gamma_CR) &&
            push!(lines, @sprintf("Cluster normalization (%s; psi_hat = %.3f): Gamma_gt,CR = %.3f | Gamma_c+e,CR = %.3f",
                                  haskey(s, :normalization) ? String(s.normalization) : "legacy",
                                  s.psi_hat,
                                  haskey(s, :Gamma_gt_CR) ? s.Gamma_gt_CR : s.Gamma_CR,
                                  s.Gamma_cmb_CR))
        haskey(s, :beta) &&
            push!(lines, @sprintf("TWFE beta_hat = %.4g   sigma = %.4g", s.beta, s.sigma))
        haskey(s, :att_target) && isfinite(s.att_target) &&
            push!(lines, @sprintf("Target-matched robust ATT = %.4g", s.att_target))
        haskey(s, :pilot_cmb) &&
            push!(lines, @sprintf("Trace-debiased point pilots c_S/sigma: gt = %.3g | c+e = %.3g | evt = %.3g | coh = %.3g",
                                  haskey(s, :pilot_gt) ? s.pilot_gt : NaN,
                                  s.pilot_cmb, s.pilot_evt, s.pilot_coh))
        if haskey(s, :size_cmb)
            l = @sprintf("Point worst-case envelopes: group-time = %.1f%% | c+e = %.1f%% | cohort %.1f%% | event %.1f%%",
                         100*(haskey(s, :size_gt) ? s.size_gt : s.size_cmb),
                         100*s.size_cmb, 100*s.size_coh, 100*s.size_evt)
            push!(lines, l)
        end
        if haskey(s, :K_lower_gt) && isfinite(s.K_lower_gt)
            selected = haskey(s, :selected_class) ? String(s.selected_class) : "group_time"
            push!(lines, @sprintf("Selected heterogeneity class: %s (each bound is a separate one-sided %.1f%% statement)",
                                  selected, 100*(1-s.gamma)))
            for (key, label) in ((:coh, "cohort"), (:evt, "event-time"),
                                 (:cmb, "additive"), (:gt, "group-time"))
                lower = getproperty(s, Symbol("K_lower_", key))
                upper = getproperty(s, Symbol("K_upper_", key))
                verdict = getproperty(s, Symbol("verdict_", key))
                if isfinite(upper)
                    push!(lines, @sprintf("  %s: lower %.3f; upper %.3f; %s",
                                          label, lower, upper, String(verdict)))
                else
                    expected = getproperty(s, Symbol("expected_rank_", key))
                    rank = getproperty(s, Symbol("covariance_rank_", key))
                    push!(lines, @sprintf("  %s: lower %.3f; upper unavailable (covariance rank %d/%d); %s",
                                          label, lower, rank, expected,
                                          String(verdict)))
                end
            end
        end
        if haskey(s, :boot) && s.boot !== nothing
            b = s.boot
            push!(lines, @sprintf("  descriptive point-envelope bootstrap (B=%d): median %.1f%%, 95%% [%.1f, %.1f]; psi in [%.2f, %.2f]",
                                  b.n, 100*b.envelope_med, 100*b.envelope_lo,
                                  100*b.envelope_hi, b.psi_lo, b.psi_hi))
        end
        if haskey(s, :size_directional)
            push!(lines, @sprintf("Directional plug-in: eta = %+.3f (alignment %+.3f), size %.1f%%",
                                  s.eta_directional, s.directional_alignment,
                                  100*s.size_directional))
            if haskey(s, :boot) && s.boot !== nothing
                b = s.boot
                push!(lines, @sprintf("  wild directional size: median %.1f%%, 95%% [%.1f, %.1f]",
                                      100*b.directional_med, 100*b.directional_lo,
                                      100*b.directional_hi))
            end
        end
        haskey(s, :sign_reversal_rms) &&
            push!(lines, @sprintf("Sign-reversal RMS threshold = %.4g", s.sign_reversal_rms))
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
        haskey(s, :reason) && !isempty(s.reason) &&
            push!(lines, "Reason: " * s.reason)
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
        if r.pathology === :twfe_heterogeneity
            @printf(io, "Worst-case |eta| envelope = %.3f", abs(r.eta))
        else
            @printf(io, "Non-centrality |eta| = %.3f", abs(r.eta))
        end
        r.threshold !== nothing &&
            @printf(io, "   Threshold (delta=%.2g) = %.3f", r.delta, r.threshold)
        println(io)
    end
    r.breakdown !== nothing &&
        @printf(io, "Breakdown threshold = %.3f\n", r.breakdown)
    if r.implied_size !== nothing
        if r.pathology === :twfe_heterogeneity
            @printf(io, "Worst-case size envelope for nominal %.0f%% test: %.1f%%\n",
                    100 * r.alpha, 100 * r.implied_size)
        else
            @printf(io, "Implied size of nominal %.0f%% test: %.1f%%\n",
                    100 * r.alpha, 100 * r.implied_size)
        end
    end
    if r.verdict === :CERTIFIED
        g = haskey(r.statistic, :gamma) ? r.statistic.gamma : nothing
        g === nothing ? @printf(io, "VERDICT: CERTIFIED at delta=%.2g", r.delta) :
            @printf(io, "VERDICT: FORMALLY CERTIFIED at (alpha, delta, gamma) = (%.2g, %.2g, %.2g)",
                    r.alpha, r.delta, g)
    elseif r.verdict === :POINT_PASS
        @printf(io, "VERDICT: POINT PASS at delta=%.2g (descriptive — not a certificate)", r.delta)
    elseif r.verdict === :FLAGGED
        if r.pathology === :twfe_heterogeneity
            @printf(io, "VERDICT: FLAGGED at delta=%.2g (uniform certificate withheld)", r.delta)
        else
            @printf(io, "VERDICT: FLAGGED at delta=%.2g", r.delta)
        end
    else
        print(io, "VERDICT: INCONCLUSIVE")
    end
    if r.pathology === :cycle_inference
        isempty(r.notes) || @printf(io,
            "\nDiagnostic notes hidden (%d); call show_notes(report) to display them.",
            length(r.notes))
    else
        for note in r.notes
            print(io, "\nNote: ", note)
        end
    end
end

"""
    show_notes([io::IO], report::AdequacyReport)

Print the detailed diagnostic notes stored in `report.notes`. Cycle-inference
notes are hidden in the default report rendering so that the headline design,
capture, interval, reason, and verdict remain easy to scan.
"""
function show_notes(io::IO, r::AdequacyReport)
    if isempty(r.notes)
        println(io, "No diagnostic notes.")
        return nothing
    end
    println(io, "Diagnostic notes for ", PATHOLOGY_TITLES[r.pathology],
            " (", length(r.notes), "):")
    for (i, note) in enumerate(r.notes)
        i > 1 && println(io)
        println(io, i, ". ", note)
    end
    return nothing
end

show_notes(r::AdequacyReport) = show_notes(stdout, r)

Base.show(io::IO, r::AdequacyReport) = print(io,
    "AdequacyReport(:", r.pathology, ", verdict=:", r.verdict, ")")

end # module
