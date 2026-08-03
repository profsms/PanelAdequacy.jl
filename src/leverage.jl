# =============================================================================
# Diffuse-regime companion — leverage / variance diagnostics
#
# SOURCE: the REPAIRED framework (corrections.tex, "Corrected Framework and
# Proofs for MS-3897", July 2026), NOT the submitted arXiv:2607.05215, whose
# Lemma 2.2 is false. The repair conditions on F_n = sigma(X, D_K) throughout
# and self-normalizes by the sample Hessian V_n = X'M_K X rather than by the
# population nQ_K. Consequences implemented here:
#
#   Hessian (lem:hess)   X'M_K X / (n Qbar_n) ->p 1 - rho   (NOT 1)
#                        so V_n estimates the (1-rho)-DEFLATED signal; the
#                        primitive is recovered as Qhat = V_n / (n(1-rho)).
#   full leverage        H_ii = (P_K)_ii + Xt_i^2 / V_n
#   self-norm. leverage  lambda_n = max_i Xt_i^2 / V_n = o_p(1)   (ass:des(ii))
#                        — the Lindeberg-type condition of the CLT, and the
#                        same statistic Paper A's concentrated regime negates.
#   uniform leverage     max_i |H_ii - rho_n| = o_p(1)            (ass:hc(ii))
#                        — a STRONG balance condition. Holds for balanced
#                        one-/two-way grouped designs with equal cell sizes;
#                        FAILS for unbalanced AKM-type bipartite designs.
#   estimators           HC0 = sum Xt_i^2 u_i^2,  HC1 = n/(n-d_K-1) * HC0,
#                        HC2 = sum Xt_i^2 u_i^2/(1-H_ii),
#                        HC3 = sum Xt_i^2 u_i^2/(1-H_ii)^2,
#                        naive sigma^2 = u'u/n, df-corrected = u'u/(n-d_K-1).
#   HC limits (thm:hc)   with omega_eff^2 = (1-rho) omega^2 + rho mu,
#                        HC0/V_n ->p (1-rho) omega_eff^2   HC1/V_n ->p omega_eff^2
#                        HC2/V_n ->p omega_eff^2           HC3/V_n ->p omega_eff^2/(1-rho)
#                        T^HCc => N(0, omega^2 / L_c).
#   sizes (thm:size)     T^df => N(0,1); T^naive => N(0, 1/(1-rho)), size
#                        2(1 - Phi(z sqrt(1-rho))). Under conditional
#                        homoskedasticity HC0 shares that limit and HC3 gives
#                        2(1 - Phi(z / sqrt(1-rho))) — UNDER-rejection.
#
# TERMINOLOGY: the residual degrees-of-freedom correction u'u/(n-d_K-1) is
# classical; the label "CJN-corrected" is dropped. HC2 is MacKinnon-White
# (1985); leave-out variance estimation in FE designs is Kline-Saggio-Solvsten
# (2020) and Jochmans (2022); the Hadamard/unbiased quadratic-form idea traces
# to Rao-type estimators analyzed in Cattaneo-Jansson-Newey (2018).
#
# EXACTNESS: the exact-t result holds conditional on (X, D_K) AND under
# Gaussian errors. Exactness is never advertised here without both.
#
# NEGATIVE RESULT, stated modestly: under the model with strict exogeneity the
# score Xt'eps is conditionally unbiased for every tau^2 > 0, so no weak-IV
# mechanism exists WITHIN THIS MODEL CLASS. The interesting case is when
# exogeneity fails — measurement error in X — which is Module B.
# =============================================================================

"""
    fe_leverage(unit, time) -> Vector{Float64}

Diagonal of the projection onto the two-way FE space, `(P_K)_ii`, for each
observation. Computed exactly from the (N+T)x(N+T) FE Gram matrix via
pseudo-inverse (handles the rank deficiency d_K = N + T - #components).
"""
function fe_leverage(unit::AbstractVector, time::AbstractVector)
    uid, tid, N, T = _integer_codes(unit, time)
    return _fe_leverage_diag(uid, tid, N, T)
end

function _fe_leverage_diag(uid::Vector{Int}, tid::Vector{Int}, N::Int, T::Int)
    G = zeros(N + T, N + T)
    for k in eachindex(uid)
        u, t = uid[k], N + tid[k]
        G[u, u] += 1.0
        G[t, t] += 1.0
        G[u, t] += 1.0
        G[t, u] += 1.0
    end
    # Julia 1.6's Symmetric pinv path forwards an unsupported SVD keyword.
    Gp = pinv(G)
    p = Vector{Float64}(undef, length(uid))
    for k in eachindex(uid)
        u, t = uid[k], N + tid[k]
        p[k] = Gp[u, u] + 2 * Gp[u, t] + Gp[t, t]
    end
    return p
end

"Score concentration from residualized treatment and regression residuals."
function _score_concentration(xt::AbstractVector{<:Real},
                              u::AbstractVector{<:Real})
    length(xt) == length(u) || throw(ArgumentError(
        "xt and residuals must have equal length"))
    score_mass = sum(abs2.(xt) .* abs2.(u))
    if !(score_mass > 0)
        return (lambda_score=NaN, H_score=NaN, n_eff_score=NaN)
    end
    shares = abs2.(xt) .* abs2.(u) ./ score_mass
    H_score = sum(abs2, shares)
    return (lambda_score=maximum(shares), H_score=H_score,
            n_eff_score=inv(H_score))
end

"""
    score_concentration(y, x, unit, time; controls=nothing) -> NamedTuple

Realized score concentration for the two-way fixed-effect regression of `y`
on `x`, after removing optional numeric nuisance `controls`:
`(lambda_score, H_score, n_eff_score)`. It is a one-outcome warning
diagnostic, not by itself a consistent estimator of population score
concentration under unrestricted heteroskedasticity.
"""
function score_concentration(y::AbstractVector{<:Real},
                             x::AbstractVector{<:Real},
                             unit::AbstractVector, time::AbstractVector;
                             controls=nothing)
    uid, tid, N, T = _integer_codes(unit, time)
    n = length(uid)
    (length(y) == n && length(x) == n) || throw(ArgumentError(
        "y, x, unit, time must have equal length"))
    partial = _partial_within_codes(x, uid, tid, N, T; controls=controls)
    xt = partial.xt
    yt = _partial_outcome_codes(y, uid, tid, N, T, partial.Q)
    V_n = sum(abs2, xt)
    V_n > 1e-12 * max(sum(abs2, Float64.(x)), 1.0) || throw(ArgumentError(
        "regressor has no within variation (collinear with the fixed effects)"))
    beta = dot(xt, yt) / V_n
    return _score_concentration(xt, yt .- beta .* xt)
end

"""
    leverage_report(y, x, unit, time; alpha=0.05, delta=0.05) -> AdequacyReport

Diffuse-regime diagnostic. Reproduces the user's FE regression of `y` on `x` with unit
and time fixed effects via Frisch-Waugh (no re-specification), then reports the
naive / degrees-of-freedom / HC0-HC3 variance hierarchy, the two leverage
conditions the corrected theory actually needs, and whether the choice of
variance estimator materially changes inference at tolerance `delta`.

Reported design conditions (both are what the proofs use, and both are
computable before any outcome is examined):

- `lambda_n = max_i Xt_i^2 / V_n` — the self-normalized leverage of
  Assumption ass:des(ii). The CLT needs it to vanish. When it does not, the
  limit need not be Gaussian and no fixed distribution-free critical value is
  uniformly valid at the fully concentrated boundary; that
  regime is Paper A's ([`cycle_report`](@ref)), not this one.
- `max_i |H_ii - rho|` — the uniform-leverage balance condition ass:hc(ii)
  under which the HC limits below are derived. It is a strong condition: it
  holds for approximately balanced grouped designs and fails for unbalanced
  AKM-type bipartite designs, where the leave-out literature applies instead.

Verdict: FLAGGED when the asymptotic naive/HC0 over-rejection exceeds
`alpha + delta` at this design's saturation `rho` (i.e. `rho > rho_dagger`),
when significance at level `alpha` flips across the estimators on this data, or
when either design condition above is violated badly enough that the reported
limits do not apply.
"""
function leverage_report(y::AbstractVector{<:Real}, x::AbstractVector{<:Real},
                         unit::AbstractVector, time::AbstractVector;
                         alpha::Real=0.05, delta::Real=0.05,
                         controls=nothing)
    uid, tid, N, T = _integer_codes(unit, time)
    n = length(uid)
    (length(y) == n && length(x) == n) ||
        throw(ArgumentError("y, x, unit, time must have equal length"))

    partial = _partial_within_codes(x, uid, tid, N, T; controls=controls)
    xt = partial.xt
    design = _design_summary_codes(uid, tid, N, T; xt=xt)
    d_K = design.d_K
    dof = n - d_K - partial.rank - 1
    dof > 0 || throw(ArgumentError(
        "no residual degrees of freedom after fixed effects, controls, and the target regressor ($dof <= 0)"))
    yt = _partial_outcome_codes(y, uid, tid, N, T, partial.Q)
    V_n = something(design.tau_star2)
    V_n > 1e-12 * max(sum(abs2, Float64.(x)), 1.0) || throw(ArgumentError(
        "regressor has no within variation (collinear with the fixed effects)"))

    beta = dot(xt, yt) / V_n
    u = yt .- beta .* xt
    rss = sum(abs2, u)

    p_fe = _fe_leverage_diag(uid, tid, N, T)
    h_controls = size(partial.Q, 2) == 0 ? zeros(n) :
                 vec(sum(abs2, partial.Q; dims=2))
    H = p_fe .+ h_controls .+ abs2.(xt) ./ V_n
    maxH = maximum(H)
    maxH < 1 - 1e-10 || throw(ArgumentError(
        "an observation has full leverage H_ii = 1; HC2/HC3 are undefined " *
        "(the bounded-leverage condition fails)"))

    hc0 = sum(abs2.(xt) .* abs2.(u))
    hc1 = n / dof * hc0
    hc2 = sum(abs2.(xt) .* abs2.(u) ./ (1 .- H))
    hc3 = sum(abs2.(xt) .* abs2.(u) ./ (1 .- H) .^ 2)

    se_naive = sqrt(rss / n / V_n)
    se_df    = sqrt(rss / dof / V_n)
    se_hc0   = sqrt(hc0) / V_n
    se_hc1   = sqrt(hc1) / V_n
    se_hc2   = sqrt(hc2) / V_n
    se_hc3   = sqrt(hc3) / V_n

    rho = d_K / n
    z = _norminv(1 - alpha / 2)
    size_naive = _size_naive(rho, alpha)
    size_hc3 = _size_hc3(rho, alpha)
    rho_dag = _rho_dagger(alpha, delta)

    # --- the two design conditions of the corrected theory ------------------
    lambda_n = something(design.lambda_n)          # ass:des(ii)
    n_eff = something(design.n_eff)                # effective support size
    unif_gap = maximum(abs.(H .- rho))            # ass:hc(ii)

    # Hessian deflation (lem:hess(b)): V_n estimates (1-rho) * n * Qbar, so the
    # primitive residual treatment variance is recovered by dividing it out.
    Q_hat = V_n / (n * (1 - rho))

    # does significance at alpha flip across estimators on this data?
    sig = [abs(beta / se) > z for se in (se_df, se_hc0, se_hc1, se_hc2, se_hc3)]
    flip = any(s != sig[1] for s in sig)

    notes = String[
        "HC2 is the recommended default among the HC0--HC3 estimators reported here; it is leverage-adjusted but is not the Kline--Saggio--Solvsten leave-out estimator",
        "implied sizes are the conditional-homoskedastic limits of thm:hc " *
        "(omega_eff^2 = sigma^2); under heteroskedasticity the HCc limits carry " *
        "the extra factor omega^2/omega_eff^2, omega_eff^2 = (1-rho) omega^2 + rho mu",
    ]
    push!(notes, @sprintf("sample Hessian V_n = %.6g estimates (1-rho) n Qbar, not n Qbar (lem:hess(b)); the deflation-corrected primitive is Qhat = V_n/(n(1-rho)) = %.6g",
                          V_n, Q_hat))

    lam_bad = lambda_n > 0.10
    if lam_bad
        push!(notes, @sprintf("CONCENTRATION WARNING: lambda_n = max_i Xt_i^2/V_n = %.3f does not look negligible (N_eff = %.1f). The Gaussian limit requires lambda_n -> 0. Along persistently concentrated sequences the limit law is not fixed across error distributions, and at the fully concentrated boundary no fixed distribution-free critical value is uniformly valid; use the exact contrast test (cycle_report, Paper A) instead.",
                              lambda_n, n_eff))
    end
    unif_bad = unif_gap > 0.25
    if unif_bad
        push!(notes, @sprintf("uniform-leverage condition strained: max_i |H_ii - rho| = %.3f (rho = %.3f). The HC limits of thm:hc are derived under approximate balance; this design is not balanced in that sense (unbalanced bipartite designs are the leave-out literature's territory, not this module's).",
                              unif_gap, rho))
    end
    if rho > 0.1
        push!(notes, @sprintf("HC3 over-correcting regime (rho = %.3f > 0.1): HC3/V_n ->p omega_eff^2/(1-rho), so HC3 intervals are artificially conservative (implied size %.1f%%) — prefer HC2 among the reported HC estimators",
                              rho, 100 * size_hc3))
    end
    spread = maxH / minimum(H)
    if spread > 2
        push!(notes, @sprintf("leverage non-uniform (hmax/hmin = %.2f): HC1 is unreliable here; prefer HC2 or a design-appropriate leave-out estimator",
                              spread))
    end
    if flip
        push!(notes, "significance at level alpha flips across variance " *
                     "estimators — inference is estimator-dependent; use HC2/LO")
    end

    verdict = (size_naive - alpha > delta || flip || lam_bad || unif_bad) ?
              :FLAGGED : :CERTIFIED

    score = _score_concentration(xt, u)
    if isfinite(score.lambda_score)
        push!(notes, @sprintf("realized score diagnostic (Paper A, Remark rem:diag): lambda_score = %.4f, N_eff,score = %.1f. This is a one-realization warning statistic, not by itself a consistent population concentration estimate.",
                              score.lambda_score, score.n_eff_score))
    end

    statistic = (beta=beta, se_naive=se_naive, se_df=se_df, se_hc0=se_hc0,
                 se_hc1=se_hc1, se_hc2=se_hc2, se_hc3=se_hc3,
                 t_hc2=beta / se_hc2, max_leverage=maxH,
                 max_fe_leverage=maximum(p_fe), leverage_spread=spread,
                 lambda_n=lambda_n, n_eff=n_eff, uniform_leverage_gap=unif_gap,
                 lambda_score=score.lambda_score, H_score=score.H_score,
                 n_eff_score=score.n_eff_score,
                 score_lambda_n=score.lambda_score, score_H_n=score.H_score,
                 score_n_eff=score.n_eff_score,
                 control_rank=partial.rank,
                 V_n=V_n, Q_hat=Q_hat, implied_size_hc3=size_hc3)

    return AdequacyReport(:leverage, design, statistic, nothing, nothing,
                          rho_dag, size_naive, verdict, Float64(alpha),
                          Float64(delta), notes)
end
