# =============================================================================
# Module A — leverage / variance diagnostics (spec §3; Paper A: arXiv 2607.05215)
#
# Formulas from Paper A:
#   full-regression leverage  H_ii = (P_K)_ii + Xt_i^2 / (X'M_K X)     (eq. 2.3)
#   HC0 = sum Xt_i^2 u_i^2                HC1 = n/(n-d_K-1) * HC0      (§3.2)
#   HC2/LO = sum Xt_i^2 u_i^2/(1-H_ii)    HC3 = sum Xt_i^2 u_i^2/(1-H_ii)^2
#   naive sigma^2 = u'u/n   vs   CJN sigma^2 = u'u/(n-d_K-1)
#   asymptotic sizes (Thm 3.1, Cor 3.1): naive/HC0 2(1-Phi(z sqrt(1-rho))),
#   HC3 2(1-Phi(z/sqrt(1-rho))); HC2/LO and CJN exact.
#   HC3 over-correction regime: rho > 0.1 (Paper A §6); HC1 vs LO divergence
#   under non-uniform leverage (Remark 3.4).
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
    Gp = pinv(Symmetric(G))
    p = Vector{Float64}(undef, length(uid))
    for k in eachindex(uid)
        u, t = uid[k], N + tid[k]
        p[k] = Gp[u, u] + 2 * Gp[u, t] + Gp[t, t]
    end
    return p
end

"""
    leverage_report(y, x, unit, time; alpha=0.05, delta=0.05) -> AdequacyReport

Module A diagnostic (Paper A). Reproduces the user's FE regression of `y` on
`x` with unit and time fixed effects via Frisch-Waugh (no re-specification),
then reports the HC0-HC3 / naive / CJN variance hierarchy, the leverage
diagnostics, and whether the variance-estimator choice materially changes
inference at tolerance `delta`.

Verdict: FLAGGED when the asymptotic HC0/naive over-rejection exceeds
`alpha + delta` at this design's saturation `rho` (i.e. `rho > rho†`), or when
significance at level `alpha` flips across the estimators on this data.
"""
function leverage_report(y::AbstractVector{<:Real}, x::AbstractVector{<:Real},
                         unit::AbstractVector, time::AbstractVector;
                         alpha::Real=0.05, delta::Real=0.05)
    uid, tid, N, T = _integer_codes(unit, time)
    n = length(uid)
    (length(y) == n && length(x) == n) ||
        throw(ArgumentError("y, x, unit, time must have equal length"))

    d_K, ncomp = fe_dimension(uid, tid, N, T)
    dof = n - d_K - 1
    dof > 0 || throw(ArgumentError(
        "no residual degrees of freedom (n - d_K - 1 = $dof <= 0)"))

    xt = _twoway_demean(Float64.(x), uid, tid, N, T)
    yt = _twoway_demean(Float64.(y), uid, tid, N, T)
    tau_star2 = sum(abs2, xt)
    tau_star2 > 1e-12 * max(sum(abs2, Float64.(x)), 1.0) || throw(ArgumentError(
        "regressor has no within variation (collinear with the fixed effects)"))

    beta = dot(xt, yt) / tau_star2
    u = yt .- beta .* xt
    rss = sum(abs2, u)

    p_fe = _fe_leverage_diag(uid, tid, N, T)
    H = p_fe .+ abs2.(xt) ./ tau_star2
    maxH = maximum(H)
    maxH < 1 - 1e-10 || throw(ArgumentError(
        "an observation has full leverage H_ii = 1; HC2/HC3 are undefined " *
        "(degenerate cell — Paper A's bounded-leverage condition fails)"))

    hc0 = sum(abs2.(xt) .* abs2.(u))
    hc1 = n / dof * hc0
    hc2 = sum(abs2.(xt) .* abs2.(u) ./ (1 .- H))
    hc3 = sum(abs2.(xt) .* abs2.(u) ./ (1 .- H) .^ 2)

    se_naive = sqrt(rss / n / tau_star2)
    se_cjn   = sqrt(rss / dof / tau_star2)
    se_hc0   = sqrt(hc0) / tau_star2
    se_hc1   = sqrt(hc1) / tau_star2
    se_hc2   = sqrt(hc2) / tau_star2
    se_hc3   = sqrt(hc3) / tau_star2

    rho = d_K / n
    z = _norminv(1 - alpha / 2)
    size_hc0 = _size_hc0(rho, alpha)
    size_hc3 = _size_hc3(rho, alpha)
    rho_dag = _rho_dagger(alpha, delta)

    # does significance at alpha flip across estimators on this data?
    sig = [abs(beta / se) > z for se in (se_cjn, se_hc0, se_hc1, se_hc2, se_hc3)]
    flip = any(s != sig[1] for s in sig)

    notes = String[
        "leave-one-out (HC2) is the recommended default for saturated FE " *
        "(Paper A, Remark 3.6)",
        "implied sizes use the homoskedastic / design-balanced limits of " *
        "Theorem 3.3 (sigma_eff^2 = omega^2)",
    ]
    if rho > 0.1
        push!(notes, @sprintf("HC3 over-correcting regime (rho = %.3f > 0.1): HC3 intervals are artificially conservative (implied size %.1f%%) — re-run with HC2/LO (Paper A §6)",
                              rho, 100 * size_hc3))
    end
    spread = maxH / minimum(H)
    if spread > 2
        push!(notes, @sprintf("leverage non-uniform (hmax/hmin = %.2f): HC1 is unreliable here; prefer HC2/LO (Paper A, Remark 3.4)",
                              spread))
    end
    if flip
        push!(notes, "significance at level alpha flips across variance " *
                     "estimators — inference is estimator-dependent; use HC2/LO")
    end
    max_x_share = maximum(abs2, xt) / tau_star2
    if max_x_share > 0.1
        push!(notes, @sprintf("treatment leverage-ratio condition strained: one observation carries %.0f%% of the within variation",
                              100 * max_x_share))
    end

    verdict = (size_hc0 - alpha > delta || flip) ? :FLAGGED : :CERTIFIED

    design = DesignSummary(n, N, T, d_K, rho, ncomp, tau_star2)
    statistic = (beta=beta, se_naive=se_naive, se_cjn=se_cjn, se_hc0=se_hc0,
                 se_hc1=se_hc1, se_hc2=se_hc2, se_hc3=se_hc3,
                 t_hc2=beta / se_hc2, max_leverage=maxH,
                 max_fe_leverage=maximum(p_fe), leverage_spread=spread,
                 max_x_share=max_x_share, implied_size_hc3=size_hc3)

    return AdequacyReport(:leverage, design, statistic, nothing, nothing,
                          rho_dag, size_hc0, verdict, Float64(alpha),
                          Float64(delta), notes)
end
