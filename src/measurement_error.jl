# =============================================================================
# Module B — measurement-error adequacy (spec §4; Paper B paper_b_fe_eiv_JoE.tex)
#
# Formulas from Paper B:
#   within reliability      lambda_hat = 1 - a_hat/tau*2, a_hat = mean(sigma_nu^2)(n - d_K)
#   feasible non-centrality |eta| = (|beta0|/sigma)(1 - lambda) sqrt(tau*2)   (Cor. cor-feasible)
#   corrected pilot         beta0_corr = beta*/lambda_hat                     (Prop. prop-pilot)
#   pilot sampling se       se(beta0_corr) = sigma/(lambda sqrt(tau*2))       (Cor. cor-slope)
#   threshold               eta†(alpha, delta): EXACT INVERSION of the size,
#                           quadratic closed form as companion — never linear  (Rem. rem-exact-cv)
#   breakdown reliability   lambda† = t*/(t* + eta†), t* = |beta*|sqrt(tau*2)/sigma
#                           (fixed point; Def. def-breakdown) — pilot-free, and
#                           equivalent to the point verdict: lambda >= lambda†
#   cluster layer           psi_hat rescales |eta| and t* by 1/sqrt(psi)      (Rem. rem-cluster)
#   implied size            exact non-central size at eta                     (eq-cv-exact)
# =============================================================================

"""
    reliability_from_interval(codelow, codehigh) -> Vector{Float64}

Per-observation measurement-error SDs from published credible-interval bounds
(V-Dem convention: the interval brackets one posterior SD, so
`sigma_nu = (codehigh - codelow)/2`; Paper B §sec-app-vdem).
"""
function reliability_from_interval(codelow::AbstractVector{<:Real},
                                   codehigh::AbstractVector{<:Real})
    length(codelow) == length(codehigh) ||
        throw(ArgumentError("codelow and codehigh must have equal length"))
    return (Float64.(codehigh) .- Float64.(codelow)) ./ 2
end

"""
    reliability_from_ratio(r, within_sd) -> Float64

Measurement-error SD implied by an external reliability ratio `r` (PSID-style
validation studies) and the within-SD of the observed regressor:
`sigma_nu = sqrt((1-r)/r) * within_sd`.
"""
function reliability_from_ratio(r::Real, within_sd::Real)
    0 < r <= 1 || throw(ArgumentError("reliability ratio must be in (0, 1]"))
    return sqrt((1 - r) / r) * within_sd
end

"""
    breakdown_reliability(beta_star, sigma, tau_star2;
                          alpha=0.05, delta=0.05, psi=1.0) -> Float64

Self-consistent breakdown reliability (Paper B Def. def-breakdown): the fixed
point lambda = lambda†(lambda) when the corrected pilot `beta*/lambda` is
evaluated at the reliability being solved for. Closed form
`lambda† = t*/(t* + eta†)` with `t* = |beta*| sqrt(tau*2)/sigma` — the
specification's conventional t-statistic, so no noise input is needed. Under
cluster-robust standardization pass the variance-inflation factor `psi`
(Remark rem-cluster): t* is deflated by `sqrt(psi)`. Uses the exact-inversion
root eta†.
"""
function breakdown_reliability(beta_star::Real, sigma::Real, tau_star2::Real;
                               alpha::Real=0.05, delta::Real=0.05, psi::Real=1.0)
    beta_star == 0 && return 0.0
    psi > 0 || throw(ArgumentError("psi must be positive"))
    eta_dag = _eta_dagger(alpha, delta)
    t_star = abs(beta_star) * sqrt(tau_star2) / (sigma * sqrt(psi))
    return t_star / (t_star + eta_dag)
end

"""
    eiv_adequacy(y, x, unit, time; <noise input>, alpha=0.05, delta=0.05,
                 gamma=0.05, pilot=:conservative,
                 cluster=:iid, psi=nothing) -> AdequacyReport

Module B diagnostic (Paper B). Reproduces the user's FE regression of `y` on
the observed regressor `x` via Frisch-Waugh, then certifies whether naive
inference is size-controlled under classical measurement error.

Noise input — exactly one of:
- `sigma_nu`     : per-observation (or scalar) measurement-error SD
- `codelow`, `codehigh` : V-Dem-style posterior interval bounds
- `reliability`  : the within reliability lambda_hat directly (external estimate
                   of the WITHIN-transformed regressor's reliability)

`pilot` (correct-by-default honesty machinery, spec §4):
- `:conservative` (default) — formal certificate: verdict evaluated at the upper
  `1-gamma` confidence bound of the corrected pilot (protocol step 6)
- `:point` — corrected point pilot; descriptive verdict (protocol step 4)
- `:naive` — attenuated `beta*` pilot; ANTI-CONSERVATIVE, for comparison only

`cluster` (Paper B, Remark rem-cluster — the standardization of the t-test):
- `:iid` (default) — homoskedastic-i.i.d. standard errors (Theorem thm-noncentral)
- `:crve` — by-unit cluster-robust (Arellano CR1) variance-inflation `psi_hat`;
  `|eta|` and the breakdown are deflated by `sqrt(psi_hat)`
- `:ar1` — parametric within-unit AR(1) variance inflation (Paper C route (i))
- or pass `psi = <value>` to supply your own variance-inflation factor

The breakdown reliability is the fixed point `lambda† = t*/(t* + eta†)` with
`t* = |beta*| sqrt(tau*2)/sigma` (Paper B Def. def-breakdown): pilot-free, and
for the point pilot `lambda_hat >= lambda†` is exactly the verdict criterion.

    eiv_adequacy(; beta_star, sigma, tau_star2, n, d_K,
                 reliability | sigma_nu2, N=0, T=0, psi=nothing,
                 kwargs...) -> AdequacyReport

Summary form for users supplying regression output directly (no raw data;
`cluster` presets are unavailable here — pass `psi` if clustering).
"""
function eiv_adequacy(y::AbstractVector{<:Real}, x::AbstractVector{<:Real},
                      unit::AbstractVector, time::AbstractVector;
                      sigma_nu::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
                      codelow::Union{Nothing,AbstractVector{<:Real}}=nothing,
                      codehigh::Union{Nothing,AbstractVector{<:Real}}=nothing,
                      reliability::Union{Nothing,Real}=nothing,
                      alpha::Real=0.05, delta::Real=0.05, gamma::Real=0.05,
                      pilot::Symbol=:conservative,
                      cluster::Symbol=:iid, psi::Union{Nothing,Real}=nothing)
    cluster in (:iid, :crve, :ar1) ||
        throw(ArgumentError("cluster must be :iid, :crve, or :ar1 (or pass psi=...)"))
    uid, tid, N, T = _integer_codes(unit, time)
    n = length(uid)
    (length(y) == n && length(x) == n) ||
        throw(ArgumentError("y, x, unit, time must have equal length"))
    d_K, ncomp = fe_dimension(uid, tid, N, T)
    dof = n - d_K - 1
    dof > 0 || throw(ArgumentError("no residual degrees of freedom"))

    xt = _twoway_demean(Float64.(x), uid, tid, N, T)
    yt = _twoway_demean(Float64.(y), uid, tid, N, T)
    tau_star2 = sum(abs2, xt)
    tau_star2 > 1e-12 * max(sum(abs2, Float64.(x)), 1.0) ||
        throw(ArgumentError("regressor has no within variation"))
    beta_star = dot(xt, yt) / tau_star2
    u = yt .- beta_star .* xt
    sigma = sqrt(sum(abs2, u) / dof)

    # noise pilot: exactly one source
    if codelow !== nothing || codehigh !== nothing
        (codelow !== nothing && codehigh !== nothing) ||
            throw(ArgumentError("supply both codelow and codehigh"))
        sigma_nu === nothing || throw(ArgumentError("multiple noise inputs"))
        sigma_nu = reliability_from_interval(codelow, codehigh)
    end
    nsources = (sigma_nu !== nothing) + (reliability !== nothing)
    nsources == 1 || throw(ArgumentError(
        "supply exactly one noise input: sigma_nu, (codelow, codehigh), or reliability"))
    if reliability !== nothing
        0 < reliability <= 1 ||
            throw(ArgumentError("reliability must be in (0, 1]"))
        lambda = Float64(reliability)
    else
        s2 = sigma_nu isa Real ? abs2(Float64(sigma_nu)) :
             mean(abs2, Float64.(sigma_nu))
        a_hat = s2 * (n - d_K)
        lambda = 1 - a_hat / tau_star2
    end

    extra = String[]
    length(unique(x)) <= 2 && push!(extra,
        "binary treatment detected: errors in binary treatments are MISCLASSIFICATION (nonclassical); this classical-EIV threshold does not apply (Paper B §5.3)")

    # cluster variance-inflation factor psi_hat (Paper B, Remark rem-cluster)
    rho_ar1 = nothing
    if psi !== nothing
        psi_hat = Float64(psi)
    elseif cluster === :crve
        psi_hat = _psi_driven(xt, u, uid, tau_star2, sigma^2, N)
    elseif cluster === :ar1
        rho_ar1 = _rho_ar1(u, uid, tid)
        psi_hat = _psi_parametric(xt, uid, tid, rho_ar1; kind=:ar1)
    else
        psi_hat = 1.0
    end

    design = DesignSummary(n, N, T, d_K, d_K / n, ncomp, tau_star2)
    return _eiv_core(design, beta_star, sigma, tau_star2, lambda;
                     alpha=alpha, delta=delta, gamma=gamma, pilot=pilot,
                     psi_hat=psi_hat, rho_ar1=rho_ar1, extra_notes=extra)
end

function eiv_adequacy(; beta_star::Real, sigma::Real, tau_star2::Real,
                      n::Integer, d_K::Integer,
                      reliability::Union{Nothing,Real}=nothing,
                      sigma_nu2::Union{Nothing,Real}=nothing,
                      N::Integer=0, T::Integer=0,
                      alpha::Real=0.05, delta::Real=0.05, gamma::Real=0.05,
                      pilot::Symbol=:conservative,
                      psi::Union{Nothing,Real}=nothing)
    (reliability !== nothing) + (sigma_nu2 !== nothing) == 1 ||
        throw(ArgumentError("supply exactly one of reliability or sigma_nu2"))
    lambda = if reliability !== nothing
        0 < reliability <= 1 ||
            throw(ArgumentError("reliability must be in (0, 1]"))
        Float64(reliability)
    else
        1 - sigma_nu2 * (n - d_K) / tau_star2
    end
    design = DesignSummary(n, N, T, d_K, d_K / n, 1, Float64(tau_star2))
    return _eiv_core(design, Float64(beta_star), Float64(sigma),
                     Float64(tau_star2), lambda; alpha=alpha, delta=delta,
                     gamma=gamma, pilot=pilot,
                     psi_hat=(psi === nothing ? 1.0 : Float64(psi)),
                     extra_notes=String[])
end

function _eiv_core(design::DesignSummary, beta_star::Float64, sigma::Float64,
                   tau_star2::Float64, lambda::Float64;
                   alpha::Real, delta::Real, gamma::Real, pilot::Symbol,
                   psi_hat::Float64=1.0, rho_ar1::Union{Nothing,Float64}=nothing,
                   extra_notes::Vector{String})
    pilot in (:conservative, :point, :naive) ||
        throw(ArgumentError("pilot must be :conservative, :point, or :naive"))
    psi_hat > 0 || throw(ArgumentError("psi must be positive"))
    eta_dag = _eta_dagger(alpha, delta)
    sqpsi = sqrt(psi_hat)
    notes = copy(extra_notes)

    # fixed-point breakdown lambda† = t*/(t* + eta†), t* = |beta*|sqrt(tau*2)/sigma
    # (Paper B Def. def-breakdown) — the pilot-free, self-consistent evaluation;
    # under clustering t* is rescaled by 1/sqrt(psi)
    t_star = abs(beta_star) * sqrt(tau_star2) / sigma
    breakdown = (t_star / sqpsi) / (t_star / sqpsi + eta_dag)

    if lambda <= 0
        push!(notes, @sprintf("implied noise exceeds ALL residual within variation (lambda_hat = %.3f <= 0): the noise pilot may be misscaled, or attenuation is total; exact size = 1", lambda))
        statistic = (lambda_hat=lambda, noise_ratio=Inf, beta_star=beta_star,
                     sigma=sigma, t_star=t_star, psi_hat=psi_hat)
        return AdequacyReport(:measurement_error, design, statistic, Inf,
                              eta_dag, breakdown, 1.0, :FLAGGED, Float64(alpha),
                              Float64(delta), notes)
    end

    beta_corr = beta_star / lambda
    se_corr = sigma / (lambda * sqrt(tau_star2))
    b_pilot = pilot === :naive ? abs(beta_star) : abs(beta_corr)
    eta_point = (b_pilot / sigma) * (1 - lambda) * sqrt(tau_star2) / sqpsi
    eta_upper = pilot === :conservative ?
        ((abs(beta_corr) + _norminv(1 - gamma) * se_corr) / sigma) *
        (1 - lambda) * sqrt(tau_star2) / sqpsi : nothing
    eta_used = pilot === :conservative ? eta_upper : eta_point

    verdict = eta_used <= eta_dag ? :CERTIFIED : :FLAGGED
    implied_size = _noncentral_size(eta_point, alpha)

    if pilot === :conservative
        push!(notes, @sprintf("formal certificate: verdict uses the upper %.0f%% confidence bound of the corrected pilot, |eta|_ub = %.3f (Paper B, protocol step 6); implied size shown is at the point pilot", 100 * (1 - gamma), eta_upper))
    elseif pilot === :point
        push!(notes, "point diagnostic (descriptive): corrected pilot at its point estimate — not a formally size-controlled certificate (Paper B, Remark rem-corr-noise)")
    else
        push!(notes, "ANTI-CONSERVATIVE naive pilot (attenuated beta*) — for comparison only; understates |eta| by the factor lambda (Paper B, Prop. prop-pilot(i))")
    end
    pilot !== :naive && push!(notes,
        "corrected pilot beta*/lambda_hat is not a consistent point estimate under weak information; reported with its sampling band (Paper B, Cor. cor-slope)")
    eta_point > 1 && push!(notes,
        "far from the threshold (|eta| > 1): the local quadratic approximation is uninformative here; the verdict uses exact inversion (Paper B, Remark rem-exact-cv)")
    lambda < 1 && push!(notes, @sprintf("power tax: local power slope attenuated by sqrt(lambda) = %.2f (Paper B, Prop. prop-power)", sqrt(lambda)))
    if psi_hat != 1.0
        dir = psi_hat > 1 ?
            "clustering deflates the measured distortion on THIS design; the iid verdict is an upper bound on the alarm" :
            "clustering WORSENS the distortion on THIS design (psi < 1)"
        push!(notes, @sprintf("cluster-robust standardization: variance-inflation psi_hat = %.3f rescales |eta| and the breakdown by 1/sqrt(psi) = %.3f (Paper B, Remark rem-cluster): %s", psi_hat, 1 / sqpsi, dir))
    end

    statistic_base = (lambda_hat=lambda, noise_ratio=(1 - lambda) / lambda,
                      beta_star=beta_star, beta_corr=beta_corr,
                      se_beta_corr=se_corr, sigma=sigma, t_star=t_star,
                      psi_hat=psi_hat,
                      eta_quad_threshold=_eta_quad(alpha, delta))
    rho_ar1 !== nothing &&
        (statistic_base = merge(statistic_base, (rho_ar1=rho_ar1,)))
    statistic = eta_upper === nothing ? statistic_base :
                merge(statistic_base, (eta_upper=eta_upper,))

    return AdequacyReport(:measurement_error, design, statistic, eta_point,
                          eta_dag, breakdown, implied_size, verdict,
                          Float64(alpha), Float64(delta), notes)
end
