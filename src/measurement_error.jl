# =============================================================================
# Measurement-error adequacy
# SOURCE: “Breakdown Reliability for Saturated Fixed-Effect Inference”.
#
# Core formulas:
#   within reliability      lambda_hat = 1 - a_hat/tau*2
#                           scalar sigma_nu: a_hat = sigma_nu^2(n-d_K)
#                           vector sigma_nu: a_hat = sum_i(1-h_ii)sigma_nu_i^2
#   feasible non-centrality |eta| = (|beta0|/s)(1 - lambda) sqrt(tau*2)     (cor-feasible)
#   corrected pilot         beta0_corr = beta*/lambda_hat                   (prop-pilot)
#   pilot sampling se       se(beta0_corr) = s_CR / (lambda sqrt(tau*2))    (rem-plugin)
#                           s_CR = sigma sqrt(psi), NOT the i.i.d. sigma of cor-slope:
#                           under clustering Psi_n ->p psi sigma^2 tau*2, so the pilot's
#                           limit is N(0, psi sigma^2/(lambda^2 tau*2)). This module has
#                           always used s_CR; the manuscript's rem-plugin cited the
#                           i.i.d. variance until the 2026-07-31 audit corrected it.
#   threshold               eta†(alpha, delta): EXACT INVERSION of the size,
#                           quadratic closed form as companion — never linear (rem-exact-cv)
#   breakdown reliability   lambda† = t*/(t* + eta†), t* = |beta*|sqrt(tau*2)/s
#                           (fixed point; def-breakdown) — pilot-free
#   certified breakdown     lambda†_gamma = (t* + z_(1-gamma_beta)) /
#                           (t* + z_(1-gamma_beta) + eta†)
#   implied size            exact non-central size at eta                    (eq-cv-exact)
#
# CLUSTER LAYER — now derived, not asserted (sec-cluster). What changed:
#
#   s_CR^2 := sigma*_CJN^2 * psi_hat = V^sc_CR / tau*2         (cor-cluster-feasible)
#     an ALGEBRAIC IDENTITY: sigma*_CJN^2 cancels exactly. Both sigma*_CJN^2 and
#     psi_hat are individually INCONSISTENT under cluster dependence
#     (psi_hat ->p psi sigma^2/varsigma^2), and the two errors are reciprocal, so
#     their product is consistent for psi sigma^2. Every scale in this module is
#     therefore s_CR, not sigma_hat — including the pilot standard error, which
#     the pre-correction code got wrong (it used the i.i.d. sigma_hat and so
#     UNDERSTATED the certificate's width by sqrt(psi)).
#
#   thm-cluster            T^CR => N(eta_CR, 1), eta_CR = eta/sqrt(psi)
#   cor-cluster-feasible(c) lambda†_CR = t^CR/(t^CR + eta†) with t^CR = t*/sqrt(psi)
#     EXACTLY the reported cluster-robust t: the cluster-robust diagnostic is the
#     i.i.d. diagnostic run on the reported cluster-robust t-statistic.
#   lem-crve               the CRVE meat is consistent UNCORRECTED. The conventional
#     factor G/(G-1) * (n-1)/(n-K) converges to 1/(1-rho), NOT 1, so it
#     OVER-corrects whenever rho is non-negligible; the (n-1)/(n-K) half must be
#     omitted. Default software applies it — see `small_sample` below.
#   ass-cluster(iv)/lem-nest  PROJECTION COMPATIBILITY. The condition with no
#     i.i.d. counterpart: the FE projection must not distort the cluster
#     loadings. lem-nest(b)'s bound varpi_n * max_g A_g is unconditional; the
#     familiar reduction to d_ne/G -> 0 is NOT — it needs the balance conditions
#     (N1) spectral, lambda_min^+(Lambda_n) ~ n/d_ne, and (N2) energy,
#     max_g A_g = O(tau*2/G). Read `ratio_ne` and `max_energy` together, and use
#     [`projection_compatibility`](@ref) to evaluate the assumption itself.
#     See [`cluster_diagnostics`](@ref).
#   rem-psi-sign           the sign of psi-1 is NOT free and NOT guessable. An
#     equicorrelated component at a level that is itself a fixed effect is
#     annihilated exactly; with a serially dependent error but a within-cluster
#     serially independent regressor the surviving terms are predominantly
#     negative and psi < 1 — clustering then makes the diagnostic MORE alarming.
#     What drives psi above one is persistence in regressor and error TOGETHER.
#     Read the direction off psi_hat; never assume it.
#
# POINT PASS vs FORMAL CERTIFICATE (rem-plugin, prop-certificate). The point
# plug-in is DESCRIPTIVE: for beta0 != 0, eta_hat_CR => (|B|/|beta0|) |eta_CR|
# with B ~ beta0 + N(0, psi sigma^2/(lambda^2 tau*2)) — a nondegenerate random
# multiple of the target, with no concentration under weak information. It is
# reported as :POINT_PASS, never :CERTIFIED. The formal certificate evaluates
# certified breakdown replaces |t| by |t| + z_(1-gamma_beta). With a supplied
# reliability lower bound of coverage error gamma_lambda, false certification
# is at most gamma_beta + gamma_lambda, without an independence assumption.
# The public keyword `gamma` is retained for compatibility and denotes
# gamma_beta. A noisy reliability point estimate must not be treated as known.
# =============================================================================

"""
    reliability_from_interval(codelow, codehigh) -> Vector{Float64}

Per-observation measurement-error SDs from published interval bounds, computed
as `sigma_nu = (codehigh - codelow)/2`. This helper is appropriate only when
the interval half-width is substantively calibrated as one measurement-error
SD; it must not be applied mechanically to highest-posterior-density bounds.
"""
function reliability_from_interval(codelow::AbstractVector{<:Real},
                                   codehigh::AbstractVector{<:Real})
    length(codelow) == length(codehigh) ||
        throw(ArgumentError("codelow and codehigh must have equal length"))
    all(codehigh .>= codelow) || throw(ArgumentError(
        "every codehigh value must be at least its codelow value"))
    return (Float64.(codehigh) .- Float64.(codelow)) ./ 2
end

"""
    reliability_from_ratio(r, within_sd; scale=:observed) -> Float64

Measurement-error SD implied by an external reliability ratio `r` from a
validation or repeat-measurement study. With the documented default
`scale=:observed`, `within_sd`
is the SD of the observed regressor and
`sigma_nu = sqrt(1-r) * within_sd`. Use `scale=:signal` only when `within_sd` is
the latent signal SD; then `sigma_nu = sqrt((1-r)/r) * within_sd`.
"""
function reliability_from_ratio(r::Real, within_sd::Real;
                                scale::Symbol=:observed)
    0 < r <= 1 || throw(ArgumentError("reliability ratio must be in (0, 1]"))
    within_sd >= 0 || throw(ArgumentError("within_sd must be non-negative"))
    scale in (:observed, :signal) || throw(ArgumentError(
        "scale must be :observed or :signal"))
    factor = scale === :observed ? sqrt(1 - r) : sqrt((1 - r) / r)
    return factor * within_sd
end

"""
    reliability_from_repeats(first, second; method=:covariance) -> Float64

Estimate the reliability of `first` from two measurements of the same latent
regressor after applying the **identical projection and complete-case sample**
to both. The default covariance estimator is

    cov(first, second) / var(first)

and identifies the first measurement's reliability when the two classical
reporting errors are uncorrelated; it does not require equal error variances.
Set `method=:equal_variance` for

    1 - var(first - second) / (2var(first)),

which additionally imposes equal reporting-error variances. Correlation across
reporting errors invalidates both formulas and must be modeled or examined as a
sensitivity. The estimate is returned without truncation so assumption or
sampling failures remain visible; [`eiv_adequacy`](@ref) requires a reliability
in `(0, 1]`.
"""
function reliability_from_repeats(first_measure::AbstractVector{<:Real},
                                  second_measure::AbstractVector{<:Real};
                                  method::Symbol=:covariance)
    length(first_measure) == length(second_measure) || throw(ArgumentError(
        "first_measure and second_measure must have equal length"))
    length(first_measure) >= 2 || throw(ArgumentError(
        "repeated measurements must contain at least two observations"))
    method in (:covariance, :equal_variance) || throw(ArgumentError(
        "method must be :covariance or :equal_variance"))
    x = Float64.(first_measure)
    z = Float64.(second_measure)
    all(isfinite, x) && all(isfinite, z) || throw(ArgumentError(
        "repeated measurements must be finite and use a common complete-case sample"))
    x .-= mean(x)
    z .-= mean(z)
    xx = sum(abs2, x)
    xx > 0 || throw(ArgumentError("first_measure has zero variance"))
    if method === :covariance
        return dot(x, z) / xx
    end
    return 1 - sum(abs2, x .- z) / (2xx)
end

"""
    cluster_diagnostics(xt, cluster, fe_levels; tau_star2) -> NamedTuple

Checkable conditions behind the cluster-robust layer (Assumption ass-cluster and
Lemma lem-nest). All are computable from the design alone, before any outcome is
examined. Returns

- `G`          : number of clusters (ass-cluster(ii) needs `G -> infinity`)
- `max_size`   : largest cluster, `nbar` (ass-cluster(ii) needs it bounded; see
                 rem-clustersize for what must be strengthened to let it grow)
- `max_energy` : `max_g A_g / tau*2`, the no-dominant-cluster condition, and
                 also the balance condition (N2) below
- `d_ne`       : number of fixed effects that cut ACROSS clusters
- `ratio_ne`   : `d_ne / G` — lem-nest(b)'s shortcut for projection
                 compatibility. Must be small, but see the caveat below.
- `nested`     : which FE dimensions are nested in clusters

`fe_levels` is a vector of `(name, ids)` pairs, one per fixed-effect dimension.
A dimension is nested when every one of its cells lies inside a single cluster,
in which case lem-nest(a) gives `M a^(g) = a^(g)` exactly and that dimension
contributes nothing to `d_ne`.

!!! warning "`ratio_ne` alone carries no warrant"
    Lemma lem-nest(b) bounds projection compatibility by `varpi_n * max_g A_g`,
    which is unconditional. The reduction of that bound to `d_ne/G -> 0` is
    not: it holds only under two balance conditions.

    - **(N1) spectral**: `lambda_min^+(Lambda_n) ~ n/d_ne`. Exact in a balanced
      two-way panel, where `Lambda_n = G (I_T - 11'/T)`, but equal cell counts
      alone do not control the smallest non-zero eigenvalue of the residualized
      Gram matrix. Not computed here.
    - **(N2) energy**: `max_g A_g = O(tau*2/G)`, returned as `max_energy`. If
      the residualized signal concentrates in `sqrt(G)` clusters then
      `max_energy ~ G^-1/2` and the requirement becomes `d_ne/sqrt(G) -> 0`.

    Read `ratio_ne` and `max_energy` together. When they disagree, or when (N1)
    is in doubt, call [`projection_compatibility`](@ref), which evaluates
    `sum_g ||M a^(g) - a^(g)||^2 / tau*2` directly and needs neither condition.

In the lead V-Dem application country effects are nested in country clusters
while the 59 year effects are not, against `G = 163`, so `ratio_ne ~ 0.36` and
the direct diagnostic is load-bearing. The repeated-report twins application
samples independent pairs and therefore uses the baseline i.i.d. theorem; it
does not invoke this cluster condition or a `psi` rescaling.
"""
function cluster_diagnostics(xt::AbstractVector{<:Real}, cluster::AbstractVector,
                             fe_levels::AbstractVector; tau_star2::Real)
    n = length(xt)
    length(cluster) == n ||
        throw(ArgumentError("cluster must have the same length as the data"))
    cid = _codes(cluster)
    G = maximum(cid)
    sizes = zeros(Int, G)
    energy = zeros(Float64, G)
    for k in 1:n
        sizes[cid[k]] += 1
        energy[cid[k]] += abs2(xt[k])
    end

    nested = String[]
    d_ne = 0
    for (name, ids) in fe_levels
        length(ids) == n ||
            throw(ArgumentError("fixed-effect ids for $name must have length n"))
        fcode = _codes(ids)
        # nested iff every level of this FE dimension sits in exactly one cluster
        seen = Dict{Int,Int}()
        is_nested = true
        for k in 1:n
            f = fcode[k]
            c = get(seen, f, 0)
            if c == 0
                seen[f] = cid[k]
            elseif c != cid[k]
                is_nested = false
                break
            end
        end
        if is_nested
            push!(nested, String(name))
        else
            d_ne += length(unique(fcode))
        end
    end

    return (G=G, max_size=maximum(sizes), min_size=minimum(sizes),
            max_energy=maximum(energy) / tau_star2,
            d_ne=d_ne, ratio_ne=d_ne / G, nested=nested)
end

"""
    projection_compatibility(xt, cluster, unit, time;
                             tau_star2=sum(abs2, xt), controls=nothing,
                             tol=1e-10, maxit=10_000,
                             max_cells=5_000_000) -> NamedTuple

Projection compatibility (Assumption ass-cluster(iv)), evaluated directly:

    sum_g ||M a^(g) - a^(g)||^2 / tau*2

for a two-way (unit and time) fixed-effect design, where `a^(g)` is the
residualized regressor restricted to cluster `g`. If `controls` are supplied,
`M` also removes their within-FE column space.

Unlike the `ratio_ne` shortcut of [`cluster_diagnostics`](@ref), this needs
neither of the balance conditions (N1) and (N2) of lem-nest(b): it is the
quantity the assumption actually names. Use it when `ratio_ne` and `max_energy`
point in different directions, when the panel is far from balanced, or whenever
the cluster-robust column is load-bearing.

Under lem-nest(a) the ratio is exactly zero when every fixed-effect cell sits
inside one cluster, so a non-zero value is entirely the work of the fixed
effects that cut across clusters.

Cost is one alternating-projection sweep per cluster. `max_cells` guards the
`n * G` work: above it the function returns `ratio = NaN` rather than running.
The guard matches the R package's so both report the same number.

Returns `(ratio, G, n, cells)`.
"""
function projection_compatibility(xt::AbstractVector{<:Real},
                                  cluster::AbstractVector,
                                  unit::AbstractVector, time::AbstractVector;
                                  tau_star2::Real=sum(abs2, xt),
                                  controls=nothing,
                                  tol::Real=1e-10, maxit::Integer=10_000,
                                  max_cells::Real=5_000_000)
    n = length(xt)
    (length(cluster) == n && length(unit) == n && length(time) == n) ||
        throw(ArgumentError("xt, cluster, unit and time must have equal length"))
    tau_star2 > 0 || throw(ArgumentError("tau_star2 must be positive"))

    cid = _codes(cluster)
    G = maximum(cid)
    cells = float(n) * G
    cells > max_cells && return (ratio=NaN, G=G, n=n, cells=cells)

    uid, tid, N, T = _integer_codes(unit, time)
    x = Float64.(xt)
    partial = _partial_within_codes(x, uid, tid, N, T; controls=controls)
    Q = partial.Q
    total = 0.0
    a = zeros(Float64, n)
    for g in 1:G
        fill!(a, 0.0)
        @inbounds for k in 1:n
            cid[k] == g && (a[k] = x[k])
        end
        Ma = _twoway_demean(copy(a), uid, tid, N, T; tol=tol, maxit=maxit)
        size(Q, 2) > 0 && (Ma .-= Q * (Q' * Ma))
        @inbounds for k in 1:n
            total += abs2(Ma[k] - a[k])
        end
    end
    return (ratio=total / tau_star2, G=G, n=n, cells=cells)
end

"""
    breakdown_reliability(beta_star, sigma, tau_star2;
                          alpha=0.05, delta=0.05, psi=1.0) -> Float64

Self-consistent breakdown reliability (Definition def-breakdown in the
accompanying measurement-error article): the fixed
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
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    tau_star2 > 0 || throw(ArgumentError("tau_star2 must be positive"))
    psi > 0 || throw(ArgumentError("psi must be positive"))
    beta_star == 0 && return 0.0
    eta_dag = _eta_dagger(alpha, delta)
    t_star = abs(beta_star) * sqrt(tau_star2) / (sigma * sqrt(psi))
    return t_star / (t_star + eta_dag)
end

"""
    certified_breakdown_reliability(beta_star, sigma, tau_star2;
                                    alpha=0.05, delta=0.05,
                                    gamma_beta=0.05, psi=1.0) -> Float64

Certified breakdown reliability from the reported t-statistic. It replaces
`|t|` by `|t| + z_(1-gamma_beta)` in [`breakdown_reliability`](@ref):
`lambda†_gamma = (|t| + z)/( |t| + z + eta†)`. Compare this requirement with a
known/consistent within reliability or with a lower confidence bound. If the
lower bound has coverage error `gamma_lambda`, the total false-certification
bound is `gamma_beta + gamma_lambda`; that second budget is attached to the
bound, not to this threshold formula.
"""
function certified_breakdown_reliability(beta_star::Real, sigma::Real,
                                         tau_star2::Real;
                                         alpha::Real=0.05,
                                         delta::Real=0.05,
                                         gamma_beta::Real=0.05,
                                         psi::Real=1.0)
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    tau_star2 > 0 || throw(ArgumentError("tau_star2 must be positive"))
    psi > 0 || throw(ArgumentError("psi must be positive"))
    (isfinite(gamma_beta) && 0 < gamma_beta <= 0.5) ||
        throw(ArgumentError("gamma_beta must be in (0, 0.5]"))
    eta_dag = _eta_dagger(alpha, delta)
    t_reported = abs(beta_star) * sqrt(tau_star2) / (sigma * sqrt(psi))
    tz = t_reported + _norminv(1 - gamma_beta)
    return tz / (tz + eta_dag)
end

"""
    eiv_adequacy(y, x, unit, time; <noise input>, controls=nothing,
                 alpha=0.05, delta=0.05,
                 gamma=0.05, gamma_lambda=0.0,
                 reliability_lower=nothing, pilot=:conservative,
                 cluster=:iid, psi=nothing) -> AdequacyReport

Measurement-error diagnostic. Reproduces the user's FE regression of `y` on
the observed regressor `x` via Frisch-Waugh, then certifies whether naive
inference is size-controlled under classical measurement error. Optional
numeric `controls` are partialled from the outcome and target regressor; their
within-FE leverage is included in the scalar or observation-specific noise
trace.

Noise input — exactly one of:
- `sigma_nu`     : per-observation (or scalar) measurement-error SD
- `codelow`, `codehigh` : interval bounds whose half-width is calibrated as one error SD
- `reliability`  : the within reliability lambda_hat directly (external estimate
                   of the WITHIN-transformed regressor's reliability)

For a formal certificate, `reliability_lower` may supply a lower confidence
bound for within reliability and `gamma_lambda` its coverage error. If omitted,
the computed `lambda_hat` is treated as known/consistent (`gamma_lambda=0`), so
the guarantee is conditional on that treatment. The backward-compatible
keyword `gamma` is the paper's coefficient budget `gamma_beta`.

`pilot` (correct-by-default honesty machinery, spec §4):
- `:conservative` (default) — formal certificate: compare the reliability lower
  bound with the certified breakdown obtained by replacing `|t|` with
  `|t| + z_(1-gamma)` (protocol step 6)
- `:point` — corrected point pilot; descriptive verdict (protocol step 4)
- `:naive` — attenuated `beta*` pilot; ANTI-CONSERVATIVE, for comparison only

`cluster` (cluster-robust extension in the accompanying article):
- `:iid` (default) — homoskedastic-i.i.d. standard errors (Theorem thm-noncentral)
- `:crve` — by-unit cluster-robust (Arellano CR1) variance-inflation `psi_hat`;
  `|eta|` and the breakdown are deflated by `sqrt(psi_hat)`
- `:ar1` — parametric within-unit AR(1) variance inflation (Paper C route (i))
- or pass `psi = <value>` to supply your own variance-inflation factor

The breakdown reliability is the fixed point `lambda† = t*/(t* + eta†)` with
`t* = |beta*| sqrt(tau*2)/sigma` (Definition def-breakdown): pilot-free, and
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
                      controls=nothing,
                      alpha::Real=0.05, delta::Real=0.05, gamma::Real=0.05,
                      reliability_lower::Union{Nothing,Real}=nothing,
                      gamma_lambda::Real=0.0,
                      pilot::Symbol=:conservative,
                      cluster::Symbol=:iid, psi::Union{Nothing,Real}=nothing)
    cluster in (:iid, :crve, :ar1) ||
        throw(ArgumentError("cluster must be :iid, :crve, or :ar1 (or pass psi=...)"))
    uid, tid, N, T = _integer_codes(unit, time)
    n = length(uid)
    (length(y) == n && length(x) == n) ||
        throw(ArgumentError("y, x, unit, time must have equal length"))
    d_K, ncomp = fe_dimension(uid, tid, N, T)
    partial = _partial_within_codes(x, uid, tid, N, T; controls=controls)
    dof = n - d_K - partial.rank - 1
    dof > 0 || throw(ArgumentError("no residual degrees of freedom"))

    xt = partial.xt
    yt = _partial_outcome_codes(y, uid, tid, N, T, partial.Q)
    tau_star2 = sum(abs2, xt)
    tau_star2 > 1e-12 * max(sum(abs2, Float64.(x)), 1.0) ||
        throw(ArgumentError("regressor has no within variation"))
    beta_star = dot(xt, yt) / tau_star2
    u = yt .- beta_star .* xt
    sigma = sqrt(sum(abs2, u) / dof)
    sigma > 0 || throw(ArgumentError("residual standard deviation is zero"))

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
        if sigma_nu isa Real
            sigma_nu >= 0 || throw(ArgumentError("sigma_nu must be non-negative"))
            s2 = abs2(Float64(sigma_nu))
            a_hat = s2 * (n - d_K - partial.rank)
        else
            length(sigma_nu) == n || throw(ArgumentError(
                "sigma_nu must be scalar or have length n = $n"))
            all(sigma_nu .>= 0) || throw(ArgumentError(
                "sigma_nu values must be non-negative"))
            p_nuisance = _fe_leverage_diag(uid, tid, N, T)
            if size(partial.Q, 2) > 0
                p_nuisance .+= vec(sum(abs2, partial.Q; dims=2))
            end
            a_hat = dot(1 .- p_nuisance, abs2.(Float64.(sigma_nu)))
        end
        lambda = 1 - a_hat / tau_star2
    end

    extra = String[]
    length(unique(x)) <= 2 && push!(extra,
        "binary treatment detected: errors in binary treatments are MISCLASSIFICATION (nonclassical); this classical-EIV threshold does not apply (scope section of the measurement-error article)")

    # cluster variance-inflation factor psi_hat (sec-cluster)
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

    # Assumption ass-cluster and projection compatibility. When the package
    # constructs a unit-clustered layer, compute the direct sample analogue from
    # the paper as well as the d_ne/G shortcut. A bare user-supplied psi with
    # cluster=:iid carries no cluster ids, so compatibility cannot be checked.
    cdiag = nothing
    cluster_in_force = cluster !== :iid
    if cluster_in_force
        cdiag = cluster_diagnostics(xt, uid, [(:unit, uid), (:time, tid)];
                                    tau_star2=tau_star2)
        pcomp = projection_compatibility(xt, uid, unit, time;
                                         tau_star2=tau_star2,
                                         controls=controls)
        cdiag = merge(cdiag, (projection_ratio=pcomp.ratio,
                              projection_cells=pcomp.cells))
        if cdiag.ratio_ne > 0.20
            push!(extra, @sprintf("PROJECTION-COMPATIBILITY SHORTCUT STRAINED: %d fixed effects cut across only %d clusters (d_ne/G = %.3f). This shortcut needs the spectral and energy balance conditions (N1)--(N2); it is not the assumption itself. Nested dimensions here: %s.",
                                  cdiag.d_ne, cdiag.G, cdiag.ratio_ne,
                                  isempty(cdiag.nested) ? "none" : join(cdiag.nested, ", ")))
        end
        if isfinite(cdiag.projection_ratio)
            push!(extra, @sprintf("direct projection-compatibility diagnostic chi_proj = %.5f = sum_g ||M a^(g)-a^(g)||^2/tau*2 (measurement-error protocol). This finite-panel number evaluates the named sample quantity but does not itself prove the asymptotic sequence condition.",
                                  cdiag.projection_ratio))
        else
            push!(extra, @sprintf("direct projection-compatibility diagnostic skipped because n*G = %.0f exceeds its allocation guard; call projection_compatibility(...; max_cells=...) deliberately to compute it.",
                                  cdiag.projection_cells))
        end
        if cdiag.max_energy > 0.10
            push!(extra, @sprintf("one cluster carries %.0f%% of the residualized signal (max_g A_g/tau*2): the no-dominant-cluster condition ass-cluster(ii) is strained and the cluster CLT may not apply",
                                  100 * cdiag.max_energy))
        end
        if cdiag.G < 30
            push!(extra, @sprintf("only G = %d clusters: ass-cluster(ii) is a many-clusters condition and the cluster-robust reading is unreliable at this G",
                                  cdiag.G))
        end
    elseif psi !== nothing && psi_hat != 1.0
        push!(extra, "psi was supplied directly with cluster=:iid, so the scale adjustment is computed but projection compatibility, cluster count and dominance cannot be checked without cluster identifiers. Pass cluster=:crve (or :ar1) when psi refers to unit clusters.")
    end

    design = _design_summary_codes(uid, tid, N, T; xt=xt)
    return _eiv_core(design, beta_star, sigma, tau_star2, lambda;
                      alpha=alpha, delta=delta, gamma=gamma, pilot=pilot,
                      reliability_lower=reliability_lower,
                      gamma_lambda=gamma_lambda,
                      psi_hat=psi_hat, rho_ar1=rho_ar1, cluster_diag=cdiag,
                      extra_notes=extra)
end

function eiv_adequacy(; beta_star::Real, sigma::Real, tau_star2::Real,
                      n::Integer, d_K::Integer,
                      reliability::Union{Nothing,Real}=nothing,
                      sigma_nu2::Union{Nothing,Real}=nothing,
                      N::Integer=0, T::Integer=0,
                      alpha::Real=0.05, delta::Real=0.05, gamma::Real=0.05,
                      reliability_lower::Union{Nothing,Real}=nothing,
                      gamma_lambda::Real=0.0,
                      pilot::Symbol=:conservative,
                      psi::Union{Nothing,Real}=nothing)
    (reliability !== nothing) + (sigma_nu2 !== nothing) == 1 ||
        throw(ArgumentError("supply exactly one of reliability or sigma_nu2"))
    n > 0 || throw(ArgumentError("n must be positive"))
    0 <= d_K < n || throw(ArgumentError("d_K must satisfy 0 <= d_K < n"))
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    tau_star2 > 0 || throw(ArgumentError("tau_star2 must be positive"))
    lambda = if reliability !== nothing
        0 < reliability <= 1 ||
            throw(ArgumentError("reliability must be in (0, 1]"))
        Float64(reliability)
    else
        sigma_nu2 >= 0 || throw(ArgumentError("sigma_nu2 must be non-negative"))
        1 - sigma_nu2 * (n - d_K) / tau_star2
    end
    design = DesignSummary(n, N, T, d_K, d_K / n, 1, Float64(tau_star2))
    return _eiv_core(design, Float64(beta_star), Float64(sigma),
                      Float64(tau_star2), lambda; alpha=alpha, delta=delta,
                      gamma=gamma, pilot=pilot,
                      reliability_lower=reliability_lower,
                      gamma_lambda=gamma_lambda,
                      psi_hat=(psi === nothing ? 1.0 : Float64(psi)),
                      extra_notes=String[])
end

"""
    eiv_adequacy_summary(beta_star, sigma, tau_star2, n, d_K; kwargs...)

Summary-output form of [`eiv_adequacy`](@ref), matching the R `panelcert` API.
Use it when raw observations are unavailable but the fitted slope, residual
scale, observed within variation, sample size, and fixed-effect dimension are
reported.
"""
function eiv_adequacy_summary(beta_star::Real, sigma::Real, tau_star2::Real,
                              n::Integer, d_K::Integer;
                              reliability::Union{Nothing,Real}=nothing,
                              sigma_nu2::Union{Nothing,Real}=nothing,
                              N::Integer=0, T::Integer=0,
                              alpha::Real=0.05, delta::Real=0.05,
                              gamma::Real=0.05,
                              reliability_lower::Union{Nothing,Real}=nothing,
                              gamma_lambda::Real=0.0,
                              pilot::Symbol=:conservative,
                              psi::Union{Nothing,Real}=nothing)
    return eiv_adequacy(; beta_star=beta_star, sigma=sigma,
                        tau_star2=tau_star2, n=n, d_K=d_K,
                        reliability=reliability, sigma_nu2=sigma_nu2,
                        N=N, T=T, alpha=alpha, delta=delta, gamma=gamma,
                        reliability_lower=reliability_lower,
                        gamma_lambda=gamma_lambda,
                        pilot=pilot, psi=psi)
end

function _eiv_core(design::DesignSummary, beta_star::Float64, sigma::Float64,
                   tau_star2::Float64, lambda::Float64;
                   alpha::Real, delta::Real, gamma::Real, pilot::Symbol,
                   reliability_lower::Union{Nothing,Real}=nothing,
                   gamma_lambda::Real=0.0,
                   psi_hat::Float64=1.0, rho_ar1::Union{Nothing,Float64}=nothing,
                   cluster_diag=nothing, extra_notes::Vector{String})
    pilot in (:conservative, :point, :naive) ||
        throw(ArgumentError("pilot must be :conservative, :point, or :naive"))
    0 < alpha < 1 || throw(ArgumentError("alpha must lie in (0, 1)"))
    0 < delta < 1 - alpha || throw(ArgumentError(
        "delta must lie in (0, 1-alpha)"))
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    tau_star2 > 0 || throw(ArgumentError("tau_star2 must be positive"))
    lambda <= 1 || throw(ArgumentError("reliability cannot exceed one"))
    psi_hat > 0 || throw(ArgumentError("psi must be positive"))
    # gamma is the coefficient budget gamma_beta and must leave
    # z_(1-gamma) >= 0. At
    # gamma > 1/2 the "upper" bound U_n = |beta0_corr| + z_{1-gamma} se would
    # DEFLATE the pilot, making the conservative verdict weaker than the point
    # verdict while still being labelled :CERTIFIED.
    (isfinite(gamma) && 0 < gamma <= 0.5) || throw(ArgumentError(
        "gamma must be in (0, 0.5]: at gamma > 0.5 the certificate's upper " *
        "confidence bound deflates rather than inflates the pilot and the " *
        "verdict is no longer conservative"))
    (isfinite(gamma_lambda) && 0 <= gamma_lambda < 1) ||
        throw(ArgumentError("gamma_lambda must be in [0, 1)"))
    gamma + gamma_lambda < 1 ||
        throw(ArgumentError("gamma + gamma_lambda must be less than one"))
    eta_dag = _eta_dagger(alpha, delta)
    sqpsi = sqrt(psi_hat)
    notes = copy(extra_notes)

    # fixed-point breakdown lambda† = t*/(t* + eta†), t* = |beta*|sqrt(tau*2)/sigma
    # (Definition def-breakdown) — the pilot-free, self-consistent evaluation;
    # under clustering t* is rescaled by 1/sqrt(psi)
    t_star = abs(beta_star) * sqrt(tau_star2) / sigma
    t_reported = t_star / sqpsi
    breakdown = t_reported / (t_reported + eta_dag)
    z_beta = _norminv(1 - gamma)
    certified_breakdown = (t_reported + z_beta) /
                          (t_reported + z_beta + eta_dag)

    if lambda <= 0
        push!(notes, @sprintf("implied noise exceeds ALL residual within variation (lambda_hat = %.3f <= 0): the noise pilot may be misscaled, or attenuation is total; exact size = 1", lambda))
        statistic = (lambda_hat=lambda, noise_ratio=Inf, beta_star=beta_star,
                     sigma=sigma, t_star=t_star, psi_hat=psi_hat)
        return AdequacyReport(:measurement_error, design, statistic, Inf,
                              eta_dag, breakdown, 1.0, :FLAGGED, Float64(alpha),
                              Float64(delta), notes)
    end

    ell = reliability_lower === nothing ? lambda : Float64(reliability_lower)
    (isfinite(ell) && 0 < ell <= 1) ||
        throw(ArgumentError("reliability_lower must be in (0, 1]"))

    # The cluster-robust scale (cor-cluster-feasible): s_CR^2 = sigma_CJN^2 * psi
    # = V^sc_CR/tau*2, an algebraic identity. EVERY scale below is s_CR — the
    # non-centrality, the pilot standard error, and hence the certificate. Using
    # the i.i.d. sigma_hat for the pilot se (as the pre-correction code did)
    # understates the certificate's width by sqrt(psi) and is anti-conservative
    # whenever psi > 1.
    s_CR = sigma * sqpsi

    beta_corr = beta_star / lambda
    se_corr = s_CR / (lambda * sqrt(tau_star2))
    b_pilot = pilot === :naive ? abs(beta_star) : abs(beta_corr)
    eta_point = (b_pilot / s_CR) * (1 - lambda) * sqrt(tau_star2)
    eta_upper = pilot === :conservative ?
        (t_reported + z_beta) * (1 - ell) / ell : nothing
    eta_used = pilot === :conservative ? eta_upper : eta_point

    # rem-plugin: only the certificate is size-controlled. The point plug-in is
    # a descriptive statistic — a nondegenerate random multiple of the target —
    # so a passing point verdict is a POINT PASS, not a certificate.
    passes = pilot === :conservative ? ell >= certified_breakdown :
                                      eta_used <= eta_dag
    verdict = passes ? (pilot === :conservative ? :CERTIFIED : :POINT_PASS) :
                       :FLAGGED
    implied_size = _noncentral_size(eta_point, alpha)

    if pilot === :conservative
        push!(notes, @sprintf("formal certificate (prop-certificate): certified breakdown lambda_dagger_gamma = %.3f is obtained by replacing |t| with |t| + z_(1-gamma_beta); the comparison uses reliability lower bound ell = %.3f. False certification is at most gamma_beta + gamma_lambda = %.3g + %.3g = %.3g, without requiring independence. The implied size shown is at the point pilot.",
                              certified_breakdown, ell, gamma, gamma_lambda,
                              gamma + gamma_lambda))
        if reliability_lower === nothing
            push!(notes, "CONDITIONAL RELIABILITY TREATMENT: no reliability_lower was supplied, so lambda_hat is treated as known/consistent and gamma_lambda = 0. A noisy finite-sample reliability estimate requires a lower confidence bound and its coverage-error budget.")
        end
    elseif pilot === :point
        push!(notes, "POINT PASS, not a certificate (rem-plugin): the corrected pilot at its point estimate is descriptive. Under weak information eta_hat converges to a nondegenerate random multiple (|B|/|beta0|)|eta| of the target, with median close to it but no concentration — its sampling variability is a first-order feature of the regime, not a vanishing approximation error. For a size-controlled statement use pilot=:conservative.")
    else
        push!(notes, "ANTI-CONSERVATIVE naive pilot (attenuated beta*) — for comparison only; understates |eta| by the factor lambda (prop-pilot(i))")
    end
    pilot !== :naive && push!(notes,
        "the corrected pilot beta*/lambda_hat is not a consistent point estimate under weak information (cor-slope); it is reported with its sampling band. Consistency is recovered only under strong information n Q_K -> infinity.")
    eta_point > 1 && push!(notes,
        "far from the threshold (|eta| > 1): the local quadratic approximation is uninformative here; the verdict uses exact inversion (rem-exact-cv)")
    lambda < 1 && push!(notes, @sprintf("power tax: local power slope attenuated by sqrt(lambda) = %.2f (prop-power)", sqrt(lambda)))
    if psi_hat != 1.0
        # rem-psi-sign: report the realized direction; never generalize it.
        dir = psi_hat > 1 ?
            "on THIS design clustering deflates the measured distortion" :
            "on THIS design clustering WORSENS the distortion (psi < 1) — the i.i.d. reading is anti-conservative here"
        push!(notes, @sprintf("cluster-robust standardization: psi_hat = %.3f, so the scale is s_CR = sigma*sqrt(psi) = %.4g and |eta|, t* are deflated by 1/sqrt(psi) = %.3f; %s. The sign of psi-1 is not free and not guessable (rem-psi-sign): an equicorrelated component at a level that is itself a fixed effect is annihilated exactly, and a serially dependent error with a within-cluster serially independent regressor gives psi < 1. What pushes psi above one is persistence in the regressor and the error together. Read the direction off psi_hat.",
                              psi_hat, s_CR, 1 / sqpsi, dir))
        push!(notes, @sprintf("lambda†_CR = %.3f is Definition def-breakdown evaluated at the reported cluster-robust t-statistic t^CR = t*/sqrt(psi) = %.3f — an algebraic identity (cor-cluster-feasible(c)), requiring no limit theory: the cluster-robust diagnostic IS the i.i.d. diagnostic run on the reported cluster-robust t.",
                              breakdown, t_star / sqpsi))
        push!(notes, "the CRVE here omits the conventional (n-1)/(n-K) small-sample factor, which converges to 1/(1-rho) rather than 1 and so OVER-corrects when rho is non-negligible (lem-crve). Default software applies it: reproducing psi_hat with such defaults will inflate it by roughly 1/(1-rho).")
    end

    statistic_base = (lambda_hat=lambda, noise_ratio=(1 - lambda) / lambda,
                      reliability_lower=ell,
                      beta_star=beta_star, beta_corr=beta_corr,
                      se_beta_corr=se_corr, sigma=sigma, s_CR=s_CR,
                      t_star=t_star, t_CR=t_star / sqpsi, psi_hat=psi_hat,
                      breakdown_certified=certified_breakdown,
                      gamma=Float64(gamma), gamma_beta=Float64(gamma),
                      gamma_lambda=Float64(gamma_lambda),
                      false_certification_bound=Float64(gamma + gamma_lambda),
                      pilot=pilot,
                      eta_quad_threshold=_eta_quad(alpha, delta))
    rho_ar1 !== nothing &&
        (statistic_base = merge(statistic_base, (rho_ar1=rho_ar1,)))
    cluster_diag !== nothing &&
        (statistic_base = merge(statistic_base, (cluster=cluster_diag,)))
    statistic = eta_upper === nothing ? statistic_base :
                merge(statistic_base, (eta_upper=eta_upper,))

    return AdequacyReport(:measurement_error, design, statistic, eta_point,
                          eta_dag, breakdown, implied_size, verdict,
                          Float64(alpha), Float64(delta), notes)
end
