# =============================================================================
# Module C — staggered-DiD / TWFE-heterogeneity adequacy
# SOURCE: Paper C, "Positive Weights Do Not Certify TWFE Inference"
# and its online supplement.
#
# Gamma, its restricted variants, the direct cluster-score normalization,
# covariance-corrected pilots, the worst-case envelope and the directional
# plug-in are implemented here. The redistributable applications reproduce
# from panels bundled with the package.
#
# Design statistics (pre-outcome, from the adoption pattern ALONE):
#   Gamma      = sqrt(N1) ||w - u||_2,  w = Dt/n_w on treated cells, u = 1/N1
#              (Def. def-gamma); Gamma = 0 iff block design (Prop. prop-gamma0)
#   Gamma_S    = sqrt(N1) ||Pi_S(w - u)||, the restricted statistic for the
#              heterogeneity subspace S (Prop. prop-restricted): cohort
#              (Gamma_coh), event-time (Gamma_evt), additive combined (Gamma_c+e);
#              Gamma_coh, Gamma_evt <= Gamma_c+e <= Gamma.
#   worst case |eta| = (c/sigma) Gamma_S (Cor. cor-gamma);
#   threshold  (c/sigma) Gamma_{S,CR} <= eta†(alpha, delta) (Cor. cor-cv).
#
# Inference layer (needs the outcome):
#   cluster    direct CR1 is the headline normalization; AR(1), iid, and a
#              positive user-supplied psi remain available.
#   pilot      COVARIANCE-AWARE (eq-pilot): each subspace pilot squares to
#                c_S^2 = n_w max{0, (1/N1)||Pi_S(delta-mean)||^2
#                                    - (1/N1) tr(Pi_S~ A Omega A' Pi_S~)},
#              delta = cell-level group-time ATTs, Omega = Cov(hat Delta_{g,t})
#              estimated by the fixed-design wild cluster bootstrap. Subtracting
#              an average of marginal variances (the legacy scalar shrinkage)
#              under-removes the projected noise when the ATT(g,t) share controls.
#   bounds     the lower test uses an HC2 multiplier radius and reverse triangle
#              inequality. The upper certificate uses covariance-aware
#              noncentral-chi-square inversion and requires full projected rank.
#              HC3 is a sensitivity. The full confidence-ball construction is
#              retained as an optional, more general but less powerful fallback.
#   direction  eta_dir,S = sqrt(n_w) (w-u)'P_S Delta_hat / q_hat, a signed
#              companion to (not a replacement for) the worst-case envelope.
#   bootstrap  fixed-design wild cluster bootstrap: D, w, Gamma_S held fixed,
#              Y* = Yhat + v_i e_it (Rademacher per unit), recomputes the whole
#              calibration; supplies Omega and the size intervals.
#
# Always-treated units (adopted before the sample; no observed untreated period)
# violate the setup g_i in {2,...,T} u {inf} and are DROPPED, with a note.
# =============================================================================

"Treatment indicator D_it = 1{time >= first_treat} on raw values (missing = never)."
_treatment_indicator(time::AbstractVector, first_treat::AbstractVector) =
    Float64[(!ismissing(first_treat[k]) && time[k] >= first_treat[k]) ? 1.0 : 0.0
            for k in eachindex(time)]

"""
Codes for the staggered design. Time codes are SORTED; returns per-unit cohort
codes: 0.0 = always-treated (adopted before the sample), Inf = never treated in
sample, else the 1-based time code of the adoption period.
"""
function _staggered_codes(unit::AbstractVector, time::AbstractVector{<:Real},
                          first_treat::AbstractVector)
    n = length(unit)
    (length(time) == n && length(first_treat) == n) ||
        throw(ArgumentError("unit, time, first_treat must have equal length"))
    umap = Dict{eltype(unit),Int}()
    uid = Vector{Int}(undef, n)
    for k in 1:n
        uid[k] = get!(umap, unit[k], length(umap) + 1)
    end
    stimes = sort(unique(time))
    tmap = Dict(t => i for (i, t) in enumerate(stimes))
    tid = Int[tmap[t] for t in time]
    N, T = length(umap), length(stimes)

    ft_of = Vector{Union{Missing,Float64}}(missing, N)
    seen = falses(N)
    for k in 1:n
        f = first_treat[k]
        if !seen[uid[k]]
            ft_of[uid[k]] = ismissing(f) ? missing : Float64(f)
            seen[uid[k]] = true
        else
            isequal(ft_of[uid[k]], ismissing(f) ? missing : Float64(f)) ||
                throw(ArgumentError("first_treat varies within unit $(unit[k])"))
        end
    end
    ftc = Vector{Float64}(undef, N)
    for i in 1:N
        f = ft_of[i]
        if ismissing(f) || !isfinite(f) || f > stimes[end]
            ftc[i] = Inf
        elseif f <= stimes[1]
            ftc[i] = 0.0
        else
            pos = findfirst(t -> Float64(t) == f, stimes)
            pos === nothing &&
                throw(ArgumentError("first_treat value $f is not an observed period"))
            ftc[i] = Float64(pos)
        end
    end
    return uid, tid, N, T, ftc
end

"""
Drop always-treated units (cohort code 0.0): they violate the setup g_i >= 2 and
have no observed untreated period. Returns re-indexed (uid, tid, N, T, ftc, keep)
plus the number of units dropped.
"""
function _drop_always_treated(uid, tid, ftc)
    N = maximum(uid)
    drop = Set(i for i in 1:N if ftc[i] == 0.0)
    n_drop = length(drop)
    n_drop == 0 && return uid, tid, ftc, trues(length(uid)), 0
    keep = [!(uid[k] in drop) for k in eachindex(uid)]
    # re-index surviving units to 1:N'
    survivors = sort([i for i in 1:N if !(i in drop)])
    remap = Dict(u => j for (j, u) in enumerate(survivors))
    uid2 = Int[remap[uid[k]] for k in eachindex(uid) if keep[k]]
    tid2 = Int[tid[k] for k in eachindex(uid) if keep[k]]
    ftc2 = Float64[ftc[survivors[j]] for j in 1:length(survivors)]
    return uid2, tid2, ftc2, keep, n_drop
end

"Weights, Gamma, negative share from the within-transformed treatment."
function _design_stats(D::Vector{Float64}, Dt::Vector{Float64})
    treated = findall(==(1.0), D)
    N1 = length(treated)
    N1 >= 2 || throw(ArgumentError("fewer than 2 treated cells — no staggered design"))
    n_w = sum(abs2, Dt)
    n_w > 1e-12 || throw(ArgumentError(
        "treatment has no within variation (single common adoption date?)"))
    w = Dt[treated] ./ sum(Dt[treated])
    Gamma = sqrt(N1) * sqrt(sum(abs2, w .- 1 / N1))
    neg_share = count(<(0), w) / N1
    return treated, N1, n_w, w, Gamma, neg_share
end

# ------------------------------------------------------------------------------
# Restricted-profile design statistics (Prop. prop-restricted) by cell-level
# projection onto cohort / event-time / additive / saturated group-time spaces.
# ------------------------------------------------------------------------------

"N1 x k indicator matrix, one column per distinct label (constant in the span)."
function _indicator_basis(labels::AbstractVector)
    labs = sort(unique(labels))
    idx = Dict(l => j for (j, l) in enumerate(labs))
    B = zeros(length(labels), length(labs))
    for (i, l) in enumerate(labels)
        B[i, idx[l]] = 1.0
    end
    return B
end

"Orthogonal projection of x onto col(B) (rank-deficient-safe)."
_proj(B::AbstractMatrix, x::Union{AbstractVector,AbstractMatrix}) =
    B * (pinv(B' * B) * (B' * x))

"Hat matrix P_S = B (B'B)^+ B'."
_hat(B::AbstractMatrix) = B * pinv(B' * B) * B'

"Orthonormal basis for the centered column span of B."
function _centered_basis(B::AbstractMatrix; tol::Real=1e-10)
    Bc = B .- mean(B; dims=1)
    F = svd(Bc)
    cutoff = tol * max(maximum(F.S; init=0.0), 1.0)
    keep = findall(>(cutoff), F.S)
    isempty(keep) && return zeros(size(B, 1), 0)
    return F.U[:, keep]
end

"CDF of a noncentral chi-square via a centered Poisson mixture."
function _ncx2_cdf(x::Real, df::Integer, ncp::Real)
    x <= 0 && return x == 0 ? 0.0 : 0.0
    df > 0 || throw(ArgumentError("chi-square degrees of freedom must be positive"))
    ncp >= 0 || throw(ArgumentError("noncentrality must be nonnegative"))
    half = Float64(ncp) / 2
    half == 0 && return gamma_inc(Float64(df) / 2, Float64(x) / 2)[1]
    mode = floor(Int, half)
    log_weight = -half + mode * log(half) - loggamma(mode + 1)
    weight_mode = exp(log_weight)
    cdf = weight_mode * gamma_inc(Float64(df) / 2 + mode,
                                  Float64(x) / 2)[1]
    weight = weight_mode
    for j in mode:-1:1
        weight *= j / half
        cdf += weight * gamma_inc(Float64(df) / 2 + j - 1,
                                  Float64(x) / 2)[1]
    end
    weight = weight_mode
    j = mode
    while j < 100_000
        j += 1
        weight *= half / j
        cdf += weight * gamma_inc(Float64(df) / 2 + j,
                                  Float64(x) / 2)[1]
        weight < 1e-15 && j > mode + 10 * sqrt(half + 1) && break
    end
    return clamp(cdf, 0.0, 1.0)
end

"One-sided upper confidence bound for chi-square noncentrality."
function _noncentrality_upper(wald::Real, rank::Integer, error_prob::Real)
    _ncx2_cdf(wald, rank, 0.0) <= error_prob && return 0.0
    upper = max(1.0, Float64(wald) + rank)
    while _ncx2_cdf(wald, rank, upper) > error_prob
        upper *= 2
        isfinite(upper) || error("failed to bracket noncentrality upper bound")
    end
    lower = 0.0
    for _ in 1:100
        middle = (lower + upper) / 2
        if _ncx2_cdf(wald, rank, middle) > error_prob
            lower = middle
        else
            upper = middle
        end
    end
    return (lower + upper) / 2
end

"Covariance-aware upper bound for the Euclidean envelope K."
function _equivalence_upper(delta_value, Q, A, score, Gamma, se, error_prob)
    expected_rank = size(Q, 2)
    if expected_rank == 0 || abs(Gamma) <= 1e-14
        return (expected_rank=expected_rank, covariance_rank=0,
                supported=true, wald=0.0, lambda_max=0.0,
                ncp_upper=0.0, K_upper=0.0)
    end
    N1 = length(delta_value)
    factor = Gamma / se
    effect = vec(Q' * delta_value) / sqrt(N1)
    score_coordinates = ((Q' * A) * score) / sqrt(N1)
    covariance = factor^2 .* (score_coordinates * score_coordinates')
    F = eigen(Symmetric((covariance + covariance') / 2))
    largest = isempty(F.values) ? 0.0 : maximum(F.values)
    tolerance = max(largest, floatmin(Float64)) *
                max(size(covariance)...) * eps(Float64) * 100
    keep = findall(>(tolerance), F.values)
    covariance_rank = length(keep)
    supported = covariance_rank == expected_rank
    covariance_rank == 0 && return (
        expected_rank=expected_rank, covariance_rank=0, supported=supported,
        wald=NaN, lambda_max=NaN, ncp_upper=NaN, K_upper=NaN)
    values = F.values[keep]
    vectors = F.vectors[:, keep]
    coordinates = vectors' * (factor .* effect)
    wald = sum(abs2.(coordinates) ./ values)
    lambda_max = maximum(values)
    supported || return (
        expected_rank=expected_rank, covariance_rank=covariance_rank,
        supported=false, wald=wald, lambda_max=lambda_max,
        ncp_upper=NaN, K_upper=NaN)
    ncp_upper = _noncentrality_upper(wald, covariance_rank, error_prob)
    return (expected_rank=expected_rank, covariance_rank=covariance_rank,
            supported=true, wald=wald, lambda_max=lambda_max,
            ncp_upper=ncp_upper, K_upper=sqrt(lambda_max * ncp_upper))
end

"""
Restricted design statistics and the cell bases. `g_cell`, `t_cell`, and
`e_cell` are the cohort, calendar time, and event time of each treated cell.
In a balanced panel the saturated group-time statistic equals the unrestricted
statistic: `Gamma_gt = Gamma`.
"""
function _restricted_gammas(w::Vector{Float64}, g_cell::Vector{Float64},
                            t_cell::Vector{Int}, e_cell::Vector{Float64}, N1::Int)
    d = w .- 1 / N1
    Bcoh = _indicator_basis(g_cell)
    Bevt = _indicator_basis(e_cell)
    Bcmb = hcat(Bcoh, Bevt)
    Bgt = _indicator_basis(collect(zip(g_cell, t_cell)))
    gam(B) = sqrt(N1) * norm(_proj(B, d))
    out = (unr = sqrt(N1) * norm(d), coh = gam(Bcoh), evt = gam(Bevt),
           cmb = gam(Bcmb), gt = gam(Bgt), Bcoh = Bcoh, Bevt = Bevt,
           Bcmb = Bcmb, Bgt = Bgt)
    isapprox(out.gt, out.unr; atol=1e-9, rtol=1e-9) ||
        error("internal group-time closure check failed")
    return out
end

# ------------------------------------------------------------------------------
# Group-time ATTs: simplified not-yet-treated difference-in-means at (g,t)
# resolution (a heterogeneity-robust order-of-magnitude pilot, NOT csdid; pass
# your own via `cohort_effects` for a paper-grade pilot). Always-treated units
# have been dropped, so the control pool is not-yet-treated + never-treated.
# ------------------------------------------------------------------------------

function _group_time_atts(uid::Vector{Int}, tid::Vector{Int}, ftc::Vector{Float64},
                          y::Vector{Float64}, T::Int; controls::Symbol=:not_yet)
    controls in (:not_yet, :never) ||
        throw(ArgumentError("controls must be :not_yet or :never"))
    key = Dict{Tuple{Int,Int},Float64}()
    for k in eachindex(uid)
        key[(uid[k], tid[k])] = y[k]
    end
    N = length(ftc)
    cohorts = sort(unique(f for f in ftc if isfinite(f) && f > 1))
    gt = Dict{Tuple{Int,Int},Float64}()
    for g in cohorts
        gunits = findall(==(g), ftc)
        base = Int(g) - 1
        for t in Int(g):T
            ctrl = controls === :never ? findall(i -> !isfinite(ftc[i]), 1:N) :
                                         findall(i -> ftc[i] > t, 1:N)
            gd = [key[(u, t)] - key[(u, base)] for u in gunits
                  if haskey(key, (u, t)) && haskey(key, (u, base))]
            cd = [key[(u, t)] - key[(u, base)] for u in ctrl
                  if haskey(key, (u, t)) && haskey(key, (u, base))]
            (isempty(gd) || isempty(cd)) && continue
            gt[(Int(g), t)] = mean(gd) - mean(cd)
        end
    end
    return gt, cohorts
end

"Cohort-level effects (cell-count-weighted mean of ATT(g,t) over t) from the gt dict."
function _cohort_means(gt::Dict{Tuple{Int,Int},Float64}, cohorts::Vector{Float64})
    out = Dict{Float64,Float64}()
    for g in cohorts
        vals = [v for ((gg, _), v) in gt if gg == Int(g)]
        isempty(vals) || (out[g] = mean(vals))
    end
    return out
end

# ------------------------------------------------------------------------------
# Covariance-aware pilot (eq-pilot). `A` maps the ordered group-time vector to
# treated cells; `Q[s]` is an orthonormal basis for the centered class.
# ------------------------------------------------------------------------------

"Cell-level effect vector delta from the gt dict (NaN where unidentified)."
function _delta_cell(gt::Dict{Tuple{Int,Int},Float64}, g_cell::Vector{Float64},
                     t_cell::Vector{Int}, N1::Int)
    return Float64[get(gt, (Int(g_cell[k]), t_cell[k]), NaN) for k in 1:N1]
end

"Design-constant noise traces, evaluated without an N1-by-N1 covariance."
function _cov_traces(A::Matrix{Float64}, Omega::Matrix{Float64},
                     Q::NamedTuple, N1::Int)
    traceone(q) = begin
        coordinates = q' * A
        sum((coordinates * Omega) .* coordinates) / N1
    end
    return (coh=traceone(Q.coh), evt=traceone(Q.evt),
            cmb=traceone(Q.cmb), gt=traceone(Q.gt))
end

"Covariance-corrected c_S/sigma per subspace from a group-time dict."
function _cov_pilots(gt, g_cell, t_cell, N1, bases, traces, n_w, sigma)
    delta = _delta_cell(gt, g_cell, t_cell, N1)
    cov = .!isnan.(delta)
    dc = delta .- mean(delta[cov])
    pil(B, tr) = begin
        comp = _proj(B[cov, :], dc[cov])
        raw = dot(comp, comp) / count(cov)
        sqrt(n_w * max(0.0, raw - tr) / sigma^2)
    end
    return (coh = pil(bases.Bcoh, traces.coh), evt = pil(bases.Bevt, traces.evt),
            cmb = pil(bases.Bcmb, traces.cmb), gt = pil(bases.Bgt, traces.gt))
end

function _directional_profile(gt, g_cell, t_cell, N1, B, w, n_w, sigma, psi)
    delta = _delta_cell(gt, g_cell, t_cell, N1)
    all(isfinite, delta) && isfinite(sigma) && sigma > 0 &&
        isfinite(psi) && psi > 0 || return (eta=NaN, alignment=NaN)
    profile = _proj(B, delta .- mean(delta))
    exposure = _proj(B, w .- 1 / N1)
    bias = dot(exposure, profile)
    den = sqrt(dot(exposure, exposure) * dot(profile, profile))
    return (eta=sqrt(n_w) * bias / (sigma * sqrt(psi)),
            alignment=den > 0 ? bias / den : 0.0)
end

function _group_time_map(uid, tid, ftc, N, T; controls::Symbol=:not_yet)
    controls in (:not_yet, :never) ||
        throw(ArgumentError("controls must be :not_yet or :never"))
    n = length(uid)
    obs = zeros(Int, N, T)
    for k in eachindex(uid); obs[uid[k], tid[k]] = k; end
    cohorts = sort(unique(filter(g -> isfinite(g) && g > 1, ftc)))
    keys = Tuple{Int,Int}[]
    for g0 in cohorts
        g = Int(g0); base = g - 1
        for t in g:T
            gunits = findall(==(g0), ftc)
            ctrl = controls === :never ? findall(!isfinite, ftc) :
                                         findall(x -> x > t, ftc)
            filter!(u -> obs[u, t] > 0 && obs[u, base] > 0, gunits)
            filter!(u -> obs[u, t] > 0 && obs[u, base] > 0, ctrl)
            !isempty(gunits) && !isempty(ctrl) && push!(keys, (g, t))
        end
    end
    L = zeros(n, length(keys))
    for (j, (g, t)) in enumerate(keys)
        base = g - 1
        gunits = findall(==(Float64(g)), ftc)
        ctrl = controls === :never ? findall(!isfinite, ftc) :
                                     findall(x -> x > t, ftc)
        filter!(u -> obs[u, t] > 0 && obs[u, base] > 0, gunits)
        filter!(u -> obs[u, t] > 0 && obs[u, base] > 0, ctrl)
        for u in gunits
            L[obs[u, t], j] = 1 / length(gunits)
            L[obs[u, base], j] = -1 / length(gunits)
        end
        for u in ctrl
            L[obs[u, t], j] = -1 / length(ctrl)
            L[obs[u, base], j] = 1 / length(ctrl)
        end
    end
    return keys, L
end

# ------------------------------------------------------------------------------
# Fixed-design wild cluster bootstrap: Omega + calibration-uncertainty draws.
# ------------------------------------------------------------------------------

function _wild_bootstrap(uid, tid, ftc, D, Dt, y, N, T, n_w, treated, N1,
                         g_cell, t_cell, d_K, gt0, B::Int, seed::Int,
                         controls::Symbol)
    n = length(uid)
    # Saturated unit + time + treated-(g,t) surface after absorbing the fixed
    # effects. Avoids the dense n x N unit-dummy matrix and scales to counties.
    keys_gt, L = _group_time_map(uid, tid, ftc, N, T; controls=controls)
    m = length(keys_gt)
    kidx = Dict(k => j for (j, k) in enumerate(keys_gt))
    Z = zeros(n, m)
    for k in eachindex(D)
        if D[k] == 1.0
            key = (Int(ftc[uid[k]]), tid[k])
            haskey(kidx, key) && (Z[k, kidx[key]] = 1.0)
        end
    end
    Zt = zeros(n, m)
    for j in 1:m
        Zt[:, j] = _twoway_demean(Z[:, j], uid, tid, N, T)
    end
    yt = _twoway_demean(y, uid, tid, N, T)
    fit = Zt * (pinv(Zt' * Zt) * (Zt' * yt))
    e = yt .- fit
    yhat = y .- e
    att_center = L' * yhat
    score = zeros(m, N)
    # HC2/HC3 scaling is applied to the treated group-time surface after the
    # unit and time effects are removed. It governs pilot uncertainty, not the
    # standard error of the TWFE statistic being diagnosed.
    Hcoef = Zt * pinv(Zt' * Zt)
    hgt = vec(sum(Hcoef .* Zt; dims=2))
    one_minus_h = max.(1 .- hgt, 1e-10)
    e_hc2 = e ./ sqrt.(one_minus_h)
    e_hc3 = e ./ one_minus_h
    score_hc2 = zeros(m, N)
    score_hc3 = zeros(m, N)
    for k in eachindex(uid), j in 1:m
        score[j, uid[k]] += L[k, j] * e[k]
        score_hc2[j, uid[k]] += L[k, j] * e_hc2[k]
        score_hc3[j, uid[k]] += L[k, j] * e_hc3[k]
    end

    dof = n - d_K - 1
    rng = MersenneTwister(seed)
    rows_sigma = Float64[]; rows_psi_ar1 = Float64[]
    rows_psi_direct = Float64[]; gtmat = Vector{Vector{Float64}}()
    error_hc2 = Matrix{Float64}(undef, m, B)
    error_hc3 = Matrix{Float64}(undef, m, B)
    for b in 1:B
        v = rand(rng, (-1.0, 1.0), N)
        error_hc2[:, b] = score_hc2 * v
        error_hc3[:, b] = score_hc3 * v
        ystar = yhat .+ v[uid] .* e
        ytw = _twoway_demean(ystar, uid, tid, N, T)
        beta = dot(Dt, ytw) / n_w
        resid = ytw .- beta .* Dt
        sigma = sqrt(sum(abs2, resid) / dof)
        rho = _rho_ar1(resid, uid, tid)
        psi_ar1 = _psi_parametric(Dt, uid, tid, rho; kind=:ar1)
        psi_direct = _psi_direct(Dt, resid, uid, n_w, sigma^2, N)
        (isfinite(psi_ar1) && psi_ar1 > 0 &&
         isfinite(psi_direct) && psi_direct > 0) || continue
        push!(rows_sigma, sigma); push!(rows_psi_ar1, psi_ar1)
        push!(rows_psi_direct, psi_direct)
        push!(gtmat, vec(att_center .+ score * v))
    end
    # Exact covariance under the fitted Rademacher wild-bootstrap distribution.
    Omega = score * score'
    return Omega, rows_sigma, rows_psi_ar1, rows_psi_direct, gtmat, keys_gt,
           score_hc2, score_hc3, error_hc2, error_hc3
end

# ------------------------------------------------------------------------------
# Public: pre-outcome design vetting
# ------------------------------------------------------------------------------

"""
    twfe_design(unit, time, first_treat; alpha=0.05, delta=0.05) -> AdequacyReport

Pre-outcome design vetting from the adoption pattern ALONE: the design statistic
`Gamma = sqrt(N1)||w - u||` and its restricted variants `Gamma_coh`, `Gamma_evt`,
`Gamma_c+e`, and `Gamma_gt` (Prop. prop-restricted), the negative-weight share,
and breakdown heterogeneity-to-noise ratios `(c/sigma)† = eta†/Gamma_S`.
In a balanced panel `Gamma_gt = Gamma`. `first_treat` is the
unit's adoption time on the same scale as `time` (`missing` = never treated;
values before the sample = always-treated, which are dropped).
"""
function twfe_design(unit::AbstractVector, time::AbstractVector{<:Real},
                     first_treat::AbstractVector;
                     alpha::Real=0.05, delta::Real=0.05)
    uid, tid, N0, T, ftc0 = _staggered_codes(unit, time, first_treat)
    uid, tid, ftc, _, n_drop = _drop_always_treated(uid, tid, ftc0)
    N = maximum(uid); n = length(uid)
    ft_expand = Float64[ftc[uid[k]] for k in 1:n]
    D = Float64[(isfinite(ft_expand[k]) && tid[k] >= ft_expand[k]) ? 1.0 : 0.0 for k in 1:n]
    Dt = _twoway_demean(D, uid, tid, N, T)
    treated, N1, n_w, w, Gamma, neg_share = _design_stats(D, Dt)
    g_cell = Float64[ftc[uid[k]] for k in treated]
    t_cell = Int[tid[k] for k in treated]
    e_cell = Float64[t_cell[i] - g_cell[i] for i in 1:N1]
    G = _restricted_gammas(w, g_cell, t_cell, e_cell, N1)
    d_K, ncomp = fe_dimension(uid, tid, N, T)
    design = _design_summary_codes(uid, tid, N, T; xt=Dt)
    eta_dag = _eta_dagger(alpha, delta)

    notes = String[]
    n_drop > 0 && push!(notes, @sprintf("%d always-treated unit(s) dropped (no observed untreated period; setup g >= 2)", n_drop))
    statistic = (Gamma=Gamma, Gamma_coh=G.coh, Gamma_evt=G.evt,
                 Gamma_cmb=G.cmb, Gamma_gt=G.gt,
                 neg_share=neg_share, N1=N1, n_w=n_w)
    if Gamma <= 1e-8
        push!(notes, "block design: within-transformed treatment is uniform, so NO heterogeneity profile distorts the t-test (Prop. prop-gamma0)")
        return AdequacyReport(:twfe_heterogeneity, design, statistic, nothing,
                              eta_dag, Inf, nothing, :CERTIFIED, Float64(alpha),
                              Float64(delta), notes)
    end
    breakdown = eta_dag / G.gt
    push!(notes, @sprintf("pre-outcome: the saturated group-time class requires c/sigma <= %.3f (= eta†/Gamma_gt); supply the outcome (twfe_adequacy) to calibrate c/sigma and obtain a boundary-robust certificate", breakdown))
    return AdequacyReport(:twfe_heterogeneity, design, statistic, nothing,
                          eta_dag, breakdown, nothing, :INCONCLUSIVE,
                          Float64(alpha), Float64(delta), notes)
end

"""
    twfe_gammas(unit, time, first_treat) -> NamedTuple

The design-statistic ladder `(unr, coh, evt, cmb, gt, neg_share, N1, n_w)` from
the adoption pattern (always-treated dropped). Convenience accessor for scripts.
"""
function twfe_gammas(unit::AbstractVector, time::AbstractVector{<:Real},
                     first_treat::AbstractVector)
    r = twfe_design(unit, time, first_treat)
    s = r.statistic
    return (unr=s.Gamma, coh=s.Gamma_coh, evt=s.Gamma_evt,
            cmb=s.Gamma_cmb, gt=s.Gamma_gt,
            neg_share=s.neg_share, N1=s.N1, n_w=s.n_w)
end

# ------------------------------------------------------------------------------
# Public: full inference layer
# ------------------------------------------------------------------------------

"""
    twfe_adequacy(y, unit, time, first_treat; alpha=0.05, delta=0.05,
                   cluster=:direct, psi=nothing, controls=:not_yet,
                   heterogeneity_class=:group_time, bootstrap=999,
                   seed=20260715, gamma=0.05, q_band=nothing)
        -> AdequacyReport

Full TWFE-heterogeneity audit. Computes the restricted design-statistic
ladder, direct CR1 rescaling `Gamma_{S,CR} = Gamma_S/sqrt(psi_hat)`, trace-
debiased point pilots `c_S/sigma`, saturated and restricted point envelopes,
and the signed directional plug-in. A fixed-design cluster-multiplier radius
gives the regular one-sided lower bound. The upper certificate inverts the
noncentral chi-square law of the projected Wald statistic and is issued only
when the score covariance has full rank in the prespecified class.

- `bootstrap`: number of wild-cluster draws (`>0` enables the covariance
  correction, descriptive summaries, and one-sided procedures; `0` falls back
  to the raw pilot without a certificate).
- `gamma`: error probability for each reported one-sided procedure.
- `cluster = :direct` (default), `:ar1`, or `:iid`; a positive `psi` overrides it.
- `controls = :not_yet` (including never-treated) or `:never`.
- `heterogeneity_class = :group_time` (default), `:additive`, `:cohort`, or
  `:event` selects the prespecified class used for the report verdict.
- `q_band`: optional relative half-width for the denominator band in the more
  general full confidence-ball construction. It does not affect the main
  decision-specific verdict.
- Always-treated units are dropped (setup g >= 2), with a note.
"""
function twfe_adequacy(y::AbstractVector{<:Real}, unit::AbstractVector,
                       time::AbstractVector{<:Real}, first_treat::AbstractVector;
                       alpha::Real=0.05, delta::Real=0.05,
                       cluster::Symbol=:direct, psi::Union{Nothing,Real}=nothing,
                       controls::Symbol=:not_yet,
                       heterogeneity_class::Symbol=:group_time,
                       bootstrap::Integer=999, seed::Integer=20260715,
                       gamma::Real=0.05,
                       q_band::Union{Nothing,Real}=nothing)
    cluster in (:direct, :ar1, :iid) ||
        throw(ArgumentError("cluster must be :direct, :ar1, or :iid"))
    controls in (:not_yet, :never) ||
        throw(ArgumentError("controls must be :not_yet or :never"))
    heterogeneity_class in (:group_time, :additive, :cohort, :event) ||
        throw(ArgumentError("heterogeneity_class must be :group_time, :additive, :cohort, or :event"))
    psi === nothing || (isfinite(psi) && psi > 0) ||
        throw(ArgumentError("psi must be positive and finite"))
    isfinite(gamma) && 0 < gamma < 0.5 ||
        throw(ArgumentError("gamma must be in (0, 0.5)"))
    q_band === nothing || (isfinite(q_band) && 0 <= q_band < 1) ||
        throw(ArgumentError("q_band must be nothing or in [0, 1)"))
    uid0, tid0, N0, T, ftc0 = _staggered_codes(unit, time, first_treat)
    n0 = length(uid0)
    length(y) == n0 || throw(ArgumentError("y must have length n = $n0"))
    yv0 = Float64.(y)
    uid, tid, ftc, keep, n_drop = _drop_always_treated(uid0, tid0, ftc0)
    yv = yv0[keep]
    N = maximum(uid); n = length(uid)
    ft_expand = Float64[ftc[uid[k]] for k in 1:n]
    D = Float64[(isfinite(ft_expand[k]) && tid[k] >= ft_expand[k]) ? 1.0 : 0.0 for k in 1:n]
    Dt = _twoway_demean(D, uid, tid, N, T)
    treated, N1, n_w, w, Gamma, neg_share = _design_stats(D, Dt)
    g_cell = Float64[ftc[uid[k]] for k in treated]
    t_cell = Int[tid[k] for k in treated]
    e_cell = Float64[t_cell[i] - g_cell[i] for i in 1:N1]
    G = _restricted_gammas(w, g_cell, t_cell, e_cell, N1)
    d_K, ncomp = fe_dimension(uid, tid, N, T)
    dof = n - d_K - 1
    dof > 0 || throw(ArgumentError("no residual degrees of freedom"))

    yt = _twoway_demean(yv, uid, tid, N, T)
    beta = dot(Dt, yt) / n_w
    resid = yt .- beta .* Dt
    sigma = sqrt(sum(abs2, resid) / dof)

    notes = String[]
    n_drop > 0 && push!(notes, @sprintf("%d always-treated unit(s) dropped (setup g >= 2)", n_drop))

    # ---- cluster layer ----
    rho_ar1 = _rho_ar1(resid, uid, tid)
    psi_ar1 = _psi_parametric(Dt, uid, tid, rho_ar1; kind=:ar1)
    psi_direct = _psi_direct(Dt, resid, uid, n_w, sigma^2, N)
    normalization = psi !== nothing ? :user_supplied : cluster
    psi_hat = psi !== nothing ? Float64(psi) :
              cluster === :direct ? psi_direct : cluster === :ar1 ? psi_ar1 : 1.0
    CR = (unr=G.unr/sqrt(psi_hat), coh=G.coh/sqrt(psi_hat),
          evt=G.evt/sqrt(psi_hat), cmb=G.cmb/sqrt(psi_hat),
          gt=G.gt/sqrt(psi_hat))

    # ---- group-time ATTs (point) ----
    gt0, cohorts = _group_time_atts(uid, tid, ftc, yv, T; controls=controls)

    # ---- covariance-aware pilots via the wild bootstrap ----
    bases = (Bcoh=G.Bcoh, Bevt=G.Bevt, Bcmb=G.Bcmb, Bgt=G.Bgt)
    directional = _directional_profile(gt0, g_cell, t_cell, N1, G.Bcmb,
                                       w, n_w, sigma, psi_hat)
    delta0 = _delta_cell(gt0, g_cell, t_cell, N1)
    local pilots, size_pt, boot, formal
    formal = nothing
    if bootstrap > 0 && !isempty(gt0)
        Omega, bsig, bpsi_ar1, bpsi_direct, bgt, keys_gt,
        score_hc2, score_hc3, error_hc2, error_hc3 =
            _wild_bootstrap(uid, tid, ftc, D, Dt, yv, N, T, n_w, treated, N1,
                            g_cell, t_cell, d_K, gt0, Int(bootstrap), Int(seed),
                            controls)
        # Incidence A and centered class bases on the fixed group-time keys.
        kidx = Dict(k => j for (j, k) in enumerate(keys_gt))
        m = length(keys_gt)
        A = zeros(N1, m)
        for k in 1:N1
            key = (Int(g_cell[k]), t_cell[k])
            haskey(kidx, key) && (A[k, kidx[key]] = 1.0)
        end
        Q = (coh=_centered_basis(G.Bcoh), evt=_centered_basis(G.Bevt),
             cmb=_centered_basis(G.Bcmb), gt=_centered_basis(G.Bgt))
        traces = _cov_traces(A, Omega, Q, N1)
        pilots = _cov_pilots(gt0, g_cell, t_cell, N1, bases, traces, n_w, sigma)
        size_pt = (coh=_noncentral_size(pilots.coh * CR.coh, alpha),
                   evt=_noncentral_size(pilots.evt * CR.evt, alpha),
                   cmb=_noncentral_size(pilots.cmb * CR.cmb, alpha),
                   gt=_noncentral_size(pilots.gt * CR.gt, alpha))
        # Per-draw envelope and directional sizes under the selected scale.
        bpsi = psi !== nothing ? fill(Float64(psi), length(bsig)) :
               cluster === :direct ? bpsi_direct :
               cluster === :ar1 ? bpsi_ar1 : ones(length(bsig))
        draw_envelope = Float64[]; draw_directional = Float64[]
        draw_directional_eta = Float64[]; draw_alignment = Float64[]
        for i in eachindex(bsig)
            gt = Dict(zip(keys_gt, bgt[i]))
            p = _cov_pilots(gt, g_cell, t_cell, N1, bases, traces, n_w, bsig[i])
            push!(draw_envelope,
                  _noncentral_size(p.cmb * G.cmb / sqrt(bpsi[i]), alpha))
            dp = _directional_profile(gt, g_cell, t_cell, N1, G.Bcmb,
                                      w, n_w, bsig[i], bpsi[i])
            push!(draw_directional_eta, dp.eta)
            push!(draw_directional, _noncentral_size(dp.eta, alpha))
            push!(draw_alignment, dp.alignment)
        end
        qfinite(v, p) = quantile(sort(filter(isfinite, v)), p)
        envelope_med=qfinite(draw_envelope, 0.5)
        envelope_lo=qfinite(draw_envelope, 0.025)
        envelope_hi=qfinite(draw_envelope, 0.975)
        envelope_p95=qfinite(draw_envelope, 0.95)

        # The trace-debiased quadratic remains a descriptive point calibration.
        # The lower test uses reverse triangle inequality and the multiplier
        # radius. The upper certificate uses projected-Wald noncentrality
        # inversion and therefore checks covariance rank.
        if all(isfinite, delta0) && all(vec(sum(A; dims=2)) .== 1)
            projected_scale(q) = sqrt(n_w / N1) * norm(q' * (delta0 .- mean(delta0)))
            function error_norms(q, E)
                projected_coordinates = (q' * A) * E
                return sqrt(n_w / N1) .* [norm(view(projected_coordinates, :, j))
                                           for j in axes(projected_coordinates, 2)]
            end
            scales = (coh=projected_scale(Q.coh), evt=projected_scale(Q.evt),
                      cmb=projected_scale(Q.cmb), gt=projected_scale(Q.gt))
            radii = (coh=qfinite(error_norms(Q.coh, error_hc2), 1-gamma),
                     evt=qfinite(error_norms(Q.evt, error_hc2), 1-gamma),
                     cmb=qfinite(error_norms(Q.cmb, error_hc2), 1-gamma),
                     gt=qfinite(error_norms(Q.gt, error_hc2), 1-gamma))
            radii_hc3 = (coh=qfinite(error_norms(Q.coh, error_hc3), 1-gamma),
                         evt=qfinite(error_norms(Q.evt, error_hc3), 1-gamma),
                         cmb=qfinite(error_norms(Q.cmb, error_hc3), 1-gamma),
                         gt=qfinite(error_norms(Q.gt, error_hc3), 1-gamma))
            q_scale = sigma * sqrt(psi_hat)
            se_scale = q_scale / sqrt(n_w)
            lower = (coh=G.coh*max(0, scales.coh-radii.coh)/q_scale,
                     evt=G.evt*max(0, scales.evt-radii.evt)/q_scale,
                     cmb=G.cmb*max(0, scales.cmb-radii.cmb)/q_scale,
                     gt=G.gt*max(0, scales.gt-radii.gt)/q_scale)
            lower_hc3 = (coh=G.coh*max(0, scales.coh-radii_hc3.coh)/q_scale,
                          evt=G.evt*max(0, scales.evt-radii_hc3.evt)/q_scale,
                          cmb=G.cmb*max(0, scales.cmb-radii_hc3.cmb)/q_scale,
                          gt=G.gt*max(0, scales.gt-radii_hc3.gt)/q_scale)
            equivalence = (
                coh=_equivalence_upper(delta0, Q.coh, A, score_hc2, G.coh,
                                       se_scale, gamma),
                evt=_equivalence_upper(delta0, Q.evt, A, score_hc2, G.evt,
                                       se_scale, gamma),
                cmb=_equivalence_upper(delta0, Q.cmb, A, score_hc2, G.cmb,
                                       se_scale, gamma),
                gt=_equivalence_upper(delta0, Q.gt, A, score_hc2, G.gt,
                                      se_scale, gamma))
            equivalence_hc3 = (
                coh=_equivalence_upper(delta0, Q.coh, A, score_hc3, G.coh,
                                       se_scale, gamma),
                evt=_equivalence_upper(delta0, Q.evt, A, score_hc3, G.evt,
                                       se_scale, gamma),
                cmb=_equivalence_upper(delta0, Q.cmb, A, score_hc3, G.cmb,
                                       se_scale, gamma),
                gt=_equivalence_upper(delta0, Q.gt, A, score_hc3, G.gt,
                                      se_scale, gamma))
            upper = (coh=equivalence.coh.K_upper, evt=equivalence.evt.K_upper,
                     cmb=equivalence.cmb.K_upper, gt=equivalence.gt.K_upper)
            upper_hc3 = (coh=equivalence_hc3.coh.K_upper,
                         evt=equivalence_hc3.evt.K_upper,
                         cmb=equivalence_hc3.cmb.K_upper,
                         gt=equivalence_hc3.gt.K_upper)
            ball_upper_raw = (coh=G.coh*(scales.coh+radii.coh)/q_scale,
                              evt=G.evt*(scales.evt+radii.evt)/q_scale,
                              cmb=G.cmb*(scales.cmb+radii.cmb)/q_scale,
                              gt=G.gt*(scales.gt+radii.gt)/q_scale)
            ball_upper_raw_hc3 = (
                coh=G.coh*(scales.coh+radii_hc3.coh)/q_scale,
                evt=G.evt*(scales.evt+radii_hc3.evt)/q_scale,
                cmb=G.cmb*(scales.cmb+radii_hc3.cmb)/q_scale,
                gt=G.gt*(scales.gt+radii_hc3.gt)/q_scale)
            ball_lower = q_band === nothing ?
                (coh=NaN, evt=NaN, cmb=NaN, gt=NaN) :
                (coh=(1-q_band)*lower.coh, evt=(1-q_band)*lower.evt,
                 cmb=(1-q_band)*lower.cmb, gt=(1-q_band)*lower.gt)
            ball_upper = q_band === nothing ?
                (coh=NaN, evt=NaN, cmb=NaN, gt=NaN) :
                (coh=(1+q_band)*ball_upper_raw.coh,
                 evt=(1+q_band)*ball_upper_raw.evt,
                 cmb=(1+q_band)*ball_upper_raw.cmb,
                 gt=(1+q_band)*ball_upper_raw.gt)
            formal = (confidence=1-Float64(gamma), scale=scales,
                       radius=radii, radius_hc3=radii_hc3,
                       lower=lower, upper=upper,
                       lower_hc3=lower_hc3, upper_hc3=upper_hc3,
                       equivalence=equivalence,
                       equivalence_hc3=equivalence_hc3,
                       ball_upper_raw=ball_upper_raw,
                       ball_upper_raw_hc3=ball_upper_raw_hc3,
                       ball_lower=ball_lower, ball_upper=ball_upper,
                       q_band=q_band)
        else
            push!(notes, "one-sided procedures unavailable: the group-time estimator does not cover every treated cell")
        end

        boot = (n=length(bsig), envelope_med=envelope_med,
                envelope_lo=envelope_lo, envelope_hi=envelope_hi,
                envelope_p95=envelope_p95,
                directional_med=qfinite(draw_directional, 0.5),
                directional_lo=qfinite(draw_directional, 0.025),
                directional_hi=qfinite(draw_directional, 0.975),
                directional_eta_med=qfinite(draw_directional_eta, 0.5),
                directional_eta_lo=qfinite(draw_directional_eta, 0.025),
                directional_eta_hi=qfinite(draw_directional_eta, 0.975),
                alignment_med=qfinite(draw_alignment, 0.5),
                psi_lo=qfinite(bpsi, 0.025), psi_hi=qfinite(bpsi, 0.975),
                psi_direct_lo=qfinite(bpsi_direct, 0.025),
                psi_direct_hi=qfinite(bpsi_direct, 0.975),
                psi_ar1_lo=qfinite(bpsi_ar1, 0.025),
                psi_ar1_hi=qfinite(bpsi_ar1, 0.975),
                Omega_trace_cmb=traces.cmb,
                Omega_trace_gt=traces.gt,
                norm=formal,
                # v0.6 compatibility aliases: these are envelope fields.
                cmb_med=envelope_med, cmb_lo=envelope_lo,
                cmb_hi=envelope_hi, cmb_p95=envelope_p95)
        push!(notes, @sprintf("point pilot: covariance trace computed from the fitted Rademacher score covariance; %d multiplier draws calibrate the %.1f%% HC2 lower norm radius; the upper procedure uses projected-Wald noncentrality inversion", boot.n, 100*(1-gamma)))
    else
        # fallback: raw (uncorrected) projected dispersion, flagged
        rawpil(B) = begin
            delta = _delta_cell(gt0, g_cell, t_cell, N1)
            cov = .!isnan.(delta); dc = delta .- mean(delta[cov])
            comp = _proj(B[cov, :], dc[cov])
            sqrt(n_w * dot(comp, comp) / count(cov) / sigma^2)
        end
        pilots = (coh=rawpil(G.Bcoh), evt=rawpil(G.Bevt),
                  cmb=rawpil(G.Bcmb), gt=rawpil(G.Bgt))
        size_pt = (coh=_noncentral_size(pilots.coh * CR.coh, alpha),
                   evt=_noncentral_size(pilots.evt * CR.evt, alpha),
                   cmb=_noncentral_size(pilots.cmb * CR.cmb, alpha),
                   gt=_noncentral_size(pilots.gt * CR.gt, alpha))
        boot = nothing
        push!(notes, "bootstrap disabled: pilots are raw projected dispersion and no boundary-robust certificate is available; set bootstrap > 0")
    end

    push!(notes, @sprintf("normalization = %s: psi_hat = %.3f; direct CR1 psi = %.3f; AR(1) psi = %.3f (rho = %.3f)",
                          String(normalization), psi_hat, psi_direct, psi_ar1, rho_ar1))
    !isfinite(directional.eta) && push!(notes,
        "directional plug-in unavailable because the group-time profile does not cover every treated cell")
    N < 40 && push!(notes, @sprintf("few clusters (G = %d): CR3/jackknife or wild bootstrap refinements advisable (Rem. sec-clusters)", N))
    T / N > 0.25 && push!(notes, @sprintf(
        "fixed-T, many-cluster approximation is strained (G = %d, T = %d)", N, T))

    eta_dag = _eta_dagger(alpha, delta)
    selected_key = (group_time=:gt, additive=:cmb, cohort=:coh,
                    event=:evt)[heterogeneity_class]
    class_names = (coh="cohort", evt="event-time", cmb="additive",
                   gt="saturated group-time")
    function decide(key)
        formal === nothing && return :INCONCLUSIVE
        lower = getproperty(formal.lower, key)
        isfinite(lower) || return :INCONCLUSIVE
        lower > eta_dag && return :FLAGGED
        eq = getproperty(formal.equivalence, key)
        upper = getproperty(formal.upper, key)
        return eq.supported && isfinite(upper) && upper <= eta_dag ?
               :CERTIFIED : :INCONCLUSIVE
    end
    class_verdicts = (coh=decide(:coh), evt=decide(:evt),
                      cmb=decide(:cmb), gt=decide(:gt))
    verdict = getproperty(class_verdicts, selected_key)
    eta_selected = getproperty(pilots, selected_key) * getproperty(CR, selected_key)
    if formal !== nothing
        eq_selected = getproperty(formal.equivalence, selected_key)
        class_name = getproperty(class_names, selected_key)
        lower_selected = getproperty(formal.lower, selected_key)
        upper_selected = getproperty(formal.upper, selected_key)
        if verdict === :CERTIFIED
            push!(notes, @sprintf("%s upper bound %.3f is below eta† %.3f",
                                  class_name, upper_selected, eta_dag))
        elseif verdict === :FLAGGED
            push!(notes, @sprintf("%s lower bound %.3f exceeds eta† %.3f; uniform certification is withheld, not the published inference invalidated",
                                  class_name, lower_selected, eta_dag))
        elseif !eq_selected.supported
            push!(notes, @sprintf("%s upper certificate unavailable: projected score covariance rank %d/%d; result is inconclusive for this structural reason",
                                  class_name, eq_selected.covariance_rank,
                                  eq_selected.expected_rank))
        else
            push!(notes, @sprintf("%s one-sided lower %.3f and upper %.3f do not determine a verdict at eta† %.3f",
                                  class_name, lower_selected, upper_selected,
                                  eta_dag))
        end
        q_band !== nothing && push!(notes, @sprintf("general confidence-ball fallback reported with denominator band q_band = %.4f; it does not determine the main verdict", q_band))
    end
    design = _design_summary_codes(uid, tid, N, T; xt=Dt)
    statistic = (Gamma=Gamma, Gamma_coh=G.coh, Gamma_evt=G.evt,
                 Gamma_cmb=G.cmb, Gamma_gt=G.gt,
                 neg_share=neg_share, psi_hat=psi_hat, normalization=normalization,
                 psi_direct=psi_direct, psi_ar1=psi_ar1,
                 psi_driven=psi_direct, rho_ar1=rho_ar1,
                 Gamma_CR=CR.unr, Gamma_coh_CR=CR.coh,
                 Gamma_evt_CR=CR.evt, Gamma_cmb_CR=CR.cmb,
                 Gamma_gt_CR=CR.gt, beta=beta, sigma=sigma,
                 se_cr1=sigma * sqrt(psi_direct / n_w),
                 q_hat=sigma * sqrt(psi_direct),
                 sign_reversal_rms=Gamma > 0 ? abs(beta) / Gamma : Inf,
                 N1=N1, n_w=n_w, n_cohorts=length(cohorts),
                 att_target=all(isfinite, delta0) ? mean(delta0) : NaN,
                 pilot_coh=pilots.coh, pilot_evt=pilots.evt,
                 pilot_cmb=pilots.cmb, pilot_gt=pilots.gt,
                  size_coh=size_pt.coh, size_evt=size_pt.evt,
                  size_cmb=size_pt.cmb, size_gt=size_pt.gt,
                  selected_class=heterogeneity_class,
                  verdict_coh=class_verdicts.coh,
                  verdict_evt=class_verdicts.evt,
                  verdict_cmb=class_verdicts.cmb,
                  verdict_gt=class_verdicts.gt,
                  K_lower_coh=formal === nothing ? NaN : formal.lower.coh,
                  K_upper_coh=formal === nothing ? NaN : formal.upper.coh,
                  K_lower_evt=formal === nothing ? NaN : formal.lower.evt,
                  K_upper_evt=formal === nothing ? NaN : formal.upper.evt,
                  K_lower_cmb=formal === nothing ? NaN : formal.lower.cmb,
                  K_upper_cmb=formal === nothing ? NaN : formal.upper.cmb,
                  K_lower_gt=formal === nothing ? NaN : formal.lower.gt,
                  K_upper_gt=formal === nothing ? NaN : formal.upper.gt,
                 K_lower_gt_hc3=formal === nothing ? NaN : formal.lower_hc3.gt,
                  K_upper_gt_hc3=formal === nothing ? NaN : formal.upper_hc3.gt,
                  expected_rank_coh=formal === nothing ? 0 : formal.equivalence.coh.expected_rank,
                  covariance_rank_coh=formal === nothing ? 0 : formal.equivalence.coh.covariance_rank,
                  expected_rank_evt=formal === nothing ? 0 : formal.equivalence.evt.expected_rank,
                  covariance_rank_evt=formal === nothing ? 0 : formal.equivalence.evt.covariance_rank,
                  expected_rank_cmb=formal === nothing ? 0 : formal.equivalence.cmb.expected_rank,
                  covariance_rank_cmb=formal === nothing ? 0 : formal.equivalence.cmb.covariance_rank,
                  expected_rank_gt=formal === nothing ? 0 : formal.equivalence.gt.expected_rank,
                  covariance_rank_gt=formal === nothing ? 0 : formal.equivalence.gt.covariance_rank,
                  q_band=q_band === nothing ? NaN : Float64(q_band),
                  K_ball_lower_gt=formal === nothing ? NaN : formal.ball_lower.gt,
                  K_ball_upper_gt=formal === nothing ? NaN : formal.ball_upper.gt,
                  gamma=Float64(gamma),
                 eta_directional=directional.eta,
                 size_directional=_noncentral_size(directional.eta, alpha),
                 directional_alignment=directional.alignment,
                 # Deprecated v0.6 aliases retained for code compatibility.
                 eta_real_cr=directional.eta,
                 size_realized=_noncentral_size(directional.eta, alpha),
                 controls=controls, boot=boot)
    return AdequacyReport(:twfe_heterogeneity, design, statistic, eta_selected,
                           eta_dag, eta_dag / getproperty(CR, selected_key),
                           getproperty(size_pt, selected_key), verdict,
                           Float64(alpha), Float64(delta), notes)
end

# ------------------------------------------------------------------------------
# psi machinery (Paper C Def. def-psi, sec-feasible)
# ------------------------------------------------------------------------------

"Pooled within-unit lag-1 residual autocorrelation (consecutive periods only)."
function _rho_ar1(resid::Vector{Float64}, uid::Vector{Int}, tid::Vector{Int})
    order = sortperm(collect(zip(uid, tid)))
    num = den = 0.0
    for j in 2:length(order)
        a, b = order[j], order[j-1]
        if uid[a] == uid[b] && tid[a] == tid[b] + 1
            num += resid[a] * resid[b]
            den += resid[b]^2
        end
    end
    return den > 0 ? num / den : 0.0
end

"psi = sum_i d_i' R_i d_i / n_w for parametric R (:ar1 with |t-s| gaps, or :exchangeable)."
function _psi_parametric(Dt::Vector{Float64}, uid::Vector{Int}, tid::Vector{Int},
                         rho::Float64; kind::Symbol=:ar1)
    N = maximum(uid)
    n_w = sum(abs2, Dt)
    paths = [Tuple{Int,Float64}[] for _ in 1:N]
    for k in eachindex(Dt)
        push!(paths[uid[k]], (tid[k], Dt[k]))
    end
    nwcr = 0.0
    for p in paths
        isempty(p) && continue
        if kind === :exchangeable
            s = sum(v for (_, v) in p)
            nwcr += (1 - rho) * sum(abs2(v) for (_, v) in p) + rho * s^2
        elseif kind === :ar1
            sort!(p)
            for a in eachindex(p), b in eachindex(p)
                nwcr += p[a][2] * p[b][2] * rho^abs(p[a][1] - p[b][1])
            end
        else
            throw(ArgumentError("kind must be :ar1 or :exchangeable"))
        end
    end
    return nwcr / n_w
end

"Direct CR1 psi from cluster scores over the homoskedastic residual scale."
function _psi_direct(Dt::Vector{Float64}, resid::Vector{Float64},
                     uid::Vector{Int}, n_w::Float64, sigma2::Float64, G::Int)
    meat = zeros(G)
    for k in eachindex(Dt)
        meat[uid[k]] += Dt[k] * resid[k]
    end
    return (G / (G - 1)) * sum(abs2, meat) / (n_w * sigma2)
end

const _psi_driven = _psi_direct
