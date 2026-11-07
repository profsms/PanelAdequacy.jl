# =============================================================================
# Module D — stepped-wedge design diagnostics (Halkiewicz and Alexeev,
# "Concurrent controls and heterogeneity tolerance in stepped-wedge trials",
# arXiv:2610.35853). Mirrors R/panelcert/R/stepped_wedge.R.
#
# A design is a vector of length T: cluster counts (or shares) for start
# periods 2..T followed by the wait-list. Exposure e = t - g + 1, L = T - 1.
# Equal sizes: S = Σ_g n_g X_g'ΩX_g − G X̄'ΩX̄ with Ω = V⁻¹ (TWFE: Ω = I − J/T).
# Unequal sizes: cluster-level GLS with explicit period effects and
# V_c = diag((1−ρ)/m_ct) + (1−r)ρ I + rρ J, or cell-size-weighted TWFE.
# =============================================================================

function _sw_check_design(design)
    length(design) >= 3 || throw(ArgumentError(
        "`design` must have length T >= 3: counts for start periods 2..T, then the wait-list"))
    d = Float64.(collect(design))
    all(isfinite, d) && all(>=(0), d) || throw(ArgumentError("`design` entries must be finite and non-negative"))
    d[1] > 0 || throw(ArgumentError(
        "the first switching sequence (start period 2) must be non-empty, so that every exposure 1..T-1 is observed"))
    return d
end

function _sw_X(T::Int, g::Int, model::Symbol)
    model === :IT && return reshape(Float64[t >= g ? 1.0 : 0.0 for t in 1:T], T, 1)
    L = T - 1
    Z = zeros(T, L)
    for t in 1:T
        e = t - g + 1
        1 <= e <= L && (Z[t, e] = 1.0)
    end
    return Z
end

function _sw_omega(T::Int; phi=nothing, icc=nothing, cac=1.0, m=nothing, estimator::Symbol=:GLS)
    Q = Matrix{Float64}(I, T, T) .- 1.0 / T
    P = fill(1.0 / T, T, T)
    estimator === :TWFE && return Q
    estimator === :GLS || throw(ArgumentError("estimator must be :GLS or :TWFE"))
    phi !== nothing && icc !== nothing && throw(ArgumentError(
        "supply either `phi` or (`icc`, `cac`, `m`), not both"))
    if phi !== nothing
        phi >= 0 || throw(ArgumentError("`phi` must be non-negative"))
        lam = isinf(phi) ? 0.0 : 1 / (1 + T * phi)
        return Q .+ lam .* P
    end
    (icc === nothing || m === nothing) && throw(ArgumentError("GLS needs `phi`, or `icc`, `cac` and `m`"))
    php = sw_phi_nested(icc, cac, m)
    c0 = (1 - cac) * icc + (1 - icc) / m
    return (Q .+ P ./ (1 + T * php)) ./ c0
end

function _sw_equal_info(design::Vector{Float64}, Omega::Matrix{Float64}, model::Symbol)
    T = length(design)
    G = sum(design)
    gs = 2:(T + 1)
    Xs = [_sw_X(T, g, model) for g in gs]
    Xbar = sum(n .* X for (n, X) in zip(design, Xs)) ./ G
    k = size(Xs[1], 2)
    S = zeros(k, k)
    for (n, X) in zip(design, Xs)
        n > 0 && (S .+= n .* (X' * Omega * X))
    end
    S .-= G .* (Xbar' * Omega * Xbar)
    return (S=S, Xs=Xs, Xbar=Xbar, gs=collect(gs))
end

function _sw_solve(S::AbstractMatrix, what::AbstractString)
    rc = try
        1 / cond(S, 1)
    catch
        0.0
    end
    (isfinite(rc) && rc >= 1e-12) || throw(ArgumentError(
        "the design does not identify the $what (singular information)"))
    return inv(S)
end

function _sw_cluster_fit(design::Vector{Float64}, sizes::AbstractMatrix, icc, cac, estimator::Symbol, model::Symbol)
    T = length(design)
    all(d -> d == round(d), design) || throw(ArgumentError("with `sizes`, `design` must contain integer cluster counts"))
    starts = reduce(vcat, [fill(g, Int(n)) for (g, n) in zip(2:(T + 1), design)])
    size(sizes) == (length(starts), T) || throw(ArgumentError(
        "`sizes` must be a $(length(starts)) x $T matrix (clusters in design order: sequence 2 first, wait-list last)"))
    all(x -> isfinite(x) && x > 0, sizes) || throw(ArgumentError("`sizes` must be positive"))
    twfe = estimator === :TWFE
    P = Matrix{Float64}(I, T, T)
    twfe && (P = P[:, 2:end])
    Om = Matrix{Float64}[]
    for i in eachindex(starts)
        m = Float64.(sizes[i, :])
        if twfe
            push!(Om, Diagonal(m) .- (m * m') ./ sum(m))
        else
            icc === nothing && throw(ArgumentError("GLS with `sizes` needs `icc` (and `cac`)"))
            V = Diagonal((1 - icc) ./ m) .+ (1 - cac) * icc .* Matrix{Float64}(I, T, T) .+ cac * icc
            push!(Om, inv(Matrix(V)))
        end
    end
    Xs = [hcat(P, _sw_X(T, g, model)) for g in starts]
    M = sum(X' * O * X for (X, O) in zip(Xs, Om))
    Mi = _sw_solve(M, String(model) * " model")
    p = size(P, 2)
    return (Mi=Mi, Xs=Xs, Om=Om, starts=starts, idx=(p + 1):size(M, 1), T=T)
end

"""
    sw_phi_nested(icc, cac, m) -> Float64

Effective Hussey–Hughes parameter under cross-sectional nested-exchangeable
correlation with within-period ICC `icc` (ρ), cluster autocorrelation `cac` (r)
and `m` individuals per cluster-period:
`φ′ = m r ρ / {m(1−r)ρ + 1 − ρ}` (Corollary 1 of Halkiewicz and Alexeev).
Every Hussey–Hughes result holds with φ replaced by φ′; at `cac = 1`,
`φ′ = mρ/(1−ρ)`.

# Example
```julia
sw_phi_nested(0.05, 0.5, 20)
```
"""
function sw_phi_nested(icc::Real, cac::Real, m::Real)
    0 <= icc < 1 || throw(ArgumentError("`icc` must lie in [0, 1)"))
    0 <= cac <= 1 || throw(ArgumentError("`cac` must lie in [0, 1]"))
    m > 0 || throw(ArgumentError("`m` must be positive"))
    return m * cac * icc / (m * (1 - cac) * icc + 1 - icc)
end

"""
    sw_exposure_weights(design; phi=nothing, icc=nothing, cac=1.0, m=nothing,
                        estimator=:GLS, sizes=nothing) -> Vector{Float64}

Exposure-time weights `W_e`, `e = 1..T-1`, of the immediate-treatment (IT)
estimator of a stepped-wedge / wait-list design, so that `E(β̂) = Σ W_e θ_e`.

`design` has length `T`: cluster counts (or shares) for start periods `2..T`,
then the wait-list. Covariance: Hussey–Hughes `phi` (φ = mρ/(1−ρ); `Inf` gives
the fixed-effects limit), or nested-exchangeable `icc`, `cac`, `m`.
`estimator = :TWFE` uses two-way fixed effects. `sizes` is an optional
`G × T` matrix of cluster-period sizes (rows in design order); it needs integer
counts and, for GLS, `icc` and `cac`.

# Example
```julia
sw_exposure_weights([6, 6, 6, 6, 0]; phi=2)
```
"""
function sw_exposure_weights(design; phi=nothing, icc=nothing, cac=1.0, m=nothing,
                             estimator::Symbol=:GLS, sizes=nothing)
    d = _sw_check_design(design)
    T = length(d); L = T - 1
    W = zeros(L)
    if sizes === nothing
        Om = _sw_omega(T; phi=phi, icc=icc, cac=cac, m=m, estimator=estimator)
        E = _sw_equal_info(d, Om, :IT)
        h = _sw_solve(E.S, "IT model")[1, 1]
        for (i, g) in enumerate(E.gs)
            d[i] == 0 && continue
            a = d[i] * h .* vec(Om * (E.Xs[i] .- E.Xbar))
            for t in 1:T
                e = t - g + 1
                1 <= e <= L && (W[e] += a[t])
            end
        end
    else
        F = _sw_cluster_fit(d, sizes, icc, cac, estimator, :IT)
        h = F.Mi[F.idx, :]
        for (i, g) in enumerate(F.starts)
            a = vec(h * F.Xs[i]' * F.Om[i])
            for t in 1:T
                e = t - g + 1
                1 <= e <= L && (W[e] += a[t])
            end
        end
    end
    return W
end

function _sw_target(L::Int, target)
    v = target === nothing ? fill(1.0 / L, L) : Float64.(collect(target))
    length(v) == L || throw(ArgumentError("`target` must have length $L"))
    abs(sum(v) - 1) <= 1e-10 || throw(ArgumentError("`target` must sum to one"))
    return v
end

"""
    sw_gamma(W; target=nothing) -> Float64
    sw_gamma(; design, target=nothing, kwargs...) -> Float64

`Γ = √L ‖W − v‖₂`: the worst-case bias of the IT estimator for the target
`v'θ` (default TATE, `v = 1/L`) per unit RMS exposure-time heterogeneity.
The keyword form computes `W` from `design` with [`sw_exposure_weights`](@ref).
"""
sw_gamma(W::AbstractVector; target=nothing) =
    (L = length(W); sqrt(L) * norm(Float64.(W) .- _sw_target(L, target)))
sw_gamma(; design, target=nothing, kwargs...) =
    sw_gamma(sw_exposure_weights(design; kwargs...); target=target)

"""
    sw_radius(W; target=nothing, alpha=0.05, epsilon=0.05, power=0.80) -> Float64
    sw_radius(; design, target=nothing, alpha=0.05, epsilon=0.05, power=0.80, kwargs...)

Adequacy radius `κ* = η_{α,ε} / {(z_{1−α/2} + z_{power}) Γ}` of the IT analysis
(`0.2329/Γ` at the defaults). Example: `sw_radius(design=[6,6,6,6,0], phi=2)`
gives 0.133.
"""
function sw_radius(W::AbstractVector; target=nothing, alpha::Real=0.05, epsilon::Real=0.05, power::Real=0.80)
    return _eta_dagger(alpha, epsilon) / ((_norminv(1 - alpha / 2) + _norminv(power)) * sw_gamma(W; target=target))
end
sw_radius(; design, target=nothing, alpha::Real=0.05, epsilon::Real=0.05, power::Real=0.80, kwargs...) =
    sw_radius(sw_exposure_weights(design; kwargs...); target=target, alpha=alpha, epsilon=epsilon, power=power)

"""
    sw_eti_cost(design; phi=nothing, icc=nothing, cac=1.0, m=nothing,
                estimator=:GLS, sizes=nothing, target=nothing) -> NamedTuple

Variance of the exposure-time-indicator (ETI) estimator of the target (default
TATE) and of the IT estimator, and both relative to the IT variance of the
balanced complete wedge with the same clusters, periods and covariance:
`(var_eti, var_it, var_ref, eti_cost, it_ratio)`.

# Example
```julia
sw_eti_cost([6, 6, 6, 6, 0]; phi=2).eti_cost    # 2.27
```
"""
function sw_eti_cost(design; phi=nothing, icc=nothing, cac=1.0, m=nothing,
                     estimator::Symbol=:GLS, sizes=nothing, target=nothing)
    d = _sw_check_design(design)
    T = length(d); L = T - 1
    c = target === nothing ? fill(1.0 / L, L) : Float64.(collect(target))
    length(c) == L || throw(ArgumentError("`target` must have length $L"))
    G = sum(d)
    mref = m; phiref = phi
    if sizes === nothing
        Om = _sw_omega(T; phi=phi, icc=icc, cac=cac, m=m, estimator=estimator)
        Se = _sw_equal_info(d, Om, :ETI).S
        Si = _sw_equal_info(d, Om, :IT).S
        var_eti = dot(c, _sw_solve(Se, "ETI model") * c)
        var_it = 1 / Si[1, 1]
        (isfinite(var_it) && var_it > 0) || throw(ArgumentError("the design does not identify the IT model"))
    else
        Fe = _sw_cluster_fit(d, sizes, icc, cac, estimator, :ETI)
        Fi = _sw_cluster_fit(d, sizes, icc, cac, estimator, :IT)
        var_eti = dot(c, Fe.Mi[Fe.idx, Fe.idx] * c)
        var_it = Fi.Mi[Fi.idx, Fi.idx][1, 1]
        mref = sum(sizes) / length(sizes)
        estimator === :GLS && (phiref = nothing)
    end
    ref = vcat(fill(G / (T - 1), T - 1), 0.0)
    Oref = (estimator === :TWFE && sizes !== nothing) ?
        _sw_omega(T; estimator=:TWFE) .* mref :
        _sw_omega(T; phi=phiref, icc=icc, cac=cac, m=mref, estimator=estimator)
    var_ref = 1 / _sw_equal_info(ref, Oref, :IT).S[1, 1]
    return (var_eti=var_eti, var_it=var_it, var_ref=var_ref,
            eti_cost=var_eti / var_ref, it_ratio=var_it / var_ref)
end

function _sw_compositions(G::Int, k::Int)
    k == 1 && return [[G]]
    out = Vector{Vector{Int}}()
    for first in 0:G, rest in _sw_compositions(G - first, k - 1)
        push!(out, vcat(first, rest))
    end
    return out
end

"""
    sw_design_search(G, T; cap=Inf, max_wait=G-1, phi=nothing, icc=nothing, cac=1.0,
                     m=nothing, estimator=:GLS, objective=:eti_cost,
                     max_it_ratio=Inf, top=1, alpha=0.05, epsilon=0.05, power=0.80)

Exhaustive search over integer allocations of `G` clusters to start periods
`2..T` and a wait-list (at least one cluster starting in period 2), subject to
a rollout `cap` per period and at most `max_wait` wait-listed clusters (equal
cluster-period sizes). `objective = :eti_cost` minimises the ETI cost;
`:radius` maximises κ* subject to `it_ratio <= max_it_ratio`.

Returns `(best, frontier, designs)`: `designs` is a matrix with columns
`start2..startT, wait, eti_cost, it_ratio, radius` (one row per identified
design, in lexicographic order), `best` its `top` best rows, and `frontier` the
best row for each wait-list bound `0..max_wait` (first column `max_wait`).

# Example
```julia
s = sw_design_search(24, 5; cap=6, max_wait=12, phi=2)
s.frontier[s.frontier[:, 1] .∈ Ref([0, 4, 6, 8, 12]), :]
```
"""
function sw_design_search(G::Integer, T::Integer; cap::Real=Inf, max_wait::Integer=G - 1,
                          phi=nothing, icc=nothing, cac=1.0, m=nothing,
                          estimator::Symbol=:GLS, objective::Symbol=:eti_cost,
                          max_it_ratio::Real=Inf, top::Integer=1,
                          alpha::Real=0.05, epsilon::Real=0.05, power::Real=0.80)
    (T >= 3 && G >= 2) || throw(ArgumentError("need T >= 3 and G >= 2"))
    objective in (:eti_cost, :radius) || throw(ArgumentError("objective must be :eti_cost or :radius"))
    binomial(big(G + T - 1), T - 1) <= 500_000 || throw(ArgumentError("too many allocations; reduce G or T"))
    T = Int(T); G = Int(G)
    comps = filter(n -> n[1] >= 1 && maximum(n[1:(T - 1)]) <= cap && n[T] <= max_wait,
                   _sw_compositions(G, T))
    Om = _sw_omega(T; phi=phi, icc=icc, cac=cac, m=m, estimator=estimator)
    ref = vcat(fill(G / (T - 1), T - 1), 0.0)
    var_ref = 1 / _sw_equal_info(ref, Om, :IT).S[1, 1]
    L = T - 1; u = fill(1.0 / L, L)
    kf = _eta_dagger(alpha, epsilon) / (_norminv(1 - alpha / 2) + _norminv(power))
    gs = 2:(T + 1)
    XE = [_sw_X(T, g, :ETI) for g in gs]
    XI = [vec(_sw_X(T, g, :IT)) for g in gs]
    AE = [X' * Om * X for X in XE]
    AI = [dot(x, Om * x) for x in XI]
    rows = Vector{Vector{Float64}}()
    for n in comps
        Xb = sum(n[i] .* XE[i] for i in eachindex(gs)) ./ G
        Se = sum(n[i] .* AE[i] for i in eachindex(gs)) .- G .* (Xb' * Om * Xb)
        rc = 1 / cond(Se, 1)
        (isfinite(rc) && rc >= 1e-12) || continue
        ve = dot(u, Se \ u)
        db = sum(n[i] .* XI[i] for i in eachindex(gs)) ./ G
        si = sum(n[i] * AI[i] for i in eachindex(gs)) - G * dot(db, Om * db)
        W = zeros(L)
        for (i, g) in enumerate(gs)
            n[i] == 0 && continue
            a = n[i] .* (Om * (XI[i] .- db)) ./ si
            for t in 1:T
                e = t - g + 1
                1 <= e <= L && (W[e] += a[t])
            end
        end
        push!(rows, vcat(Float64.(n), ve / var_ref, (1 / si) / var_ref, kf / (sqrt(L) * norm(W .- u))))
    end
    D = reduce(vcat, permutedims.(rows))
    score = objective === :eti_cost ? D[:, T + 1] :
        [D[r, T + 2] <= max_it_ratio + 1e-12 ? -D[r, T + 3] : Inf for r in axes(D, 1)]
    ord = sortperm(collect(zip(score, axes(D, 1))))
    nbest = min(top, count(isfinite, score))
    best = D[ord[1:nbest], :]
    fr = Vector{Vector{Float64}}()
    for k in 0:max_wait
        idx = [r for r in axes(D, 1) if D[r, T] <= k && isfinite(score[r])]
        isempty(idx) && continue
        j = idx[sortperm(collect(zip(score[idx], idx)))[1]]
        push!(fr, vcat(Float64(k), D[j, :]))
    end
    frontier = reduce(vcat, permutedims.(fr))
    return (best=best, frontier=frontier, designs=D,
            columns=vcat(["start$g" for g in 2:T], ["wait", "eti_cost", "it_ratio", "radius"]),
            var_ref=var_ref)
end
