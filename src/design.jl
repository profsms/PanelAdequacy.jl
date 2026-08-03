# =============================================================================
# Shared infrastructure (spec §2.1): design_summary, union-find d_K,
# two-way demeaning. Computational core ported from the verified gate code
# (gate1_eiv_reliability.jl in the research program package).
# =============================================================================

"""
    DesignSummary

Design-level summary of a two-way fixed-effect panel (spec §2.1). Reused by
all four inference/diagnostic modules and useful standalone as a pre-outcome design
vetting tool.

Fields:
- `n`           : number of observations
- `N`           : number of units
- `T`           : number of periods
- `d_K`         : fixed-effect dimension, `N + T - ncomponents` (union-find
                  on the bipartite unit-time graph)
- `rho`         : `d_K / n`
- `ncomponents` : connected components of the unit-time graph
- `tau_star2`   : within (two-way-demeaned) sum of squares of the supplied
                  regressor, `X*'M X*`; `nothing` if no regressor supplied
- `lambda_n`    : largest self-normalized within-treatment share,
                  `max_i x_tilde_i^2 / V_n`; `nothing` without `x`
- `n_eff`       : inverse Herfindahl effective support size of the within
                  treatment; `nothing` without `x`
"""
struct DesignSummary
    n::Int
    N::Int
    T::Int
    d_K::Int
    rho::Float64
    ncomponents::Int
    tau_star2::Union{Nothing,Float64}
    lambda_n::Union{Nothing,Float64}
    n_eff::Union{Nothing,Float64}
end

# Compatibility constructor for summary-form callers that cannot recover the
# observation-level treatment shares from an aggregate V_n alone.
DesignSummary(n::Int, N::Int, T::Int, d_K::Int, rho::Float64,
              ncomponents::Int, tau_star2::Union{Nothing,Float64}) =
    DesignSummary(n, N, T, d_K, rho, ncomponents, tau_star2, nothing, nothing)

"Dense 1-based integer codes for a single id vector, in order of first appearance."
function _codes(ids::AbstractVector)
    isempty(ids) && throw(ArgumentError("empty id vector"))
    m = Dict{eltype(ids),Int}()
    out = Vector{Int}(undef, length(ids))
    for k in eachindex(ids)
        out[k] = get!(m, ids[k], length(m) + 1)
    end
    return out
end

"Map raw unit/time identifiers (any type) to dense integer codes 1:N, 1:T."
function _integer_codes(unit::AbstractVector, time::AbstractVector)
    length(unit) == length(time) ||
        throw(ArgumentError("unit and time must have equal length"))
    isempty(unit) && throw(ArgumentError("empty panel"))
    umap = Dict{eltype(unit),Int}()
    tmap = Dict{eltype(time),Int}()
    uid = Vector{Int}(undef, length(unit))
    tid = Vector{Int}(undef, length(time))
    for k in eachindex(unit)
        uid[k] = get!(umap, unit[k], length(umap) + 1)
        tid[k] = get!(tmap, time[k], length(tmap) + 1)
    end
    return uid, tid, length(umap), length(tmap)
end

"""
    fe_dimension(uid, tid, N, T) -> (d_K, ncomponents)

Fixed-effect dimension of the two-way design via union-find on the bipartite
unit-time graph: `d_K = N + T - ncomponents`.
"""
function fe_dimension(uid::Vector{Int}, tid::Vector{Int}, N::Int, T::Int)
    parent = collect(1:(N + T))
    rank = zeros(Int, N + T)
    function find(a)
        while parent[a] != a
            parent[a] = parent[parent[a]]   # path compression
            a = parent[a]
        end
        return a
    end
    function union!(a, b)
        ra, rb = find(a), find(b)
        ra == rb && return
        if rank[ra] < rank[rb]
            parent[ra] = rb
        elseif rank[ra] > rank[rb]
            parent[rb] = ra
        else
            parent[rb] = ra
            rank[ra] += 1
        end
    end
    for k in eachindex(uid)
        union!(uid[k], N + tid[k])
    end
    ncomp = length(Set(find(i) for i in 1:(N + T)))
    return N + T - ncomp, ncomp
end

"Exact fixed-effect dimension and connected components for a multiway design."
function _multiway_fe_dimension(fe_levels::AbstractVector)
    n = length(first(fe_levels))
    codes = [_codes(ids) for ids in fe_levels]
    levels = maximum.(codes)
    offsets = cumsum(vcat(0, levels[1:end-1]))
    ncols = sum(levels)

    # Sparse QR gives the actual rank, including dependencies that the usual
    # sum(levels) - (J-1)components shortcut misses in incomplete hypergraphs.
    rows = repeat(collect(1:n), outer=length(codes))
    cols = Vector{Int}(undef, n * length(codes))
    pos = 1
    for j in eachindex(codes), i in 1:n
        cols[pos] = offsets[j] + codes[j][i]
        pos += 1
    end
    D = sparse(rows, cols, ones(Float64, length(rows)), n, ncols)
    d_K = rank(qr(D))

    # Components are reported as a graph diagnostic only; unlike the two-way
    # formula they are not used to infer rank.
    parent = collect(1:ncols)
    function find(a)
        while parent[a] != a
            parent[a] = parent[parent[a]]
            a = parent[a]
        end
        return a
    end
    function union!(a, b)
        ra, rb = find(a), find(b)
        ra != rb && (parent[rb] = ra)
    end
    for i in 1:n
        anchor = offsets[1] + codes[1][i]
        for j in 2:length(codes)
            union!(anchor, offsets[j] + codes[j][i])
        end
    end
    ncomp = length(Set(find(i) for i in 1:ncols))
    return d_K, ncomp, levels
end

"""
    twoway_demean(x, unit, time; tol=1e-10, maxit=10_000) -> Vector{Float64}

Two-way within transformation `M x` by iterative alternating projections
(unbalanced-safe). `unit` and `time` are raw identifier vectors of any type.
Converges when the maximum absolute time-mean update falls below `tol`.
"""
function twoway_demean(x::AbstractVector{<:Real}, unit::AbstractVector,
                       time::AbstractVector; tol::Real=1e-10, maxit::Integer=10_000)
    uid, tid, N, T = _integer_codes(unit, time)
    return _twoway_demean(Float64.(x), uid, tid, N, T; tol=tol, maxit=maxit)
end

function _twoway_demean(x::Vector{Float64}, uid::Vector{Int}, tid::Vector{Int},
                        N::Int, T::Int; tol::Real=1e-10, maxit::Integer=10_000)
    w = copy(x)
    usum = zeros(N); ucnt = zeros(Int, N)
    tsum = zeros(T); tcnt = zeros(Int, T)
    for k in eachindex(uid)
        ucnt[uid[k]] += 1
        tcnt[tid[k]] += 1
    end
    converged = false
    for _ in 1:maxit
        fill!(usum, 0.0)
        for k in eachindex(w); usum[uid[k]] += w[k]; end
        umean = usum ./ max.(ucnt, 1)
        for k in eachindex(w); w[k] -= umean[uid[k]]; end
        fill!(tsum, 0.0)
        for k in eachindex(w); tsum[tid[k]] += w[k]; end
        tmean = tsum ./ max.(tcnt, 1)
        delta = maximum(abs, tmean)
        for k in eachindex(w); w[k] -= tmean[tid[k]]; end
        if delta < tol
            converged = true
            break
        end
    end
    converged || @warn "two-way demeaning did not converge within $maxit iterations"
    return w
end

"Normalize an optional numeric nuisance-covariate vector or matrix."
function _control_matrix(controls, n::Int)
    controls === nothing && return zeros(Float64, n, 0)
    Z = if controls isa AbstractVector{<:Real}
        reshape(Float64.(controls), n, 1)
    elseif controls isa AbstractMatrix{<:Real}
        Matrix{Float64}(controls)
    else
        throw(ArgumentError("controls must be nothing, a numeric vector, or a numeric matrix"))
    end
    size(Z, 1) == n || throw(ArgumentError(
        "controls has $(size(Z, 1)) rows; expected n = $n"))
    all(isfinite, Z) || throw(ArgumentError("controls must contain only finite values"))
    return Z
end

"Deterministic modified Gram--Schmidt basis, preserving supplied column order."
function _control_basis(Z::Matrix{Float64}; tol::Real=1e-10)
    n, k = size(Z)
    Q = zeros(Float64, n, k)
    r = 0
    for j in 1:k
        v = copy(@view Z[:, j])
        colscale = max(norm(v), 1.0)
        for _ in 1:2
            for h in 1:r
                q = @view Q[:, h]
                v .-= dot(q, v) .* q
            end
        end
        nv = norm(v)
        if nv > tol * colscale
            r += 1
            Q[:, r] .= v ./ nv
        end
    end
    return Q[:, 1:r], r
end

"Within-residualize a target against the fixed effects and optional controls."
function _partial_within_codes(x::AbstractVector{<:Real}, uid::Vector{Int},
                               tid::Vector{Int}, N::Int, T::Int;
                               controls=nothing, tol::Real=1e-10,
                               maxit::Integer=10_000)
    n = length(uid)
    length(x) == n || throw(ArgumentError("x must have length n = $n"))
    xf = _twoway_demean(Float64.(x), uid, tid, N, T;
                         tol=tol, maxit=maxit)
    Z = _control_matrix(controls, n)
    size(Z, 2) == 0 && return (xt=xf, Q=zeros(Float64, n, 0), rank=0)
    Zf = Matrix{Float64}(undef, n, size(Z, 2))
    for j in axes(Z, 2)
        Zf[:, j] = _twoway_demean(copy(@view Z[:, j]), uid, tid, N, T;
                                   tol=tol, maxit=maxit)
    end
    Q, r = _control_basis(Zf; tol=tol)
    xt = copy(xf)
    for j in 1:r
        q = @view Q[:, j]
        xt .-= dot(q, xt) .* q
    end
    return (xt=xt, Q=Q, rank=r)
end

"Apply the same fixed-effect/control nuisance projection to an outcome."
function _partial_outcome_codes(y::AbstractVector{<:Real}, uid::Vector{Int},
                                tid::Vector{Int}, N::Int, T::Int,
                                Q::AbstractMatrix{<:Real};
                                tol::Real=1e-10, maxit::Integer=10_000)
    yf = _twoway_demean(Float64.(y), uid, tid, N, T;
                         tol=tol, maxit=maxit)
    for j in axes(Q, 2)
        q = @view Q[:, j]
        yf .-= dot(q, yf) .* q
    end
    return yf
end

"""
    multiway_demean(x, fe_levels...; tol=1e-10, maxit=10_000)

Residualize `x` with respect to any number of categorical fixed-effect
dimensions by alternating projections. This is the scalable counterpart of
forming the full dummy matrix and is used by [`contrast_system`](@ref) for
applications with more than two fixed effects.
"""
function multiway_demean(x::AbstractVector{<:Real},
                         fe_levels::AbstractVector...;
                         tol::Real=1e-10, maxit::Integer=10_000)
    return _multiway_demean(Float64.(x), collect(fe_levels);
                            tol=tol, maxit=maxit)
end

function _multiway_demean(x::Vector{Float64}, fe_levels::AbstractVector;
                          tol::Real=1e-10, maxit::Integer=10_000)
    n = length(x)
    isempty(fe_levels) && throw(ArgumentError("at least one fixed-effect dimension is required"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    maxit > 0 || throw(ArgumentError("maxit must be positive"))

    codes = Vector{Vector{Int}}(undef, length(fe_levels))
    counts = Vector{Vector{Int}}(undef, length(fe_levels))
    sums = Vector{Vector{Float64}}(undef, length(fe_levels))
    for j in eachindex(fe_levels)
        ids = fe_levels[j]
        length(ids) == n || throw(ArgumentError(
            "fixed-effect dimension $j has length $(length(ids)); expected $n"))
        codes[j] = _codes(ids)
        ng = maximum(codes[j])
        counts[j] = zeros(Int, ng)
        for g in codes[j]
            counts[j][g] += 1
        end
        sums[j] = zeros(Float64, ng)
    end

    w = copy(x)
    converged = false
    for _ in 1:maxit
        max_update = 0.0
        for j in eachindex(codes)
            c = codes[j]
            s = sums[j]
            fill!(s, 0.0)
            @inbounds for k in eachindex(w)
                s[c[k]] += w[k]
            end
            @inbounds for g in eachindex(s)
                s[g] /= counts[j][g]
                max_update = max(max_update, abs(s[g]))
            end
            @inbounds for k in eachindex(w)
                w[k] -= s[c[k]]
            end
        end
        if max_update < tol
            converged = true
            break
        end
    end
    converged || @warn "multiway demeaning did not converge within $maxit iterations"
    return w
end

"Concentration diagnostics from an already residualized treatment."
function _treatment_concentration(xt::AbstractVector{<:Real})
    V_n = sum(abs2, xt)
    V_n > 0 || return (V_n=Float64(V_n), lambda_n=nothing, n_eff=nothing)
    shares = abs2.(xt) ./ V_n
    return (V_n=Float64(V_n), lambda_n=Float64(maximum(shares)),
            n_eff=Float64(inv(sum(abs2, shares))))
end

function _design_summary_codes(uid::Vector{Int}, tid::Vector{Int}, N::Int, T::Int;
                               x::Union{Nothing,AbstractVector{<:Real}}=nothing,
                               xt::Union{Nothing,AbstractVector{<:Real}}=nothing)
    x === nothing || xt === nothing || throw(ArgumentError(
        "supply raw x or residualized xt, not both"))
    n = length(uid)
    d_K, ncomp = fe_dimension(uid, tid, N, T)
    within = if xt !== nothing
        length(xt) == n || throw(ArgumentError("xt must have length n = $n"))
        Float64.(xt)
    elseif x !== nothing
        length(x) == n || throw(ArgumentError("x must have length n = $n"))
        _twoway_demean(Float64.(x), uid, tid, N, T)
    else
        nothing
    end
    if within === nothing
        return DesignSummary(n, N, T, d_K, d_K / n, ncomp,
                             nothing, nothing, nothing)
    end
    c = _treatment_concentration(within)
    return DesignSummary(n, N, T, d_K, d_K / n, ncomp,
                         c.V_n, c.lambda_n, c.n_eff)
end

"""
    design_summary(unit, time; x=nothing, controls=nothing) -> DesignSummary

Design-summary primitive (spec §2.1). `unit` and `time` are raw identifier
vectors; `x`, if supplied, is a regressor whose residual variation
`tau_star2 = X*'M X*` is computed after the fixed effects and optional numeric
`controls` are removed. `d_K` and `rho` continue to describe the fixed-effect
space alone.
"""
function design_summary(unit::AbstractVector, time::AbstractVector;
                        x::Union{Nothing,AbstractVector{<:Real}}=nothing,
                        controls=nothing)
    uid, tid, N, T = _integer_codes(unit, time)
    if x === nothing
        controls === nothing || throw(ArgumentError(
            "controls require x in design_summary"))
        return _design_summary_codes(uid, tid, N, T)
    end
    partial = _partial_within_codes(x, uid, tid, N, T; controls=controls)
    return _design_summary_codes(uid, tid, N, T; xt=partial.xt)
end

"""
    design_summary(fe_levels; x=nothing) -> DesignSummary

Multiway design summary for a non-empty collection of categorical fixed-effect
vectors. The fixed-effect dimension is the numerical rank of the sparse dummy
matrix, not the generally incorrect connected-component shortcut. `N` and `T`
record the first two level counts (or zero when absent); `ncomponents` is the
connectivity diagnostic for the multipartite incidence graph.
"""
function design_summary(fe_levels::AbstractVector;
                        x::Union{Nothing,AbstractVector{<:Real}}=nothing)
    isempty(fe_levels) && throw(ArgumentError(
        "fe_levels must be a non-empty collection of id vectors"))
    all(ids -> ids isa AbstractVector, fe_levels) || throw(ArgumentError(
        "fe_levels must contain id vectors"))
    n = length(first(fe_levels))
    n > 0 || throw(ArgumentError("empty panel"))
    for (j, ids) in enumerate(fe_levels)
        length(ids) == n || throw(ArgumentError(
            "fixed-effect dimension $j has length $(length(ids)); expected $n"))
    end
    d_K, ncomp, levels = _multiway_fe_dimension(fe_levels)
    N = levels[1]
    T = length(levels) >= 2 ? levels[2] : 0
    within = if x === nothing
        nothing
    else
        length(x) == n || throw(ArgumentError("x must have length n = $n"))
        _multiway_demean(Float64.(x), fe_levels)
    end
    if within === nothing
        return DesignSummary(n, N, T, d_K, d_K / n, ncomp,
                             nothing, nothing, nothing)
    end
    c = _treatment_concentration(within)
    return DesignSummary(n, N, T, d_K, d_K / n, ncomp,
                         c.V_n, c.lambda_n, c.n_eff)
end

function Base.show(io::IO, ::MIME"text/plain", d::DesignSummary)
    println(io, "Panel Design Summary")
    @printf(io, "  n = %d obs | N = %d units | T = %d periods\n", d.n, d.N, d.T)
    @printf(io, "  d_K = %d (connected components: %d) | rho = d_K/n = %.4f",
            d.d_K, d.ncomponents, d.rho)
    if d.tau_star2 !== nothing
        @printf(io, "\n  within variation tau*^2 = %.6g", d.tau_star2)
        if d.lambda_n !== nothing
            @printf(io, " | lambda_n = %.6g | N_eff = %.3f",
                    d.lambda_n, d.n_eff)
        end
    end
    if d.ncomponents > 1
        print(io, "\n  NOTE: design is disconnected ($(d.ncomponents) components); ",
              "within comparisons exist only inside each component.")
    end
end

Base.show(io::IO, d::DesignSummary) = @printf(io,
    "DesignSummary(n=%d, N=%d, T=%d, d_K=%d, rho=%.4f)", d.n, d.N, d.T, d.d_K, d.rho)
