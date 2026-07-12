# =============================================================================
# Shared infrastructure (spec §2.1): design_summary, union-find d_K,
# two-way demeaning. Computational core ported from the verified gate code
# (gate1_eiv_reliability.jl in the research program package).
# =============================================================================

"""
    DesignSummary

Design-level summary of a two-way fixed-effect panel (spec §2.1). Reused by
all three diagnostic modules and useful standalone as a pre-outcome design
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
"""
struct DesignSummary
    n::Int
    N::Int
    T::Int
    d_K::Int
    rho::Float64
    ncomponents::Int
    tau_star2::Union{Nothing,Float64}
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

"""
    design_summary(unit, time; x=nothing) -> DesignSummary

Design-summary primitive (spec §2.1). `unit` and `time` are raw identifier
vectors; `x`, if supplied, is a regressor whose within residual variation
`tau_star2 = X*'M X*` is computed by two-way demeaning.
"""
function design_summary(unit::AbstractVector, time::AbstractVector;
                        x::Union{Nothing,AbstractVector{<:Real}}=nothing)
    uid, tid, N, T = _integer_codes(unit, time)
    n = length(uid)
    d_K, ncomp = fe_dimension(uid, tid, N, T)
    tau_star2 = if x === nothing
        nothing
    else
        length(x) == n || throw(ArgumentError("x must have length n = $n"))
        xt = _twoway_demean(Float64.(x), uid, tid, N, T)
        sum(abs2, xt)
    end
    return DesignSummary(n, N, T, d_K, d_K / n, ncomp, tau_star2)
end

function Base.show(io::IO, ::MIME"text/plain", d::DesignSummary)
    println(io, "Panel Design Summary")
    @printf(io, "  n = %d obs | N = %d units | T = %d periods\n", d.n, d.N, d.T)
    @printf(io, "  d_K = %d (connected components: %d) | rho = d_K/n = %.4f",
            d.d_K, d.ncomponents, d.rho)
    if d.tau_star2 !== nothing
        @printf(io, "\n  within variation tau*^2 = %.6g", d.tau_star2)
    end
    if d.ncomponents > 1
        print(io, "\n  NOTE: design is disconnected ($(d.ncomponents) components); ",
              "within comparisons exist only inside each component.")
    end
end

Base.show(io::IO, d::DesignSummary) = @printf(io,
    "DesignSummary(n=%d, N=%d, T=%d, d_K=%d, rho=%.4f)", d.n, d.N, d.T, d.d_K, d.rho)
