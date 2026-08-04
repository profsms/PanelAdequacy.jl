# =============================================================================
# Cycle-inference module — exact inference under concentrated identifying variation
# SOURCE: Paper A / JBES (cycle_inference.tex), "Exact Inference in Fixed-Effect
# Regressions with Concentrated Identifying Variation".
#
# WHY THIS MODULE EXISTS. The diffuse companion's Gaussian limits require the self-normalized
# leverage lambda_n = max_i Xt_i^2/V_n to vanish. When it does not, the t-statistic
# converges to a convolution of raw errors with a Gaussian component, and the
# limiting null law is NOT FIXED across symmetric error laws of equal variance
# (prop ex:rade). So no FIXED distribution-free critical value is uniformly
# valid over that class, and studentization does not repair it (cor:stud). Validity
# must instead come from adapting to the unknown error law. That is what this
# module does, exactly, in finite samples.
#
# THE CONSTRUCTION.
#   annihilating contrast (def:contrast)  a vector v with v'D = 0 for the whole
#     FE design matrix D, so v'y carries no nuisance parameter at all — exactly,
#     not asymptotically.
#   cycle space (prop:cyclespace)  in a two-way design the annihilating contrasts
#     are EXACTLY the cycle space of the observation multigraph (units and
#     periods are the two vertex classes; each observation is an edge).
#   exactness (thm:exact)  with supports disjoint and each support a union of
#     complete dependence blocks, the joint score law is invariant under
#     independent sign flips under blockwise-symmetric errors (ass:sym), so the
#     sign-flip randomization test is exact in finite samples: arbitrary
#     heteroskedasticity, no homogeneity across clusters, no restriction on the
#     fixed-effect dimension.
#   capture ratio (def:kappa)  kappa_C = sum_c (v_c'x)^2 / V_n, the share of the
#     identifying variation the contrast system retains. It equals Pitman
#     efficiency only under BOTH diffuseness conditions (P1) and (P2) of
#     thm:power; outside that regime it remains an observable capture statistic.
#   capture-granularity trade-off (§6)  capture reaches one at block granularity,
#     but the randomization group carries ONE SIGN PER SUPPORT, so the smallest
#     full-enumeration p-value is 2^(1-C) in C treatment-loaded supports because
#     global sign reversal duplicates the absolute statistic. Thus level alpha
#     needs 2^(C-1) >= 1/alpha.
#   exact confidence set  the contrast scores are affine in beta_0,
#     U_c(beta_0) = v_c'y - b_c beta_0, so the entire randomization orbit is
#     affine too: T*(s; beta_0) = |A_s - beta_0 B_s| with A_s, B_s computed ONCE
#     per sign pattern. Inverting the test over a grid therefore costs no more
#     than a single test.
#
# The greedy packing is a LOWER BOUND on achievable capture; the structured
# (firm-pair four-cycle) packing is what the paper reports. Weighted edge-disjoint
# cycle packing is NP-hard in general, so these are the structure-exploiting
# heuristics proved optimal within their scope.
# =============================================================================

function _binary_floor_info(x::AbstractVector{<:Real}, alpha::Real)
    all(isfinite, x) || throw(ArgumentError("x must contain only finite values"))
    is_binary = all(z -> iszero(z) || z == one(z), x)
    if !is_binary
        return (is_binary=false, n_treated=nothing, binary_floor_ok=true,
                floor=nothing,
                reason="not a binary treatment; the binary granularity floor does not apply.")
    end

    n1 = count(z -> z == one(z), x)
    n0 = length(x) - n1
    cap = min(n1, n0)
    floor = exp2(1 - cap)
    ok = floor <= alpha
    relation = ok ? "<=" : ">"
    ending = ok ?
        "The structural granularity floor permits a two-sided level-alpha test." :
        "Not repairable by repacking."
    reason = @sprintf(
        "binary treatment, n1 = %d treated and n0 = %d untreated observations; at most min(n1,n0) = %d contrasts can carry loading; floor 2^(1-%d) = %.6g %s alpha = %.6g. %s",
        n1, n0, cap, cap, floor, relation, alpha, ending)
    return (is_binary=true, n_treated=n1, binary_floor_ok=ok,
            floor=floor, reason=reason)
end

"""
    CycleSystem

A system of annihilating contrasts for a two-way design. Fields:

- `rows`      : for each contrast, the observation indices in its support
- `signs`     : the matching +/-1 pattern (unit-normalized weights are
                `signs / sqrt(length)`)
- `loadings`  : `b_c = v_c' x`, the contrast loadings on the treatment
- `V_n`       : `x' M_D x`, the total within variation
- `kappa`     : `sum_c b_c^2 / V_n`, the capture ratio (equal to Pitman
                efficiency only under Theorem thm:power's (P1)--(P2))
- `max_share` : `max_c b_c^2 / sum_c b_c^2`, the (P1) balance statistic
"""
struct CycleSystem
    rows::Vector{Vector{Int}}
    signs::Vector{Vector{Float64}}
    loadings::Vector{Float64}
    V_n::Float64
    kappa::Float64
    max_share::Float64
end

Base.length(cs::CycleSystem) = length(cs.rows)

function Base.show(io::IO, cs::CycleSystem)
    @printf(io, "CycleSystem(C=%d, kappa=%.4f, max_share=%.4f, capture-implied SE ratio=%.3fx)",
            length(cs), cs.kappa, cs.max_share, 1 / sqrt(cs.kappa))
end

# -----------------------------------------------------------------------------
# Two-way residualization (alternating projections; unbalanced-safe)
# -----------------------------------------------------------------------------
function _ap_demean(x::Vector{Float64}, a::Vector{Int}, b::Vector{Int},
                    na::Int, nb::Int; tol=1e-10, maxiter=5000)
    xt = x .- mean(x)
    sa = zeros(Float64, na); ca = zeros(Int, na)
    sb = zeros(Float64, nb); cb = zeros(Int, nb)
    for _ in 1:maxiter
        fill!(sa, 0.0); fill!(ca, 0)
        for k in eachindex(xt)
            sa[a[k]] += xt[k]; ca[a[k]] += 1
        end
        m1 = 0.0
        for g in 1:na
            if ca[g] > 0
                sa[g] /= ca[g]; m1 = max(m1, abs(sa[g]))
            end
        end
        for k in eachindex(xt)
            xt[k] -= sa[a[k]]
        end
        fill!(sb, 0.0); fill!(cb, 0)
        for k in eachindex(xt)
            sb[b[k]] += xt[k]; cb[b[k]] += 1
        end
        m2 = 0.0
        for g in 1:nb
            if cb[g] > 0
                sb[g] /= cb[g]; m2 = max(m2, abs(sb[g]))
            end
        end
        for k in eachindex(xt)
            xt[k] -= sb[b[k]]
        end
        max(m1, m2) < tol && break
    end
    return xt
end

# A tiny sorted set of edge ids. Deterministic ascending iteration is what makes
# the greedy packing reproducible across languages; a hash Set is not, and
# greedy cycle packing is order-dependent.
struct SortedEdgeSet
    v::Vector{Int}
    SortedEdgeSet() = new(Int[])
end
Base.length(s::SortedEdgeSet) = length(s.v)
Base.iterate(s::SortedEdgeSet, st=1) = st > length(s.v) ? nothing : (s.v[st], st + 1)
Base.first(s::SortedEdgeSet) = s.v[1]
Base.collect(s::SortedEdgeSet) = copy(s.v)
function Base.push!(s::SortedEdgeSet, e::Int)
    i = searchsortedfirst(s.v, e)
    (i > length(s.v) || s.v[i] != e) && insert!(s.v, i, e)
    return s
end
function Base.delete!(s::SortedEdgeSet, e::Int)
    i = searchsortedfirst(s.v, e)
    (i <= length(s.v) && s.v[i] == e) && deleteat!(s.v, i)
    return s
end

# -----------------------------------------------------------------------------
# Greedy edge-disjoint cycle packing: digons first, then DFS-extracted cycles
# -----------------------------------------------------------------------------
function _greedy_packing(a::Vector{Int}, b::Vector{Int}, xt::Vector{Float64})
    m = length(a)
    rows = Vector{Vector{Int}}()
    signs = Vector{Vector{Float64}}()

    # Step 1: digons from parallel edges (repeat observations of the same cell)
    pair = Dict{Tuple{Int,Int},Vector{Int}}()
    for e in 1:m
        push!(get!(pair, (a[e], b[e]), Int[]), e)
    end
    used = falses(m)
    for es in values(pair)
        sort!(es)
        for k in 1:2:(length(es) - 1)
            e1, e2 = es[k], es[k + 1]
            push!(rows, [e1, e2]); push!(signs, [1.0, -1.0])
            used[e1] = used[e2] = true
        end
    end

    # Step 2: greedy DFS cycle extraction on the remaining multigraph.
    # Vertices: unit u -> u, period t -> na + t. Adjacency is iterated in
    # ASCENDING EDGE ID and vertices in FIRST-APPEARANCE order, matching the
    # reference implementation's traversal — greedy packing is order-dependent,
    # so this is what pins the published capture numbers.
    na = maximum(a); nb = maximum(b)
    adj = [SortedEdgeSet() for _ in 1:(na + nb)]
    ends = Vector{Tuple{Int,Int}}(undef, m)
    order = Int[]
    seen = falses(na + nb)
    for e in 1:m
        used[e] && continue
        va, vb = a[e], na + b[e]
        push!(adj[va], e); push!(adj[vb], e)
        ends[e] = (va, vb)
        for v in (va, vb)
            if !seen[v]
                seen[v] = true
                push!(order, v)
            end
        end
    end
    other(e, v) = ends[e][1] == v ? ends[e][2] : ends[e][1]

    function peel!(seed)
        st = collect(seed)
        while !isempty(st)
            v = pop!(st)
            if length(adj[v]) == 1
                e = first(adj[v]); u = other(e, v)
                delete!(adj[v], e); delete!(adj[u], e)
                length(adj[u]) <= 1 && push!(st, u)
            end
        end
    end
    peel!(order)

    ptr = 1
    while ptr <= length(order)
        if length(adj[order[ptr]]) < 2
            ptr += 1
            continue
        end
        start = order[ptr]
        parent_edge = Dict{Int,Int}(start => 0)
        stack = [start]
        cycle = Int[]
        while !isempty(stack) && isempty(cycle)
            v = stack[end]
            advanced = false
            for e in collect(adj[v])
                parent_edge[v] != 0 && e == parent_edge[v] && continue
                u = other(e, v)
                if haskey(parent_edge, u)
                    path = [e]; w = v
                    while w != u
                        pe = parent_edge[w]
                        push!(path, pe)
                        w = other(pe, w)
                    end
                    cycle = path
                    break
                end
                parent_edge[u] = e
                push!(stack, u)
                advanced = true
                break
            end
            !advanced && isempty(cycle) && pop!(stack)
        end
        if isempty(cycle)
            ptr += 1
            continue
        end
        L = length(cycle)
        push!(rows, copy(cycle))
        push!(signs, [isodd(k) ? 1.0 : -1.0 for k in 1:L])
        touched = Int[]
        for e in cycle
            va, vb = ends[e]
            delete!(adj[va], e); delete!(adj[vb], e)
            push!(touched, va); push!(touched, vb)
        end
        peel!(touched)
    end
    return rows, signs
end

function _packing_capture(rows, signs, x::Vector{Float64})
    return sum((sum(signs[c][k] * x[e]
                    for (k, e) in enumerate(rows[c])) / sqrt(length(rows[c])))^2
               for c in eachindex(rows))
end

function _greedy_multistart(a::Vector{Int}, b::Vector{Int}, x::Vector{Float64})
    m = length(a)
    idx = collect(1:m)
    prime = 2_147_483_647
    orders = Vector{Vector{Int}}()
    push!(orders, idx)
    push!(orders, sortperm(idx; by=i -> (a[i], x[i], b[i]), alg=MergeSort))
    for multiplier in (130_363, 13_262_647)
        push!(orders, sortperm(idx;
            by=i -> mod(i * multiplier, prime), alg=MergeSort))
    end

    best_rows = Vector{Vector{Int}}()
    best_signs = Vector{Vector{Float64}}()
    best_capture = -Inf
    for order in orders
        rows, signs = _greedy_packing(a[order], b[order], x[order])
        mapped = [[order[e] for e in row] for row in rows]
        capture = _packing_capture(mapped, signs, x)
        if capture > best_capture + 1e-14 * max(capture, best_capture, 1.0)
            best_rows, best_signs, best_capture = mapped, signs, capture
        end
    end
    return best_rows, best_signs
end

# -----------------------------------------------------------------------------
# Structured packing: unit-pair four-cycles, greedy by squared loading
# -----------------------------------------------------------------------------
"""
Structured packing: enumerate every (unit-pair x period-pair) four-cycle,
take them greedily by descending squared loading subject to edge-disjointness,
then run the greedy DFS packing on whatever edges are left over.

Unit and period pairs are enumerated in order of their ORIGINAL LABELS, and the
candidate sort is stable. Both matter: with a discrete treatment many four-cycles
carry identical squared loadings, so the greedy path — and the resulting capture
— is decided by tie-breaking. This ordering is the one that pins the published
numbers.
"""
function _structured_packing(a::Vector{Int}, b::Vector{Int}, xr::Vector{Float64},
                             units::Vector{Int}, periods::Vector{Int};
                             reverse_ties::Bool=false)
    m = length(a)
    cell = Dict{Tuple{Int,Int},Int}()
    for e in 1:m
        cell[(a[e], b[e])] = e
    end
    cands = Tuple{Float64,NTuple{4,Int}}[]
    for i in eachindex(units), j in (i + 1):length(units)
        f1, f2 = units[i], units[j]
        for k in eachindex(periods), l in (k + 1):length(periods)
            c1, c2 = periods[k], periods[l]
            q1 = get(cell, (f1, c1), 0); q1 == 0 && continue
            q2 = get(cell, (f2, c1), 0); q2 == 0 && continue
            q3 = get(cell, (f2, c2), 0); q3 == 0 && continue
            q4 = get(cell, (f1, c2), 0); q4 == 0 && continue
            val = (xr[q1] - xr[q2] + xr[q3] - xr[q4]) / 2
            push!(cands, (val^2, (q1, q2, q3, q4)))
        end
    end
    sort!(cands; by=first, rev=true, alg=MergeSort)   # stable
    if reverse_ties
        first_tie = 1
        while first_tie <= length(cands)
            last_tie = first_tie
            while last_tie < length(cands) &&
                  cands[last_tie + 1][1] == cands[first_tie][1]
                last_tie += 1
            end
            reverse!(cands, first_tie, last_tie)
            first_tie = last_tie + 1
        end
    end

    sg = [1.0, -1.0, 1.0, -1.0]
    used = falses(m)
    rows = Vector{Vector{Int}}(); signs = Vector{Vector{Float64}}()
    for (_, q) in cands
        (used[q[1]] || used[q[2]] || used[q[3]] || used[q[4]]) && continue
        push!(rows, collect(q)); push!(signs, copy(sg))
        for e in q
            used[e] = true
        end
    end

    # DFS pass on the leftover edges, exactly as the reference does
    left = findall(!, used)
    if !isempty(left)
        sub_rows, sub_signs = _greedy_packing(a[left], b[left], xr[left])
        for (r, s) in zip(sub_rows, sub_signs)
            push!(rows, [left[e] for e in r]); push!(signs, s)
        end
    end
    return rows, signs
end

# -----------------------------------------------------------------------------
# Sparse (mobility-network) structured packing
#
# The dense routine enumerates every unit-pair x period-pair four-cycle, which is
# O(N^2 T^2) and hopeless on a matched employer-employee network (35,807 workers
# against 5,301 firms is ~10^9 candidate pairs). Mobility networks have a
# different structure to exploit: most workers contribute exactly two edges, so
#   - a worker with two edges at the SAME firm is a digon, capture d^2/2;
#   - a worker with two edges at DIFFERENT firms is a "mover" carrying the
#     within-match difference d = xtilde[e1] - xtilde[e2] for that firm pair;
#     two movers over the same firm pair close a four-cycle with capture
#     (d_j - d_i)^2/4.
# Within a firm pair the optimal edge-disjoint pairing is the EXTREME (nested)
# one -- pair the largest d with the smallest, and recurse inward (Lemma
# lem:nested). Everything left over goes to the DFS packing.
#
# O(m log m), so it runs on the full network in seconds.
# -----------------------------------------------------------------------------
function _sparse_packing(a::Vector{Int}, b::Vector{Int}, xt::Vector{Float64})
    m = length(a)
    wedges = Dict{Int,Vector{Int}}()
    for e in 1:m
        push!(get!(wedges, a[e], Int[]), e)
    end

    rows = Vector{Vector{Int}}(); signs = Vector{Vector{Float64}}()
    # firm pair -> movers, each (difference, edge1, edge2)
    pairs = Dict{Tuple{Int,Int},Vector{Tuple{Float64,Int,Int}}}()
    others = Int[]

    for w in sort!(collect(keys(wedges)))
        es = wedges[w]
        if length(es) == 2
            e1, e2 = es[1], es[2]
            f1, f2 = b[e1], b[e2]
            if f1 == f2                                   # stayer: a digon
                push!(rows, [e1, e2]); push!(signs, [1.0, -1.0])
            else
                if f1 > f2
                    f1, f2 = f2, f1
                    e1, e2 = e2, e1
                end
                push!(get!(pairs, (f1, f2), Tuple{Float64,Int,Int}[]),
                      (xt[e1] - xt[e2], e1, e2))
            end
        else
            append!(others, es)
        end
    end

    leftover = Int[]
    for k in sort!(collect(keys(pairs)))
        lst = pairs[k]
        sort!(lst; by=first, alg=MergeSort)
        i, j = 1, length(lst)
        while i < j
            # four-cycle from the extreme pair: +e1(j) -e2(j) +e2(i) -e1(i)
            push!(rows, [lst[j][2], lst[j][3], lst[i][3], lst[i][2]])
            push!(signs, [1.0, -1.0, 1.0, -1.0])
            i += 1; j -= 1
        end
        if i == j                                          # unpaired mover
            push!(leftover, lst[i][2], lst[i][3])
        end
    end

    left = sort!(unique(vcat(leftover, others)))
    if !isempty(left)
        sub_rows, sub_signs = _greedy_packing(a[left], b[left], xt[left])
        for (r, s) in zip(sub_rows, sub_signs)
            push!(rows, [left[e] for e in r]); push!(signs, s)
        end
    end
    return rows, signs
end

# Combine disjoint FE-annihilating cycles in pairs so the resulting support
# contrast also annihilates one continuous nuisance covariate. For two base
# contrasts q_i and q_j, (g_j q_i - g_i q_j)/sqrt(g_i^2+g_j^2), where
# g_c=q_c'z, is the unique normalized combination orthogonal to z.
function _controlled_pairing(rows::Vector{Vector{Int}},
                             signs::Vector{Vector{Float64}},
                             x::Vector{Float64}, z::Vector{Float64})
    C = length(rows)
    load = zeros(Float64, C)
    nuisance = zeros(Float64, C)
    for c in 1:C
        q = signs[c] ./ sqrt(length(rows[c]))
        load[c] = dot(q, x[rows[c]])
        nuisance[c] = dot(q, z[rows[c]])
    end
    tol = 64 * eps(Float64) * max(norm(z), 1.0)
    zero = findall(g -> abs(g) <= tol, nuisance)
    nonzero = findall(g -> abs(g) > tol, nuisance)
    out_rows = Vector{Vector{Int}}()
    out_signs = Vector{Vector{Float64}}()
    used = falses(C)
    for c in zero
        push!(out_rows, rows[c]); push!(out_signs, signs[c]); used[c] = true
    end
    candidates = Tuple{Float64,Int,Int}[]
    if length(nonzero) >= 2
        for ii in 1:(length(nonzero) - 1), jj in (ii + 1):length(nonzero)
            i, j = nonzero[ii], nonzero[jj]
            den = nuisance[i]^2 + nuisance[j]^2
            num = nuisance[j] * load[i] - nuisance[i] * load[j]
            push!(candidates, (num^2 / den, i, j))
        end
    end
    sort!(candidates; by=c -> (-c[1], c[2], c[3]))
    for (_, i, j) in candidates
        (used[i] || used[j]) && continue
        den = hypot(nuisance[i], nuisance[j])
        a1, a2 = nuisance[j] / den, -nuisance[i] / den
        q1 = signs[i] ./ sqrt(length(rows[i]))
        q2 = signs[j] ./ sqrt(length(rows[j]))
        r = vcat(rows[i], rows[j])
        q = vcat(a1 .* q1, a2 .* q2)
        push!(out_rows, r)
        push!(out_signs, q .* sqrt(length(r)))
        used[i] = used[j] = true
    end
    return out_rows, out_signs
end

# Dense one-control construction. A 2-by-3 complete rectangle has a
# two-dimensional FE cycle space; projecting the target cycle coordinates off
# the control coordinate leaves a one-dimensional contrast that annihilates
# unit effects, period effects, and the control exactly.
function _controlled_rectangles(a::Vector{Int}, b::Vector{Int},
                                x::Vector{Float64}, z::Vector{Float64})
    cell = Dict{Tuple{Int,Int},Int}()
    for e in eachindex(a)
        key = (a[e], b[e])
        haskey(cell, key) && return Vector{Vector{Int}}(), Vector{Vector{Float64}}()
        cell[key] = e
    end
    N, T = maximum(a), maximum(b)
    T >= 3 || return Vector{Vector{Int}}(), Vector{Vector{Float64}}()
    q1 = [0.5, -0.5, 0.0, -0.5, 0.5, 0.0]
    q2 = [1.0, 1.0, -2.0, -1.0, -1.0, 2.0] ./ sqrt(12.0)
    candidates = Tuple{Float64,Vector{Int},Vector{Float64},Int}[]
    serial = 0
    for u1 in 1:(N - 1), u2 in (u1 + 1):N
        for t1 in 1:(T - 2), t2 in (t1 + 1):(T - 1), t3 in (t2 + 1):T
            keys = ((u1,t1), (u1,t2), (u1,t3),
                    (u2,t1), (u2,t2), (u2,t3))
            all(k -> haskey(cell, k), keys) || continue
            r = [cell[k] for k in keys]
            l1, l2 = dot(q1, x[r]), dot(q2, x[r])
            g1, g2 = dot(q1, z[r]), dot(q2, z[r])
            gss = g1^2 + g2^2
            if gss > 64 * eps(Float64) * max(sum(abs2, z[r]), 1.0)
                proj = (g1 * l1 + g2 * l2) / gss
                a1, a2 = l1 - proj * g1, l2 - proj * g2
            else
                a1, a2 = l1, l2
            end
            anorm = hypot(a1, a2)
            anorm > 64 * eps(Float64) * max(norm(x[r]), 1.0) || continue
            q = (a1 .* q1 .+ a2 .* q2) ./ anorm
            serial += 1
            push!(candidates, (anorm^2, r, q .* sqrt(6.0), serial))
        end
    end
    sort!(candidates; by=c -> (-c[1], c[4]))
    used = falses(length(a))
    rows = Vector{Vector{Int}}()
    signs = Vector{Vector{Float64}}()
    for (_, r, s, _) in candidates
        any(used[r]) && continue
        push!(rows, r); push!(signs, s); used[r] .= true
    end
    return rows, signs
end

"""
    cycle_contrasts(x, unit, time; method=:structured, controls=nothing) -> CycleSystem

Build a system of nuisance-annihilating contrasts for the two-way design
`(unit, time)` with treatment `x`. With one linearly independent continuous
control, `:structured` uses complete 2-by-3 rectangles and local cycle-space
projection; the other methods pair base cycles. Every returned contrast then
annihilates the fixed effects and the control. More than one continuous control
is rejected rather than silently producing an invalid exact test.

- `:structured` (default) — unit-pair four-cycles taken greedily by squared
  loading. This is what the paper reports; on the public matched
  employer-employee network it attains `kappa ~ 0.51` for a match-level
  treatment against `0.26` for naive packing.
- `:sparse` — for MOBILITY NETWORKS (matched employer-employee data), where
  enumerating all unit-pair four-cycles is hopeless. Uses stayer digons, then
  firm-pair four-cycles built from movers with the optimal extreme (nested)
  pairing, then a DFS pass. `O(m log m)`; this is the method for AKM-style
  designs and the one the article's headline match-level figure uses.
- `:greedy` — digons first, then DFS-extracted cycles, retaining the best of
  four fixed traversal orders. A LOWER BOUND on achievable capture; cheap,
  deterministic, and always available.

Each contrast annihilates both sets of fixed effects identically, so `v_c'y`
is free of the nuisance parameters exactly, in finite samples.
"""
function cycle_contrasts(x::AbstractVector{<:Real}, unit::AbstractVector,
                         time::AbstractVector; method::Symbol=:structured,
                         controls=nothing)
    method in (:structured, :sparse, :greedy) ||
        throw(ArgumentError("method must be :structured, :sparse or :greedy"))
    n = length(unit)
    length(time) == n || throw(ArgumentError(
        "x, unit, time must have equal length"))
    length(x) == n || throw(ArgumentError("x, unit, time must have equal length"))
    (any(ismissing, unit) || any(ismissing, time)) && throw(ArgumentError(
        "unit and time identifiers cannot be missing"))
    xr0 = Float64.(x)
    all(isfinite, xr0) || throw(ArgumentError("x must contain only finite values"))

    # Canonical row order makes every packing invariant to input row order.
    # The returned supports are mapped back to the caller's observation indices.
    order = sortperm(collect(1:n);
                     by=i -> (unit[i], time[i], xr0[i]), alg=MergeSort)
    uc = unit[order]
    tc = time[order]
    xr = xr0[order]
    Zr = _control_matrix(controls, n)[order, :]
    uid, tid, N, T = _integer_codes(uc, tc)
    # V_n needs the residualized treatment; the CONTRASTS do not. Every
    # annihilating contrast satisfies M_D v_c = v_c, hence v_c'xtilde = v_c'x
    # EXACTLY. Ranking and loadings therefore use the raw treatment: the
    # alternating projections contribute no round-off to them, so the greedy
    # path is decided by the design rather than by the last ulp of an iterative
    # solve — which is what makes the packing reproducible across languages.
    partial = _partial_within_codes(xr, uid, tid, N, T; controls=Zr)
    partial.rank <= 1 || throw(ArgumentError(
        "cycle packing currently supports at most one linearly independent continuous control"))
    xt = partial.xt
    V_n = sum(abs2, xt)
    V_n > 1e-12 * max(sum(abs2, xr), 1.0) ||
        throw(ArgumentError("regressor has no within variation"))

    if method === :structured
        # enumerate unit/period pairs in ORIGINAL-LABEL order (see
        # _structured_packing: tie-breaking decides the greedy path)
        ulab = Vector{eltype(unit)}(undef, N)
        tlab = Vector{eltype(time)}(undef, T)
        for k in eachindex(uid)
            ulab[uid[k]] = uc[k]
            tlab[tid[k]] = tc[k]
        end
        rows, signs = _structured_packing(uid, tid, xr,
                                          sortperm(ulab), sortperm(tlab))
        rows2, signs2 = _structured_packing(uid, tid, xr,
                                            sortperm(ulab), sortperm(tlab);
                                            reverse_ties=true)
        if _packing_capture(rows2, signs2, xr) >
           _packing_capture(rows, signs, xr) + 1e-14 * max(V_n, 1.0)
            rows, signs = rows2, signs2
        end
    elseif method === :sparse
        rows, signs = _sparse_packing(uid, tid, xr)
    else
        rows, signs = _greedy_multistart(uid, tid, xr)
    end
    if method !== :greedy
        greedy_rows, greedy_signs = _greedy_multistart(uid, tid, xr)
        if _packing_capture(greedy_rows, greedy_signs, xr) >
           _packing_capture(rows, signs, xr) + 1e-14 * max(V_n, 1.0)
            rows, signs = greedy_rows, greedy_signs
        end
    end
    if partial.rank == 1
        z = vec(partial.Q[:, 1])
        paired_rows, paired_signs = _controlled_pairing(rows, signs, xr, z)
        if method === :structured
            rect_rows, rect_signs = _controlled_rectangles(uid, tid, xr, z)
            if _packing_capture(rect_rows, rect_signs, xr) >
               _packing_capture(paired_rows, paired_signs, xr) +
               1e-14 * max(V_n, 1.0)
                rows, signs = rect_rows, rect_signs
            else
                rows, signs = paired_rows, paired_signs
            end
        else
            rows, signs = paired_rows, paired_signs
        end
    end
    C = length(rows)
    loadings = Vector{Float64}(undef, C)
    for c in 1:C
        s = 0.0
        for (k, e) in enumerate(rows[c])
            s += signs[c][k] * xr[e]
        end
        loadings[c] = s / sqrt(length(rows[c]))
    end
    ssq = sum(abs2, loadings)
    mapped_rows = [[order[e] for e in r] for r in rows]
    return CycleSystem(mapped_rows, signs, loadings, V_n, ssq / V_n,
                       C == 0 ? 0.0 : maximum(abs2, loadings) / max(ssq, eps()))
end

"""
    contrast_system(x, fe_levels, rows, weights; blocks=nothing,
                    tol=1e-10, maxit=10_000) -> CycleSystem

Build and validate a design-only annihilating contrast system supplied by the
caller. `fe_levels` is a collection of categorical fixed-effect id vectors;
`rows[c]` contains the observation indices in support `c`; and `weights[c]`
contains the corresponding contrast coefficients. The routine normalizes each
contrast, verifies pairwise-disjoint supports and `q_c'D = 0` for every fixed
effect, and computes `V_n`, the treatment loadings, capture, and (P1) balance.

This constructor covers Paper A applications with more than two fixed effects,
including the worker--firm--year paired-stayer design. Construction must still
use `(x,D)` only. If `blocks` is supplied, every treatment-loaded support must
also be a union of complete blocks; incompatible systems are rejected.
"""
function contrast_system(x::AbstractVector{<:Real}, fe_levels::AbstractVector,
                         rows::AbstractVector, weights::AbstractVector;
                         blocks::Union{Nothing,AbstractVector}=nothing,
                         tol::Real=1e-10, maxit::Integer=10_000)
    n = length(x)
    length(rows) == length(weights) || throw(ArgumentError(
        "rows and weights must contain the same number of supports"))
    isempty(rows) && throw(ArgumentError("at least one support is required"))
    isempty(fe_levels) && throw(ArgumentError(
        "at least one fixed-effect dimension is required"))
    for (j, ids) in enumerate(fe_levels)
        length(ids) == n || throw(ArgumentError(
            "fixed-effect dimension $j has length $(length(ids)); expected $n"))
    end

    out_rows = Vector{Vector{Int}}(undef, length(rows))
    out_signs = Vector{Vector{Float64}}(undef, length(rows))
    used = falses(n)
    fe_codes = [_codes(ids) for ids in fe_levels]
    for c in eachindex(rows)
        r = Int.(collect(rows[c]))
        w = Float64.(collect(weights[c]))
        isempty(r) && throw(ArgumentError("support $c is empty"))
        length(r) == length(w) || throw(ArgumentError(
            "support $c has $(length(r)) rows but $(length(w)) weights"))
        length(unique(r)) == length(r) || throw(ArgumentError(
            "support $c repeats an observation"))
        all(e -> 1 <= e <= n, r) || throw(ArgumentError(
            "support $c contains an observation index outside 1:$n"))
        any(used[r]) && throw(ArgumentError("contrast supports are not disjoint"))
        used[r] .= true

        nw = sqrt(sum(abs2, w))
        nw > 0 || throw(ArgumentError("support $c has zero-norm weights"))
        q = w ./ nw
        for (j, code) in enumerate(fe_codes)
            balance = Dict{Int,Float64}()
            for (k, e) in enumerate(r)
                balance[code[e]] = get(balance, code[e], 0.0) + q[k]
            end
            maximum(abs, values(balance)) <= tol || throw(ArgumentError(
                "support $c does not annihilate fixed-effect dimension $j"))
        end
        out_rows[c] = r
        # CycleSystem stores coefficients divided by sqrt(length) at use time.
        out_signs[c] = q .* sqrt(length(r))
    end

    xr = Float64.(x)
    xt = _multiway_demean(copy(xr), fe_levels; tol=tol, maxit=maxit)
    V_n = sum(abs2, xt)
    V_n > 1e-12 * max(sum(abs2, xr), 1.0) || throw(ArgumentError(
        "regressor has no variation after removing the supplied fixed effects"))
    loadings = [sum(out_signs[c][k] * xr[e]
                    for (k, e) in enumerate(out_rows[c])) / sqrt(length(out_rows[c]))
                for c in eachindex(out_rows)]
    ssq = sum(abs2, loadings)
    cs = CycleSystem(out_rows, out_signs, loadings, V_n, ssq / V_n,
                     isempty(loadings) ? 0.0 : maximum(abs2, loadings) / max(ssq, eps()))
    if blocks !== nothing
        comp = support_compatibility(cs, blocks)
        comp.compatible || throw(ArgumentError(
            "contrast supports are not unions of complete dependence blocks; " *
            "$(comp.incompatible_blocks) block(s) are partial or split"))
    end
    return cs
end

"Treatment-loaded supports; zero-loading contrasts do not enlarge the statistic's orbit."
function _active_supports(cs::CycleSystem)
    scale = max(sqrt(cs.V_n), 1.0)
    tol = 64 * eps(Float64) * scale
    return findall(b -> abs(b) > tol, cs.loadings)
end

"""
    support_compatibility(cs, blocks) -> NamedTuple

Check Assumption ass:sym(iii) of Paper A. `blocks` assigns every observation to
a dependence block. A treatment-loaded contrast support is compatible iff it
contains all or none of every block; blocks may not be split across supports.
Zero-loading supports are ignored because they do not enter the test statistic.
"""
function support_compatibility(cs::CycleSystem, blocks::AbstractVector)
    n = length(blocks)
    assignment = zeros(Int, n)
    active = _active_supports(cs)
    for c in active
        for e in cs.rows[c]
            1 <= e <= n || throw(ArgumentError(
                "blocks has length $n but a support contains observation $e"))
            assignment[e] == 0 || throw(ArgumentError(
                "contrast supports are not disjoint at observation $e"))
            assignment[e] = c
        end
    end
    bcode = _codes(blocks)
    first_assignment = fill(-1, maximum(bcode))
    incompatible = falses(maximum(bcode))
    for k in eachindex(bcode)
        g = bcode[k]
        if first_assignment[g] == -1
            first_assignment[g] = assignment[k]
        elseif first_assignment[g] != assignment[k]
            incompatible[g] = true
        end
    end
    bad = count(incompatible)
    return (compatible=bad == 0, incompatible_blocks=bad,
            nblocks=length(first_assignment), effective_C=length(active))
end

"""
    cycle_capture(x, unit, time; method=:structured) -> Float64

The capture ratio `kappa_C` alone (Definition def:kappa). It equals the Pitman
efficiency relative to the Gaussian oracle only when both (P1) and (P2) of
Theorem thm:power hold. It remains a design-only capture statistic elsewhere.
"""
cycle_capture(x, unit, time; method::Symbol=:structured, controls=nothing) =
    cycle_contrasts(x, unit, time; method=method, controls=controls).kappa

# -----------------------------------------------------------------------------
# Exact sign-flip test, and its inversion
# -----------------------------------------------------------------------------
"""
    signflip_test(y, x, unit, time; beta0=0, nflips=99999, rng, method,
                  blocks=nothing)
    signflip_test(y, cs; beta0=0, nflips=99999, rng, blocks=nothing)

Monte Carlo sign-flip randomization test of `H0: beta = beta0` (Paper A,
Theorem thm:exact). With `blocks=nothing`, exactness is stated for independent,
observation-level symmetric errors (singleton blocks). For clustered errors,
pass one block id per observation: the routine rejects a contrast system unless
every treatment-loaded support is a union of complete blocks. Conditional on
that compatibility check, the joint score law is invariant under independent
sign flips; the scores need not be exchangeable or identically distributed.

The identity transform is adjoined, giving the valid
`(1 + #{T* >= T0})/(1 + B)` Monte Carlo p-value. `full_enumeration_floor` is
`2^(1-effective_C)` for the default absolute statistic, because global sign
reversal duplicates every orbit value. `min_pvalue` is retained as an alias.
"""
function signflip_test(y::AbstractVector{<:Real}, x::AbstractVector{<:Real},
                       unit::AbstractVector, time::AbstractVector;
                       beta0::Real=0.0, nflips::Integer=99999,
                       rng::AbstractRNG=Random.default_rng(),
                       method::Symbol=:structured,
                       controls=nothing,
                       blocks::Union{Nothing,AbstractVector}=nothing)
    length(y) == length(x) || throw(ArgumentError(
        "y, x, unit and time must have equal length"))
    cs = cycle_contrasts(x, unit, time; method=method, controls=controls)
    return signflip_test(y, cs; beta0=beta0, nflips=nflips, rng=rng,
                         blocks=blocks)
end

function _contrast_terms(y::AbstractVector{<:Real}, cs::CycleSystem;
                         blocks::Union{Nothing,AbstractVector}=nothing)
    C = length(cs)
    C > 0 || throw(ArgumentError("no admissible contrasts"))
    nneeded = maximum(maximum(r) for r in cs.rows)
    length(y) >= nneeded || throw(ArgumentError(
        "y has length $(length(y)); contrast supports require at least $nneeded observations"))
    if blocks !== nothing
        length(blocks) == length(y) || throw(ArgumentError(
            "blocks must have the same length as y"))
        comp = support_compatibility(cs, blocks)
        comp.compatible || throw(ArgumentError(
            "contrast supports are not unions of complete dependence blocks; " *
            "$(comp.incompatible_blocks) block(s) are partial or split"))
    end
    active = _active_supports(cs)
    isempty(active) && throw(ArgumentError(
        "no treatment-loaded contrasts: every b_c = v_c'x is zero"))
    yv = Float64.(y)
    Vy = Vector{Float64}(undef, length(active))
    b = cs.loadings[active]
    for (j, c) in enumerate(active)
        s = 0.0
        for (k, e) in enumerate(cs.rows[c])
            s += cs.signs[c][k] * yv[e]
        end
        Vy[j] = s / sqrt(length(cs.rows[c]))
    end
    p1 = b .* Vy
    p2 = b .^ 2
    A0, B0 = sum(p1), sum(p2)
    B0 > 0 || throw(ArgumentError("captured treatment variation is zero"))
    return p1, p2, A0, B0, length(active)
end

function signflip_test(y::AbstractVector{<:Real}, cs::CycleSystem;
                       beta0::Real=0.0, nflips::Integer=99999,
                       rng::AbstractRNG=Random.default_rng(),
                       blocks::Union{Nothing,AbstractVector}=nothing)
    nflips > 0 || throw(ArgumentError("nflips must be positive"))
    p1, p2, A0, B0, Ceff = _contrast_terms(y, cs; blocks=blocks)
    T0 = abs(A0 - beta0 * B0)
    ge = 0
    for _ in 1:nflips
        A = 0.0; B = 0.0
        for c in eachindex(p1)
            s = rand(rng, Bool) ? 1.0 : -1.0
            A += s * p1[c]; B += s * p2[c]
        end
        abs(A - beta0 * B) >= T0 - 1e-12 && (ge += 1)
    end
    floor = exp2(1 - Ceff)
    return (p=(1 + ge) / (1 + nflips), C=length(cs), effective_C=Ceff,
            kappa=cs.kappa, beta_tilde=A0 / B0,
            full_enumeration_floor=floor, min_pvalue=floor, statistic=T0)
end

"""
    signflip_interval(y, x, unit, time; alpha=0.05, ngrid=24001,
                      nflips=99999, rng, method, blocks=nothing, span=12)
    signflip_interval(y, cs; ...)

Invert [`signflip_test`](@ref) over a grid using common random numbers. For the
unstudentized absolute statistic, every sign pattern's acceptance interval
contains `beta_tilde`, so the exact acceptance set is an interval. `contiguous`
is retained as a numerical audit and should be `true`. If acceptance reaches a
grid boundary, the corresponding reported endpoint is infinite and
`grid_truncated=true`; the routine never presents a search boundary as a finite
confidence limit.
"""
function signflip_interval(y::AbstractVector{<:Real}, x::AbstractVector{<:Real},
                           unit::AbstractVector, time::AbstractVector;
                           alpha::Real=0.05, ngrid::Integer=24001,
                           nflips::Integer=99999,
                           rng::AbstractRNG=Random.default_rng(),
                           method::Symbol=:structured,
                           controls=nothing,
                           blocks::Union{Nothing,AbstractVector}=nothing,
                           span::Real=12.0)
    length(y) == length(x) || throw(ArgumentError(
        "y, x, unit and time must have equal length"))
    cs = cycle_contrasts(x, unit, time; method=method, controls=controls)
    return signflip_interval(y, cs; alpha=alpha, ngrid=ngrid,
                             nflips=nflips, rng=rng, blocks=blocks, span=span)
end

function signflip_interval(y::AbstractVector{<:Real}, cs::CycleSystem;
                           alpha::Real=0.05, ngrid::Integer=24001,
                           nflips::Integer=99999,
                           rng::AbstractRNG=Random.default_rng(),
                           blocks::Union{Nothing,AbstractVector}=nothing,
                           span::Real=12.0)
    0 < alpha < 1 || throw(ArgumentError("alpha must lie in (0, 1)"))
    ngrid >= 3 || throw(ArgumentError("ngrid must be at least 3"))
    nflips > 0 || throw(ArgumentError("nflips must be positive"))
    span > 0 || throw(ArgumentError("span must be positive"))
    p1, p2, A0, B0, Ceff = _contrast_terms(y, cs; blocks=blocks)
    beta_tilde = A0 / B0

    As = Vector{Float64}(undef, nflips)
    Bs = Vector{Float64}(undef, nflips)
    for j in 1:nflips
        A = 0.0; B = 0.0
        for c in eachindex(p1)
            s = rand(rng, Bool) ? 1.0 : -1.0
            A += s * p1[c]; B += s * p2[c]
        end
        As[j] = A; Bs[j] = B
    end

    scale = sqrt(sum(abs2, p1)) / B0
    if scale <= eps(Float64) * max(abs(beta_tilde), 1.0)
        floor = exp2(1 - Ceff)
        if floor > alpha
            grid = [beta_tilde - 1.0, beta_tilde, beta_tilde + 1.0]
            return (lo=-Inf, hi=Inf, contiguous=true, grid_truncated=true,
                    beta_tilde=beta_tilde, C=length(cs), effective_C=Ceff,
                    kappa=cs.kappa, grid=grid, pvalue=ones(3))
        end
        return (lo=beta_tilde, hi=beta_tilde, contiguous=true,
                grid_truncated=false, beta_tilde=beta_tilde, C=length(cs),
                effective_C=Ceff, kappa=cs.kappa, grid=[beta_tilde], pvalue=[1.0])
    end

    grid = range(beta_tilde - span * scale, beta_tilde + span * scale;
                 length=ngrid)
    # For |Bs| < B0, |As-beta*Bs| >= |A0-beta*B0| holds exactly between
    # the two equality roots. Sweep those intervals over the sorted grid in
    # O(B log G + G), instead of evaluating all B*G statistic pairs.
    diff = zeros(Int, ngrid + 1)
    scale_orbit = max(abs(A0), B0, 1.0)
    orbit_tol = 64 * eps(Float64) * scale_orbit
    @inbounds for j in 1:nflips
        if abs(abs(Bs[j]) - B0) <= orbit_tol
            orientation = Bs[j] >= 0 ? 1.0 : -1.0
            if abs(As[j] - orientation * A0) <= orbit_tol
                diff[1] += 1; diff[end] -= 1
                continue
            end
        end
        d1, d2 = B0 - Bs[j], B0 + Bs[j]
        if abs(d1) <= orbit_tol || abs(d2) <= orbit_tol
            # This can only arise from an all-equal sign vector in exact
            # arithmetic; retain a conservative direct fallback if rounding
            # produces a near-degenerate exceptional draw.
            for g in 1:ngrid
                b0 = grid[g]
                abs(As[j] - b0 * Bs[j]) >= abs(A0 - b0 * B0) - 1e-12 &&
                    (diff[g] += 1; diff[g + 1] -= 1)
            end
            continue
        end
        r1 = (A0 - As[j]) / d1
        r2 = (A0 + As[j]) / d2
        lo, hi = minmax(r1, r2)
        left = searchsortedfirst(grid, lo)
        right = searchsortedlast(grid, hi)
        if left <= right
            diff[left] += 1
            diff[right + 1] -= 1
        end
    end
    counts = cumsum(@view diff[1:ngrid])
    pv = (1 .+ counts) ./ (1 + nflips)

    acc = findall(>(alpha), pv)
    if isempty(acc)
        return (lo=nothing, hi=nothing, contiguous=true, grid_truncated=false,
                beta_tilde=beta_tilde, C=length(cs), effective_C=Ceff,
                kappa=cs.kappa, grid=collect(grid), pvalue=pv)
    end
    first_acc, last_acc = first(acc), last(acc)
    left_truncated = first_acc == 1
    right_truncated = last_acc == ngrid
    lo = left_truncated ? -Inf : grid[first_acc]
    hi = right_truncated ? Inf : grid[last_acc]
    contiguous = all(pv[i] > alpha for i in first_acc:last_acc)
    return (lo=lo, hi=hi, contiguous=contiguous,
            grid_truncated=left_truncated || right_truncated,
            beta_tilde=beta_tilde, C=length(cs), effective_C=Ceff,
            kappa=cs.kappa, grid=collect(grid), pvalue=pv)
end

"""
    cycle_report(y, x, unit, time; alpha=0.05, delta=0.05, method=:structured,
                 nflips=99999, rng, interval=true) -> AdequacyReport

Paper A diagnostic. Reports treatment and realized-score concentration, the
capture the contrast system achieves, and — when
`interval=true` — the exact sign-flip confidence set.

With `blocks=nothing`, concentration is measured across observations as in
Remark rem:diag. With dependence blocks, treatment mass is aggregated as
`sum_{i in g} xt_i^2`, and realized score concentration uses the squared block
score `(sum_{i in g} xt_i u_i)^2`; this is the clustered analogue used in the
worker--firm application.

The verdict answers the question this module exists for: **is the Gaussian
approximation underlying conventional inference trustworthy on this design?**

Detailed diagnostic caveats are retained in `report.notes` but hidden from the
default display. Call [`show_notes`](@ref) on the returned report to print them;
an `:INCONCLUSIVE` reason remains visible in the concise display.

- `:POINT_PASS` — `lambda_n` is below the user-supplied finite-sample heuristic.
  This is not a theorem-level certificate: the paper requires `lambda_n -> 0`.
- `:FLAGGED` — the identifying variation is concentrated. Conventional critical
  values can enter the regime in which no fixed distribution-free critical
  value is uniformly valid; use the exact confidence set reported here.
- `:INCONCLUSIVE` — the full-enumeration floor `2^(1-effective_C)` exceeds
  `alpha`, so the default non-randomized two-sided test cannot reject at that
  level, whatever the capture.
"""
function cycle_report(y::AbstractVector{<:Real}, x::AbstractVector{<:Real},
                      unit::AbstractVector, time::AbstractVector;
                      alpha::Real=0.05, delta::Real=0.05,
                      method::Symbol=:structured, nflips::Integer=99999,
                      rng::AbstractRNG=Random.default_rng(), interval::Bool=true,
                      lambda_max::Real=0.10,
                      controls=nothing,
                      blocks::Union{Nothing,AbstractVector}=nothing)
    0 < alpha < 1 || throw(ArgumentError("alpha must lie in (0, 1)"))
    0 < delta < 1 - alpha || throw(ArgumentError(
        "delta must lie in (0, 1-alpha)"))
    lambda_max > 0 || throw(ArgumentError("lambda_max must be positive"))
    nflips > 0 || throw(ArgumentError("nflips must be positive"))
    uid, tid, N, T = _integer_codes(unit, time)
    n = length(uid)
    (length(y) == n && length(x) == n) ||
        throw(ArgumentError("y, x, unit, time must have equal length"))
    d_K, ncomp = fe_dimension(uid, tid, N, T)

    cs = cycle_contrasts(x, unit, time; method=method, controls=controls)
    C = length(cs)
    Ceff = length(_active_supports(cs))
    if blocks !== nothing
        length(blocks) == n || throw(ArgumentError(
            "blocks must have the same length as the data"))
        comp = support_compatibility(cs, blocks)
        comp.compatible || throw(ArgumentError(
            "contrast supports are not unions of complete dependence blocks; " *
            "$(comp.incompatible_blocks) block(s) are partial or split"))
    end
    partial = _partial_within_codes(x, uid, tid, N, T; controls=controls)
    xt = partial.xt
    V_n = cs.V_n
    concentration_level = blocks === nothing ? :observation : :block
    bcode = blocks === nothing ? nothing : _codes(blocks)
    treatment_mass = if bcode === nothing
        abs2.(xt)
    else
        out = zeros(Float64, maximum(bcode))
        for k in eachindex(xt)
            out[bcode[k]] += abs2(xt[k])
        end
        out
    end
    treatment_share = treatment_mass ./ V_n
    lambda_n = maximum(treatment_share)
    H_n = sum(abs2, treatment_share)
    n_eff = 1 / H_n

    yt = _partial_outcome_codes(y, uid, tid, N, T, partial.Q)
    beta_ols = dot(xt, yt) / V_n
    u = yt .- beta_ols .* xt
    score_contribution = if bcode === nothing
        xt .* u
    else
        out = zeros(Float64, maximum(bcode))
        for k in eachindex(xt)
            out[bcode[k]] += xt[k] * u[k]
        end
        out
    end
    score_mass = sum(abs2, score_contribution)
    if score_mass > 0
        score_share = abs2.(score_contribution) ./ score_mass
        score_lambda_n = maximum(score_share)
        score_H_n = sum(abs2, score_share)
        score_n_eff = 1 / score_H_n
    else
        score_lambda_n = score_H_n = score_n_eff = NaN
    end
    cyc_dim = n - (N + T) + ncomp - partial.rank

    itv = interval && Ceff > 0 ?
          signflip_interval(y, cs; alpha=alpha, nflips=nflips,
                            rng=rng, blocks=blocks) : nothing

    min_p = Ceff > 0 ? exp2(1 - Ceff) : 1.0
    binary = _binary_floor_info(x, alpha)
    notes = String[]
    partial.rank > 0 && push!(notes, @sprintf(
        "every support contrast annihilates both fixed effects and %d linearly independent continuous nuisance control(s); no global outcome residualization is used in the exact test.",
        partial.rank))
    push!(notes, @sprintf("capture kappa_C = %.4f over C = %d supports (%d treatment-loaded); the capture-implied signal/SE ratio is 1/sqrt(kappa) = %.3fx. Kappa equals Pitman efficiency only when both (P1) and the diffuse-design condition (P2) of Theorem thm:power hold.",
                          cs.kappa, C, Ceff, 1 / sqrt(cs.kappa)))
    push!(notes, @sprintf("cycle-space dimension = %d; the contrast system uses %d supports (%.1f%%). Weighted edge-disjoint cycle packing is NP-hard, so the packing is a heuristic and kappa_C is a LOWER BOUND on achievable capture. Where a discrete treatment makes many four-cycle loadings exactly tied, this implementation ranks on the raw treatment (exact, no alternating-projection round-off), uses canonical label ordering, and compares fixed traversal orders. The selected supports are therefore deterministic, row-order invariant, and identical across languages.",
                          cyc_dim, C, 100 * C / max(cyc_dim, 1)))

    # Any finite lambda cutoff is a user-facing warning convention, not a theorem:
    # Paper A's condition is the sequence statement lambda_n -> 0.
    concentrated = lambda_n > lambda_max
    if concentrated
        push!(notes, @sprintf("CONCENTRATION WARNING: lambda_n = %.3f (N_eff = %.1f) exceeds the heuristic cutoff %.3f. Along sequences where such concentration persists, the limiting t law is not fixed across symmetric error laws of equal variance; Proposition ex:rade therefore rules out a fixed distribution-free critical value at the fully concentrated boundary. It does not rule out every adaptive or bootstrap procedure. The exact contrast test below uses a different finite-sample invariance argument.",
                              lambda_n, n_eff, lambda_max))
    else
        push!(notes, @sprintf("DESCRIPTIVE DESIGN PASS: lambda_n = %.4f (N_eff = %.1f) is below the heuristic cutoff %.3f. This is compatible with the diffuse regime, but it is not a finite-sample certificate: the theorem requires lambda_n -> 0 along a sequence.",
                              lambda_n, n_eff, lambda_max))
    end
    if isfinite(score_lambda_n)
        unit_label = blocks === nothing ? "observation" : "dependence-block"
        push!(notes, @sprintf("realized %s score diagnostic: lambda_score = %.4f and N_eff,score = %.1f. This one-outcome residual diagnostic is a warning statistic, not by itself a consistent estimator of population score concentration (Remark rem:diag).",
                              unit_label, score_lambda_n, score_n_eff))
    end
    if cs.max_share > 0.5
        push!(notes, @sprintf("(P1) WARNING: one support carries %.1f%% of captured variation. The finite-sample exactness theorem is unaffected, but the Gaussian power formula and kappa efficiency interpretation are not licensed when such dominance persists.",
                              100 * cs.max_share))
    end
    needed = 1 + ceil(Int, log2(1 / alpha))
    push!(notes, @sprintf("default two-sided full-enumeration floor is 2^(1-C_eff) = %.2g: global sign reversal duplicates every absolute-statistic orbit value, so level %.3g requires 2^(C_eff-1) >= 1/alpha (C_eff >= %d).",
                          min_p, alpha, needed))
    if blocks === nothing
        push!(notes, "exactness is evaluated for singleton observation blocks: errors must be independent across observations and individually symmetric. For clustered errors, pass blocks=... so support compatibility is checked rather than assumed.")
    else
        push!(notes, @sprintf("block compatibility verified for %d design-defined dependence blocks: every treatment-loaded support is a union of complete blocks.",
                              length(unique(blocks))))
    end
    if itv !== nothing && !itv.contiguous
        push!(notes, "NUMERICAL WARNING: inversion was non-contiguous even though the unstudentized statistic has an interval acceptance set; increase nflips/ngrid and inspect tolerances.")
    end
    if itv !== nothing && itv.grid_truncated
        push!(notes, "the accepted set reached the inversion grid boundary; the reported infinite endpoint is conservative and is not a finite grid edge masquerading as a confidence limit.")
    end

    verdict = min_p > alpha ? :INCONCLUSIVE :
              concentrated ? :FLAGGED : :POINT_PASS
    reason = if verdict !== :INCONCLUSIVE
        ""
    elseif binary.is_binary && !binary.binary_floor_ok
        binary.reason
    else
        @sprintf("selected packing has effective_C = %d; floor 2^(1-%d) = %.6g > alpha = %.6g. A stronger compatible packing may repair this.",
                 Ceff, Ceff, min_p, alpha)
    end
    !isempty(reason) && push!(notes, reason)

    design = _design_summary_codes(uid, tid, N, T; xt=xt)
    stat = (kappa=cs.kappa, C=C, effective_C=Ceff, cycle_dim=cyc_dim,
            concentration_level=concentration_level,
            concentration_blocks=blocks === nothing ? n : maximum(bcode),
            lambda_n=lambda_n, H_n=H_n, n_eff=n_eff,
            score_lambda_n=score_lambda_n, score_H_n=score_H_n,
            score_n_eff=score_n_eff, beta_ols=beta_ols,
            control_rank=partial.rank,
            lambda_score=score_lambda_n, H_score=score_H_n,
            n_eff_score=score_n_eff,
            max_share=cs.max_share, se_price=1 / sqrt(cs.kappa),
            full_enumeration_floor=min_p, min_pvalue=min_p, method=method,
            n_treated=binary.n_treated,
            binary_floor_ok=binary.binary_floor_ok, reason=reason,
            beta_tilde=itv === nothing ? nothing : itv.beta_tilde,
            ci_lo=itv === nothing ? nothing : itv.lo,
            ci_hi=itv === nothing ? nothing : itv.hi,
            ci_level=1 - Float64(alpha),
            ci_contiguous=itv === nothing ? nothing : itv.contiguous,
            ci_grid_truncated=itv === nothing ? nothing : itv.grid_truncated)

    return AdequacyReport(:cycle_inference, design, stat, nothing, nothing,
                          nothing, nothing, verdict, Float64(alpha),
                          Float64(delta), notes)
end
