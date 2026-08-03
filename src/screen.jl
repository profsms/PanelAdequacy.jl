"""
    applicable(x, unit, time; alpha=0.05) -> (ok, reason)

Run the outcome-free pre-flight check for Paper A. For a binary treatment with
`n1` treated and `n0` untreated observations, no more than `min(n1,n0)`
disjoint contrasts can carry a nonzero loading, so a two-sided level-`alpha`
test requires `min(n1,n0) >= 1 + log2(1/alpha)`. Other designs pass this structural check; packing
quality is assessed separately by [`cycle_report`](@ref).
"""
function applicable(x::AbstractVector{<:Real}, unit::AbstractVector,
                    time::AbstractVector; alpha::Real=0.05, controls=nothing)
    0 < alpha < 1 || throw(ArgumentError("alpha must lie in (0, 1)"))
    design_summary(unit, time; x=x, controls=controls)
    info = _binary_floor_info(x, alpha)
    return (ok=info.binary_floor_ok, reason=info.reason)
end

"""
    adequacy_row(x, unit, time; y=nothing, method=:structured, alpha=0.05)

Compose the design and cycle diagnostics into one flat screening record. The
base record is outcome-free. Supplying `y` appends realized score concentration
and the Paper A verdict.

`lei_bickel_feasible` evaluates the cyclic-permutation feasibility condition
`n / (d_K + 1) >= 1/alpha - 1` at the requested level. `treated_obs` counts
observations with nonzero raw treatment.
"""
function adequacy_row(x::AbstractVector{<:Real}, unit::AbstractVector,
                      time::AbstractVector;
                      y::Union{Nothing,AbstractVector{<:Real}}=nothing,
                      method::Symbol=:structured, alpha::Real=0.05,
                      controls=nothing)
    0 < alpha < 1 || throw(ArgumentError("alpha must lie in (0, 1)"))
    d = design_summary(unit, time; x=x, controls=controls)
    V_n = something(d.tau_star2)
    V_n > 0 || throw(ArgumentError("regressor has no within variation"))
    greedy = cycle_contrasts(x, unit, time; method=:greedy, controls=controls)
    designed = method === :greedy ? greedy :
               cycle_contrasts(x, unit, time; method=method, controls=controls)
    binary = _binary_floor_info(x, alpha)
    base = (n=d.n, N=d.N, T=d.T, d_K=d.d_K, rho=d.rho,
            treated_obs=count(z -> !iszero(z), x), Vn=V_n,
            n_treated=binary.n_treated,
            binary_floor_ok=binary.binary_floor_ok,
            lambda_n=something(d.lambda_n), n_eff=something(d.n_eff),
            lei_bickel_feasible=d.n / (d.d_K + 1) >= inv(alpha) - 1,
            kappa_greedy=greedy.kappa,
            kappa_designed=designed.kappa,
            se_price=designed.kappa > 0 ? inv(sqrt(designed.kappa)) : Inf)
    y === nothing && return base
    length(y) == d.n || throw(ArgumentError(
        "y, x, unit, time must have equal length"))
    score = score_concentration(y, x, unit, time; controls=controls)
    report = cycle_report(y, x, unit, time; alpha=alpha, method=method,
                          controls=controls, interval=false)
    return merge(base, score, (verdict=report.verdict,))
end
