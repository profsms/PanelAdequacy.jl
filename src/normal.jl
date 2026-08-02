# Normal CDF / quantile via SpecialFunctions (high precision — these feed the
# cross-language parity numbers, so no hand-rolled approximations).

_normcdf(x::Real) = erfc(-x / sqrt(2)) / 2
_norminv(p::Real) = -sqrt(2) * erfcinv(2p)
_normpdf(x::Real) = exp(-x^2 / 2) / sqrt(2pi)

# ------------------------------------------------------------------------------
# Non-central size machinery shared by Modules B and C (Papers B and C use the
# same limit experiment: T => N(eta, 1), size even in eta, leading distortion
# QUADRATIC z*phi(z)*eta^2. The discarded linear surrogate must never be used.)
# ------------------------------------------------------------------------------

"Exact two-sided size of the nominal-alpha test when T ~ N(eta, 1)."
function _noncentral_size(eta::Real, alpha::Real)
    z = _norminv(1 - alpha / 2)
    return _normcdf(-z - abs(eta)) + 1 - _normcdf(z - abs(eta))
end

"""
Exact-inversion threshold eta†(alpha, delta): unique positive root of
size(eta) = alpha + delta (Paper B eq-cv-exact / Paper C cor-cv). The headline
operational threshold; ≈ 0.652 at alpha = delta = 0.05.
"""
function _eta_dagger(alpha::Real, delta::Real)
    0 < delta < 1 - alpha || throw(ArgumentError("need 0 < delta < 1 - alpha"))
    target = alpha + delta
    lo, hi = 0.0, 1.0
    while _noncentral_size(hi, alpha) < target
        hi *= 2
        hi > 1e6 && error("eta_dagger bracket failed")
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        _noncentral_size(mid, alpha) < target ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"Quadratic closed-form companion threshold sqrt(delta / (z phi(z))) (Paper B eq-cv)."
function _eta_quad(alpha::Real, delta::Real)
    z = _norminv(1 - alpha / 2)
    return sqrt(delta / (z * _normpdf(z)))
end

# ------------------------------------------------------------------------------
# Diffuse-companion asymptotic size maps (corrections.tex thm:size, thm:hc), under the
# CONDITIONAL-HOMOSKEDASTIC limits, where omega_eff^2 = omega^2 = sigma^2 and
# the four HC limits collapse to (1-rho)sigma^2, sigma^2, sigma^2, sigma^2/(1-rho).
# Under heteroskedasticity each carries the extra factor omega^2/omega_eff^2
# with omega_eff^2 = (1-rho) omega^2 + rho mu, which these maps do NOT model.
# ------------------------------------------------------------------------------

"""
Asymptotic size of the nominal-alpha t-test using the naive variance u'u/n:
`2(1 - Phi(z sqrt(1-rho)))`, from `T^naive => N(0, 1/(1-rho))` (thm:size).
Under conditional homoskedasticity HC0 shares this limit; under
heteroskedasticity it does not, and the HC0 size also depends on
`omega^2/omega_eff^2`.
"""
function _size_naive(rho::Real, alpha::Real)
    z = _norminv(1 - alpha / 2)
    return 2 * (1 - _normcdf(z * sqrt(1 - rho)))
end

# Retained name: HC0 shares the naive limit under conditional homoskedasticity.
const _size_hc0 = _size_naive

"""
Asymptotic size of the nominal-alpha t-test using HC3: `2(1 - Phi(z/sqrt(1-rho)))`,
from `HC3/V_n ->p omega_eff^2/(1-rho)`. This is BELOW alpha — HC3 under-rejects
in saturated designs.
"""
function _size_hc3(rho::Real, alpha::Real)
    z = _norminv(1 - alpha / 2)
    return 2 * (1 - _normcdf(z / sqrt(1 - rho)))
end

"Breakdown saturation rho† at which the naive/HC0 size reaches alpha + delta."
function _rho_dagger(alpha::Real, delta::Real)
    return 1 - (_norminv(1 - (alpha + delta) / 2) / _norminv(1 - alpha / 2))^2
end

# ------------------------------------------------------------------------------
# Module B critical-value map (Paper B cor-cv, eq-cv)
# ------------------------------------------------------------------------------

"""
    tau2_crit(rho, c2, beta0, sigma; alpha=0.05, delta=0.05,
              method=:exact)

Stock-Yogo-style critical value for the residual treatment variance. The
default follows Paper B's headline exact-normal inversion (eq-cv-exact):

    tau2_crit = beta0^2 c^4 (1-rho) / (sigma^2 eta_dagger^2) - c^2 .

Set `method=:quadratic` for the closed-form companion (eq-cv):

    tau2_crit = beta0^2 c^4 (1-rho) z phi(z) / (sigma^2 delta) - c^2 .

The quadratic boundary is mildly anti-conservative, so it is not the default.
Two features trace to the corrected Hessian limit and are easy to get wrong:
the leading term is LINEAR in `(1-rho)`, not quadratic, and the subtracted term
is `c^2`, not `c^2 (1-rho)`. Returns a negative number when the design is
adequate at every tau^2 (i.e. when `beta0^2 c^2 (1-rho) z phi(z) <= sigma^2 delta`).
"""
function tau2_crit(rho::Real, c2::Real, beta0::Real, sigma::Real;
                   alpha::Real=0.05, delta::Real=0.05,
                   method::Symbol=:exact)
    0 <= rho < 1 || throw(ArgumentError("need 0 <= rho < 1"))
    c2 >= 0 || throw(ArgumentError("c2 must be non-negative"))
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    0 < alpha < 1 || throw(ArgumentError("alpha must lie in (0, 1)"))
    0 < delta < 1 - alpha || throw(ArgumentError(
        "delta must lie in (0, 1-alpha)"))
    method in (:exact, :quadratic) || throw(ArgumentError(
        "method must be :exact or :quadratic"))
    factor = if method === :exact
        inv(_eta_dagger(alpha, delta)^2)
    else
        z = _norminv(1 - alpha / 2)
        z * _normpdf(z) / delta
    end
    return beta0^2 * c2^2 * (1 - rho) * factor / sigma^2 - c2
end

"""
    eta_finite_n(beta0, c2, sigma, tau2, rho)

The finite-n non-centrality mapping (Paper B eq-eta-n) under the local drift
`sigma_nu^2 = c^2/n`:

    eta_n = -beta0 c^2 sqrt(1-rho) / (sigma sqrt(tau^2 + c^2)) .

The `sqrt(1-rho)` factor is the common-scale shrinkage implied by
`X'M_K X/(nQ_K) ->p 1-rho`; it was absent from the pre-correction formula.
"""
function eta_finite_n(beta0::Real, c2::Real, sigma::Real, tau2::Real, rho::Real)
    0 <= rho < 1 || throw(ArgumentError("need 0 <= rho < 1"))
    c2 >= 0 || throw(ArgumentError("c2 must be non-negative"))
    tau2 >= 0 || throw(ArgumentError("tau2 must be non-negative"))
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    tau2 + c2 > 0 || throw(ArgumentError("tau2 + c2 must be positive"))
    return -beta0 * c2 * sqrt(1 - rho) / (sigma * sqrt(tau2 + c2))
end
