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
# Paper A asymptotic size maps (Theorem 3.1, Theorem 3.3, Corollary 3.1), under
# the homoskedastic / design-balanced limits (sigma_eff^2 = omega^2).
# ------------------------------------------------------------------------------

"Asymptotic size of the nominal-alpha t-test using naive/HC0 variance: 2(1 - Phi(z sqrt(1-rho)))."
function _size_hc0(rho::Real, alpha::Real)
    z = _norminv(1 - alpha / 2)
    return 2 * (1 - _normcdf(z * sqrt(1 - rho)))
end

"Asymptotic size of the nominal-alpha t-test using HC3 variance: 2(1 - Phi(z / sqrt(1-rho)))."
function _size_hc3(rho::Real, alpha::Real)
    z = _norminv(1 - alpha / 2)
    return 2 * (1 - _normcdf(z / sqrt(1 - rho)))
end

"Breakdown saturation rho† at which the HC0/naive size reaches alpha + delta."
function _rho_dagger(alpha::Real, delta::Real)
    return 1 - (_norminv(1 - (alpha + delta) / 2) / _norminv(1 - alpha / 2))^2
end
