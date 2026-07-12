# =============================================================================
# Module C — staggered-DiD / TWFE-heterogeneity adequacy
# (spec §5; Paper C paper_c_twfe_v1.tex)
#
# Formulas from Paper C:
#   design statistic   Gamma = sqrt(N1) ||w - u||_2, w = Dt/n_w on treated cells,
#                      u = 1/N1 (Def. def-gamma); Gamma = 0 iff block design
#   worst-case         |eta| = (c/sigma) Gamma (Cor. cor-gamma)
#   threshold          (c/sigma) Gamma_CR <= eta†(alpha, delta) (Cor. cor-cv)
#   heterogeneity pilot c_hat = sd(ATT_g) sqrt(n_w), SHRINKAGE-CORRECTED:
#                      var_shrunk = max(0, var(ATT_hat_g) - mean(se_g^2)) (eq-shrink)
#   cluster layer      psi = sum_i d_i' R_i d_i / n_w, Gamma_CR = Gamma/sqrt(psi)
#                      (Thm. thm-cluster) — psi COMPUTED on the realized design,
#                      direction never asserted (spec §5.3)
#
# API-FREEZE CAVEAT (spec §5.3): the cluster-robust inference layer follows the
# paper_c_twfe_v1 DRAFT; its exact inputs stay flexible until Paper C is final.
# Anything downstream of psi_hat is marked with a note in the report.
# =============================================================================

"Treatment indicator D_it = 1{time >= first_treat} on raw values (missing = never)."
_treatment_indicator(time::AbstractVector, first_treat::AbstractVector) =
    Float64[(!ismissing(first_treat[k]) && time[k] >= first_treat[k]) ? 1.0 : 0.0
            for k in eachindex(time)]

"""
Codes for the staggered design. Time codes are SORTED (cohort arithmetic
compares them); returns per-unit cohort codes: 0.0 = always-treated (adopted
before the sample), Inf = never treated in sample, else the 1-based time code.
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

    # per-unit first_treat: must be constant within unit
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
    # cohort code per unit
    ftc = Vector{Float64}(undef, N)
    for i in 1:N
        f = ft_of[i]
        if ismissing(f) || !isfinite(f) || f > stimes[end]
            ftc[i] = Inf                       # never treated in sample
        elseif f <= stimes[1]
            ftc[i] = 0.0                       # always treated: D = 1, no cohort
        else
            pos = findfirst(t -> Float64(t) == f, stimes)
            pos === nothing &&
                throw(ArgumentError("first_treat value $f is not an observed period"))
            ftc[i] = Float64(pos)
        end
    end
    return uid, tid, N, T, ftc
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

"""
    twfe_design(unit, time, first_treat; alpha=0.05, delta=0.05) -> AdequacyReport

Pre-outcome design vetting (spec §5.1 — Paper C's killer feature): the design
statistic `Gamma = sqrt(N1)||w - u||`, the negative-weight share, and the
breakdown heterogeneity-to-noise ratio `(c/sigma)† = eta†/Gamma`, from the
adoption pattern ALONE. `first_treat` is the unit's adoption time on the same
scale as `time` (`missing` = never treated; values before the sample =
always-treated).
"""
function twfe_design(unit::AbstractVector, time::AbstractVector{<:Real},
                     first_treat::AbstractVector;
                     alpha::Real=0.05, delta::Real=0.05)
    uid, tid, N, T, ftc = _staggered_codes(unit, time, first_treat)
    n = length(uid)
    D = _treatment_indicator(time, first_treat)
    Dt = _twoway_demean(D, uid, tid, N, T)
    _, N1, n_w, _, Gamma, neg_share = _design_stats(D, Dt)
    d_K, ncomp = fe_dimension(uid, tid, N, T)
    design = DesignSummary(n, N, T, d_K, d_K / n, ncomp, n_w)
    eta_dag = _eta_dagger(alpha, delta)

    statistic = (Gamma=Gamma, neg_share=neg_share, N1=N1, n_w=n_w)
    if Gamma <= 1e-8
        notes = ["block design: within-transformed treatment is uniform on treated cells, so NO heterogeneity profile can distort the t-test (eta = 0 for every profile; Paper C, Prop. prop-gamma0)"]
        return AdequacyReport(:twfe_heterogeneity, design, statistic, nothing,
                              eta_dag, Inf, nothing, :CERTIFIED, Float64(alpha),
                              Float64(delta), notes)
    end
    breakdown = eta_dag / Gamma   # (c/sigma)†: max heterogeneity-to-noise ratio
    notes = [
        @sprintf("pre-outcome design statistic: naive TWFE inference is size-controlled iff the heterogeneity-to-noise ratio c/sigma <= %.3f (= eta†/Gamma; Paper C, Cor. cor-cv) — supply the outcome (twfe_adequacy) to pilot c/sigma", breakdown),
    ]
    return AdequacyReport(:twfe_heterogeneity, design, statistic, nothing,
                          eta_dag, breakdown, nothing, :INCONCLUSIVE,
                          Float64(alpha), Float64(delta), notes)
end

# ------------------------------------------------------------------------------
# Cohort-effect pilot (simplified not-yet-treated difference-in-means).
# Deliberately NOT a Callaway–Sant'Anna implementation (spec §0 scope
# discipline): an order-of-magnitude pilot that points to csdid/did.
# ------------------------------------------------------------------------------

function _cohort_pilot(uid::Vector{Int}, tid::Vector{Int}, ftc::Vector{Float64},
                       y::Vector{Float64}, T::Int)
    key = Dict{Tuple{Int,Int},Float64}()
    for k in eachindex(uid)
        key[(uid[k], tid[k])] = y[k]
    end
    N = length(ftc)
    cohorts = sort(unique(f for f in ftc if isfinite(f) && f > 1))
    atts = Float64[]; ses = Float64[]; gs = Float64[]
    for g in cohorts
        gunits = findall(==(g), ftc)
        base = Int(g) - 1
        diffs = Float64[]
        for t in Int(g):T
            # controls: not-yet-treated at t, never-treated, or always-treated
            ctrl = findall(i -> ftc[i] > t || ftc[i] <= 0, 1:N)
            gt = [key[(u, t)] - key[(u, base)] for u in gunits
                  if haskey(key, (u, t)) && haskey(key, (u, base))]
            ct = [key[(u, t)] - key[(u, base)] for u in ctrl
                  if haskey(key, (u, t)) && haskey(key, (u, base))]
            (isempty(gt) || isempty(ct)) && continue
            push!(diffs, mean(gt) - mean(ct))
        end
        isempty(diffs) && continue
        push!(gs, g)
        push!(atts, mean(diffs))
        push!(ses, length(diffs) > 1 ? std(diffs) / sqrt(length(diffs)) : NaN)
    end
    return gs, atts, ses
end

"""
    twfe_adequacy(y, unit, time, first_treat; alpha=0.05, delta=0.05,
                  cluster=:ar1, psi=nothing,
                  cohort_effects=nothing, cohort_ses=nothing) -> AdequacyReport

Module C inference layer (spec §5.2-§5.3). Computes the realized and worst-case
non-centrality of the TWFE t-test under treatment-effect heterogeneity, with
the SHRINKAGE-CORRECTED cohort-dispersion pilot (Paper C eq-shrink) and the
cluster-robust rescaling `Gamma_CR = Gamma/sqrt(psi_hat)` with `psi_hat`
computed on the realized design (no direction assumed).

- `cluster = :ar1` (default; Paper C's parametric route (i)) or `:iid`;
  or pass `psi = <value>` to supply your own variance-inflation factor.
- `cohort_effects` / `cohort_ses`: cohort-level effect estimates (aligned with
  the SORTED adoption times) from a heterogeneity-robust estimator
  (csdid / did / fixest::sunab). If omitted, an internal order-of-magnitude
  pilot is used and labelled as such.
"""
function twfe_adequacy(y::AbstractVector{<:Real}, unit::AbstractVector,
                       time::AbstractVector{<:Real}, first_treat::AbstractVector;
                       alpha::Real=0.05, delta::Real=0.05,
                       cluster::Symbol=:ar1, psi::Union{Nothing,Real}=nothing,
                       cohort_effects::Union{Nothing,AbstractVector{<:Real}}=nothing,
                       cohort_ses::Union{Nothing,AbstractVector{<:Real}}=nothing)
    cluster in (:ar1, :iid) ||
        throw(ArgumentError("cluster must be :ar1 or :iid (or pass psi=...)"))
    uid, tid, N, T, ftc = _staggered_codes(unit, time, first_treat)
    n = length(uid)
    length(y) == n || throw(ArgumentError("y must have length n = $n"))
    D = _treatment_indicator(time, first_treat)
    Dt = _twoway_demean(D, uid, tid, N, T)
    treated, N1, n_w, w, Gamma, neg_share = _design_stats(D, Dt)
    d_K, ncomp = fe_dimension(uid, tid, N, T)
    dof = n - d_K - 1
    dof > 0 || throw(ArgumentError("no residual degrees of freedom"))

    yv = Float64.(y)
    yt = _twoway_demean(yv, uid, tid, N, T)
    beta = dot(Dt, yt) / n_w
    resid = yt .- beta .* Dt
    sigma = sqrt(sum(abs2, resid) / dof)

    notes = String[]

    # ---- cohort-effect pilot ----
    cohorts = sort(unique(f for f in ftc if isfinite(f) && f > 1))
    if cohort_effects === nothing
        _, atts, ses = _cohort_pilot(uid, tid, ftc, yv, T)
        push!(notes, "cohort-effect pilot is a simplified not-yet-treated difference-in-means — an order-of-magnitude pilot, NOT a Callaway–Sant'Anna implementation; pass cohort_effects from csdid/did/sunab for a paper-grade pilot (spec §0 scope discipline)")
    else
        length(cohort_effects) == length(cohorts) || throw(ArgumentError(
            "cohort_effects must have one entry per adoption cohort ($(length(cohorts)), sorted by adoption time)"))
        atts = Float64.(cohort_effects)
        if cohort_ses === nothing
            ses = fill(0.0, length(atts))
            push!(notes, "no cohort SEs supplied: cohort dispersion NOT shrinkage-corrected (assumed noiseless)")
        else
            length(cohort_ses) == length(atts) ||
                throw(ArgumentError("cohort_ses must match cohort_effects"))
            ses = Float64.(cohort_ses)
        end
    end
    good = .!isnan.(ses)
    att_by_cohort = Dict(g => a for (g, a) in zip(cohorts, atts))

    # realized eta: cell-level cohort deviations under the dCdH weights
    cell_att = [get(att_by_cohort, ftc[uid[k]], NaN) for k in treated]
    gcell = .!isnan.(cell_att)
    att_bar = mean(cell_att[gcell])
    eta_real_iid = (dot(w[gcell], cell_att[gcell]) / sum(w[gcell]) - att_bar) *
                   sqrt(n_w) / sigma

    # shrinkage-corrected dispersion (eq-shrink) — REQUIRED default (spec §5.2)
    if count(good) >= 2
        raw_var = var(atts[good])
        noise = mean(abs2, ses[good])
        shr_var = max(0.0, raw_var - noise)
        csd_raw, csd_shr = sqrt(raw_var), sqrt(shr_var)
        shrink_factor = raw_var > 0 ? sqrt(shr_var / raw_var) : 1.0
    else
        csd_raw = csd_shr = 0.0
        shrink_factor = 1.0
        push!(notes, "fewer than two cohorts with usable SEs: cohort dispersion not estimable; worst-case eta unavailable")
    end
    if csd_shr == 0.0 && csd_raw > 0
        push!(notes, @sprintf("cohort dispersion (raw sd %.4g) is entirely attributable to sampling noise — shrunk sd = 0; worst-case eta at the shrunk pilot is 0 (raw-pilot worst-case reported for comparison; Paper C eq-shrink)", csd_raw))
    elseif csd_raw > 0
        push!(notes, @sprintf("shrinkage-corrected cohort dispersion: raw sd %.4g -> shrunk %.4g (factor %.3f) — heterogeneity %s (Paper C eq-shrink)", csd_raw, csd_shr, shrink_factor, shrink_factor > 0.9 ? "genuine, not a small-cohort artifact" : "partly sampling noise"))
    end

    # ---- cluster layer (§5.3 — API flexible until Paper C final) ----
    rho_ar1 = _rho_ar1(resid, uid, tid)
    psi_driven = _psi_driven(Dt, resid, uid, n_w, sigma^2, N)
    psi_hat = psi !== nothing ? Float64(psi) :
              cluster === :iid ? 1.0 : _psi_parametric(Dt, uid, tid, rho_ar1; kind=:ar1)
    Gamma_CR = Gamma / sqrt(psi_hat)

    eta = eta_real_iid / sqrt(psi_hat)
    eta_worst = (csd_shr * sqrt(n_w) / sigma) * Gamma_CR
    eta_worst_raw = (csd_raw * sqrt(n_w) / sigma) * Gamma_CR

    if psi_hat != 1.0
        dir = psi_hat > 1 ?
            "clustering shrinks the non-centrality on THIS design; the iid implied size is an upper bound here" :
            "clustering WORSENS the distortion on THIS design (psi < 1)"
        push!(notes, @sprintf("realized variance-inflation psi_hat = %.3f (AR(1) fit rho = %.3f): %s — direction computed from the design, never assumed (Paper C §5.3)", psi_hat, rho_ar1, dir))
    end
    if psi_hat > 0 && max(psi_driven / psi_hat, psi_hat / psi_driven) > 1.5
        push!(notes, @sprintf("estimator-driven cross-check psi = %.2f diverges from the parametric %.2f; route (ii) is only valid when the homoskedastic sigma_hat is consistent (Paper C §sec-feasible) — parametric route reported", psi_driven, psi_hat))
    end
    N < 40 && push!(notes, @sprintf("few clusters (G = %d): feasible CR1 has its own finite-cluster error; CR3/jackknife or wild cluster bootstrap refinements are advisable (Paper C, Rem. sec-clusters)", N))
    push!(notes, "cluster-robust layer follows the paper_c_twfe_v1 DRAFT; feasible CR1-t validity requires moderate-to-large (G, T), so clustered sizes are design-computed approximations — this layer's API stays flexible until Paper C is final (spec §5.3)")

    verdict = max(abs(eta), eta_worst) <= _eta_dagger(alpha, delta) ?
              :CERTIFIED : :FLAGGED
    eta_dag = _eta_dagger(alpha, delta)
    design = DesignSummary(n, N, T, d_K, d_K / n, ncomp, n_w)
    statistic = (Gamma=Gamma, neg_share=neg_share, Gamma_CR=Gamma_CR,
                 psi_hat=psi_hat, psi_driven=psi_driven, rho_ar1=rho_ar1,
                 beta=beta, sigma=sigma, att_bar=att_bar,
                 cohort_sd_raw=csd_raw, cohort_sd_shrunk=csd_shr,
                 shrink_factor=shrink_factor, eta_worst=eta_worst,
                 eta_worst_raw=eta_worst_raw, N1=N1, n_w=n_w,
                 n_cohorts=length(cohorts))
    return AdequacyReport(:twfe_heterogeneity, design, statistic, eta, eta_dag,
                          eta_dag / Gamma_CR, _noncentral_size(eta, alpha),
                          verdict, Float64(alpha), Float64(delta), notes)
end

# ------------------------------------------------------------------------------
# psi machinery (Paper C Def. def-psi, §sec-feasible)
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
            # d'Rd = (1-rho)||d||^2 + rho (d'1)^2 — exact, and d'1 = 0 after
            # the within transform (Paper C, Lem. lem-center)
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

"Estimator-driven cross-check: psi from the CR1 sandwich over homoskedastic sigma^2."
function _psi_driven(Dt::Vector{Float64}, resid::Vector{Float64},
                     uid::Vector{Int}, n_w::Float64, sigma2::Float64, G::Int)
    meat = zeros(G)
    for k in eachindex(Dt)
        meat[uid[k]] += Dt[k] * resid[k]
    end
    return (G / (G - 1)) * sum(abs2, meat) / (n_w * sigma2)
end
