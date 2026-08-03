# =============================================================================
# Module C — staggered-DiD / TWFE-heterogeneity adequacy
# SOURCE: Paper C, "Is Bias Correction Enough? A Design Diagnostic for TWFE
# Inference under Treatment-Effect Heterogeneity" (paper_c_twfe_v1.tex, the
# submitted version). The earlier title, "Stock-Yogo Critical Values for the
# Two-Way Fixed-Effects t-Test under Treatment-Effect Heterogeneity", is
# superseded; NO formula changed between the two drafts.
#
# The submitted paper states that Gamma, its restricted variants, psi_hat,
# Gamma_CR, the covariance-corrected pilots and the implied sizes are ALL
# implemented here, and that both audit tables reproduce from the two adoption
# panels bundled with the package via reproduction/reproduce_with_package.jl.
# Verified: 19/19 parity checks pass.
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
#   cluster    psi = sum_i d_i' R_i d_i / n_w, Gamma_{S,CR} = Gamma_S/sqrt(psi)
#              (Thm. thm-cluster) — psi computed on the realized design.
#   pilot      COVARIANCE-AWARE (eq-pilot): each subspace pilot squares to
#                c_S^2 = n_w max{0, (1/N1)||Pi_S(delta-mean)||^2
#                                    - (1/N1) tr(Pi_S~ A Omega A' Pi_S~)},
#              delta = cell-level group-time ATTs, Omega = Cov(hat Delta_{g,t})
#              estimated by the fixed-design wild cluster bootstrap. Subtracting
#              an average of marginal variances (the legacy scalar shrinkage)
#              under-removes the projected noise when the ATT(g,t) share controls.
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
# projection onto the cohort / event-time / additive-combined subspaces.
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
_proj(B::AbstractMatrix, x::AbstractVector) = B * (pinv(B' * B) * (B' * x))

"Hat matrix P_S = B (B'B)^+ B'."
_hat(B::AbstractMatrix) = B * pinv(B' * B) * B'

"""
Restricted design statistics and the cell bases. `g_cell`, `e_cell` are the
cohort and event-time of each treated cell. Returns a NamedTuple with the four
Gamma_S and the three bases (coh, evt, cmb).
"""
function _restricted_gammas(w::Vector{Float64}, g_cell::Vector{Float64},
                            e_cell::Vector{Float64}, N1::Int)
    d = w .- 1 / N1
    Bcoh = _indicator_basis(g_cell)
    Bevt = _indicator_basis(e_cell)
    Bcmb = hcat(Bcoh, Bevt)
    gam(B) = sqrt(N1) * norm(_proj(B, d))
    return (unr = sqrt(N1) * norm(d), coh = gam(Bcoh), evt = gam(Bevt),
            cmb = gam(Bcmb), Bcoh = Bcoh, Bevt = Bevt, Bcmb = Bcmb)
end

# ------------------------------------------------------------------------------
# Group-time ATTs: simplified not-yet-treated difference-in-means at (g,t)
# resolution (a heterogeneity-robust order-of-magnitude pilot, NOT csdid; pass
# your own via `cohort_effects` for a paper-grade pilot). Always-treated units
# have been dropped, so the control pool is not-yet-treated + never-treated.
# ------------------------------------------------------------------------------

function _group_time_atts(uid::Vector{Int}, tid::Vector{Int}, ftc::Vector{Float64},
                          y::Vector{Float64}, T::Int)
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
            ctrl = findall(i -> ftc[i] > t, 1:N)   # not-yet-treated or never
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
# Covariance-aware pilot (eq-pilot). `A` maps the (ordered) group-time vector to
# treated cells; `Pt[s]` is the centered projector P_S~ = P_S - J/N1.
# ------------------------------------------------------------------------------

"Cell-level effect vector delta from the gt dict (NaN where unidentified)."
function _delta_cell(gt::Dict{Tuple{Int,Int},Float64}, g_cell::Vector{Float64},
                     t_cell::Vector{Int}, N1::Int)
    return Float64[get(gt, (Int(g_cell[k]), t_cell[k]), NaN) for k in 1:N1]
end

"Design-constant noise traces (1/N1) tr(P_S~ A Omega A' P_S~) per subspace."
function _cov_traces(A::Matrix{Float64}, Omega::Matrix{Float64},
                     Pt::NamedTuple, N1::Int)
    M = A * Omega * A'
    return (coh = sum(Pt.coh .* M) / N1, evt = sum(Pt.evt .* M) / N1,
            cmb = sum(Pt.cmb .* M) / N1)
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
            cmb = pil(bases.Bcmb, traces.cmb))
end

# ------------------------------------------------------------------------------
# Fixed-design wild cluster bootstrap: Omega + calibration-uncertainty draws.
# ------------------------------------------------------------------------------

function _wild_bootstrap(uid, tid, ftc, D, Dt, y, N, T, n_w, treated, N1,
                         g_cell, t_cell, bases, gammas_CR, d_K, cohorts, gt0,
                         B::Int, seed::Int)
    n = length(uid)
    # saturated mean surface: unit + time + cohort-by-time (treated cells)
    Umat = zeros(n, N); for k in 1:n; Umat[k, uid[k]] = 1.0; end
    Tmat = zeros(n, T); for k in 1:n; Tmat[k, tid[k]] = 1.0; end
    gtlab = [D[k] == 1.0 ? "$(Int(ftc[uid[k]]))_$(tid[k])" : "none" for k in 1:n]
    GTmat = _indicator_basis(gtlab)
    X = hcat(Umat, Tmat, GTmat)
    yhat = X * (pinv(X' * X) * (X' * y))
    e = y .- yhat

    keys_gt = sort(collect(keys(gt0)))
    flat(gt) = Float64[get(gt, k, NaN) for k in keys_gt]

    dof = n - d_K - 1
    rng = MersenneTwister(seed)
    rows_sigma = Float64[]; rows_psi = Float64[]
    gtmat = Vector{Vector{Float64}}()
    eta_real = Float64[]
    for _ in 1:B
        v = rand(rng, (-1.0, 1.0), N)
        ystar = yhat .+ v[uid] .* e
        yt = _twoway_demean(ystar, uid, tid, N, T)
        beta = dot(Dt, yt) / n_w
        resid = yt .- beta .* Dt
        sigma = sqrt(sum(abs2, resid) / dof)
        rho = _rho_ar1(resid, uid, tid)
        psi = _psi_parametric(Dt, uid, tid, rho; kind=:ar1)
        (isfinite(psi) && psi > 0) || continue
        gt, _ = _group_time_atts(uid, tid, ftc, ystar, T)
        push!(rows_sigma, sigma); push!(rows_psi, psi); push!(gtmat, flat(gt))
        # realized cluster-robust eta from cohort means
        cm = _cohort_means(gt, cohorts)
        cell = Float64[get(cm, g_cell[k], NaN) for k in 1:N1]
        good = .!isnan.(cell)
        if count(good) >= 2
            w = Dt[treated] ./ sum(Dt[treated])
            ab = mean(cell[good])
            er = (dot(w[good], cell[good]) / sum(w[good]) - ab) * sqrt(n_w) / sigma
            push!(eta_real, er / sqrt(psi))
        else
            push!(eta_real, NaN)
        end
    end
    # Omega = wild-cluster covariance of the group-time vector
    G = reduce(hcat, gtmat)'          # (ndraw x m)
    Omega = cov(G; dims=1)
    return Omega, rows_sigma, rows_psi, gtmat, eta_real, keys_gt
end

# ------------------------------------------------------------------------------
# Public: pre-outcome design vetting
# ------------------------------------------------------------------------------

"""
    twfe_design(unit, time, first_treat; alpha=0.05, delta=0.05) -> AdequacyReport

Pre-outcome design vetting from the adoption pattern ALONE: the design statistic
`Gamma = sqrt(N1)||w - u||` and its restricted variants `Gamma_coh`, `Gamma_evt`,
`Gamma_c+e` (Prop. prop-restricted), the negative-weight share, and the breakdown
heterogeneity-to-noise ratios `(c/sigma)† = eta†/Gamma_S`. `first_treat` is the
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
    G = _restricted_gammas(w, g_cell, e_cell, N1)
    d_K, ncomp = fe_dimension(uid, tid, N, T)
    design = _design_summary_codes(uid, tid, N, T; xt=Dt)
    eta_dag = _eta_dagger(alpha, delta)

    notes = String[]
    n_drop > 0 && push!(notes, @sprintf("%d always-treated unit(s) dropped (no observed untreated period; setup g >= 2)", n_drop))
    statistic = (Gamma=Gamma, Gamma_coh=G.coh, Gamma_evt=G.evt, Gamma_cmb=G.cmb,
                 neg_share=neg_share, N1=N1, n_w=n_w)
    if Gamma <= 1e-8
        push!(notes, "block design: within-transformed treatment is uniform, so NO heterogeneity profile distorts the t-test (Prop. prop-gamma0)")
        return AdequacyReport(:twfe_heterogeneity, design, statistic, nothing,
                              eta_dag, Inf, nothing, :CERTIFIED, Float64(alpha),
                              Float64(delta), notes)
    end
    breakdown = eta_dag / G.cmb   # combined-class breakdown (headline)
    push!(notes, @sprintf("pre-outcome: naive TWFE inference is size-controlled iff the combined-class ratio c/sigma <= %.3f (= eta†/Gamma_c+e); supply the outcome (twfe_adequacy) to pilot c/sigma", breakdown))
    return AdequacyReport(:twfe_heterogeneity, design, statistic, nothing,
                          eta_dag, breakdown, nothing, :INCONCLUSIVE,
                          Float64(alpha), Float64(delta), notes)
end

"""
    twfe_gammas(unit, time, first_treat) -> NamedTuple

Just the design-statistic ladder `(unr, coh, evt, cmb, neg_share, N1, n_w)` from
the adoption pattern (always-treated dropped). Convenience accessor for scripts.
"""
function twfe_gammas(unit::AbstractVector, time::AbstractVector{<:Real},
                     first_treat::AbstractVector)
    r = twfe_design(unit, time, first_treat)
    s = r.statistic
    return (unr=s.Gamma, coh=s.Gamma_coh, evt=s.Gamma_evt, cmb=s.Gamma_cmb,
            neg_share=s.neg_share, N1=s.N1, n_w=s.n_w)
end

# ------------------------------------------------------------------------------
# Public: full inference layer
# ------------------------------------------------------------------------------

"""
    twfe_adequacy(y, unit, time, first_treat; alpha=0.05, delta=0.05,
                  cluster=:ar1, psi=nothing, bootstrap=999, seed=20260715)
        -> AdequacyReport

Full TWFE-heterogeneity audit (Paper C). Computes the restricted design-statistic
ladder, the cluster-robust rescaling `Gamma_{S,CR} = Gamma_S/sqrt(psi_hat)`, the
COVARIANCE-AWARE pilots `c_S/sigma` (eq-pilot; `Omega` from a fixed-design wild
cluster bootstrap of `bootstrap` draws), the combined-class worst-case implied
size (the headline) with its bootstrap median and interval, and the verdict.

- `bootstrap`: number of wild-cluster draws (`>0` enables the covariance
  correction and the size intervals; `0` falls back to the raw pilot with a note).
- `cluster = :ar1` (default) or `:iid`, or pass `psi = <value>`.
- Always-treated units are dropped (setup g >= 2), with a note.
"""
function twfe_adequacy(y::AbstractVector{<:Real}, unit::AbstractVector,
                       time::AbstractVector{<:Real}, first_treat::AbstractVector;
                       alpha::Real=0.05, delta::Real=0.05,
                       cluster::Symbol=:ar1, psi::Union{Nothing,Real}=nothing,
                       bootstrap::Integer=999, seed::Integer=20260715)
    cluster in (:ar1, :iid) ||
        throw(ArgumentError("cluster must be :ar1 or :iid (or pass psi=...)"))
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
    G = _restricted_gammas(w, g_cell, e_cell, N1)
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
    psi_driven = _psi_driven(Dt, resid, uid, n_w, sigma^2, N)
    psi_hat = psi !== nothing ? Float64(psi) :
              cluster === :iid ? 1.0 : _psi_parametric(Dt, uid, tid, rho_ar1; kind=:ar1)
    CR = (unr=G.unr/sqrt(psi_hat), coh=G.coh/sqrt(psi_hat),
          evt=G.evt/sqrt(psi_hat), cmb=G.cmb/sqrt(psi_hat))

    # ---- group-time ATTs (point) ----
    gt0, cohorts = _group_time_atts(uid, tid, ftc, yv, T)
    cm0 = _cohort_means(gt0, cohorts)
    cell0 = Float64[get(cm0, g_cell[k], NaN) for k in 1:N1]
    good0 = .!isnan.(cell0)
    att_bar = count(good0) >= 1 ? mean(cell0[good0]) : NaN
    eta_real_iid = count(good0) >= 2 ?
        (dot(w[good0], cell0[good0]) / sum(w[good0]) - att_bar) * sqrt(n_w) / sigma : NaN
    eta_real_cr = eta_real_iid / sqrt(psi_hat)

    # ---- covariance-aware pilots via the wild bootstrap ----
    bases = (Bcoh=G.Bcoh, Bevt=G.Bevt, Bcmb=G.Bcmb)
    local pilots, size_pt, boot
    if bootstrap > 0 && !isempty(gt0)
        Omega, bsig, bpsi, bgt, beta_eta, keys_gt =
            _wild_bootstrap(uid, tid, ftc, D, Dt, yv, N, T, n_w, treated, N1,
                            g_cell, t_cell, bases, CR, d_K, cohorts, gt0,
                            Int(bootstrap), Int(seed))
        # incidence A and centered projectors on the FIXED keys
        kidx = Dict(k => j for (j, k) in enumerate(keys_gt))
        m = length(keys_gt)
        A = zeros(N1, m)
        for k in 1:N1
            key = (Int(g_cell[k]), t_cell[k])
            haskey(kidx, key) && (A[k, kidx[key]] = 1.0)
        end
        J = fill(1.0 / N1, N1, N1)
        Pt = (coh=_hat(G.Bcoh) .- J, evt=_hat(G.Bevt) .- J, cmb=_hat(G.Bcmb) .- J)
        traces = _cov_traces(A, Omega, Pt, N1)
        pilots = _cov_pilots(gt0, g_cell, t_cell, N1, bases, traces, n_w, sigma)
        size_pt = (coh=_noncentral_size(pilots.coh * CR.coh, alpha),
                   evt=_noncentral_size(pilots.evt * CR.evt, alpha),
                   cmb=_noncentral_size(pilots.cmb * CR.cmb, alpha))
        # per-draw combined-class sizes and psi intervals
        draw_cmb = Float64[]
        for i in eachindex(bsig)
            gt = Dict(zip(keys_gt, bgt[i]))
            p = _cov_pilots(gt, g_cell, t_cell, N1, bases, traces, n_w, bsig[i])
            push!(draw_cmb, _noncentral_size(p.cmb * G.cmb / sqrt(bpsi[i]), alpha))
        end
        q(v, p) = quantile(sort(filter(isfinite, v)), p)
        boot = (n=length(bsig), cmb_med=q(draw_cmb, 0.5),
                cmb_lo=q(draw_cmb, 0.025), cmb_hi=q(draw_cmb, 0.975),
                cmb_p95=q(draw_cmb, 0.95), psi_lo=q(bpsi, 0.025), psi_hi=q(bpsi, 0.975),
                Omega_trace_cmb=traces.cmb)
        push!(notes, @sprintf("covariance-aware pilot (eq-pilot): Omega from %d wild-cluster draws; combined-class trace removes the shared-control estimation noise", boot.n))
    else
        # fallback: raw (uncorrected) projected dispersion, flagged
        rawpil(B) = begin
            delta = _delta_cell(gt0, g_cell, t_cell, N1)
            cov = .!isnan.(delta); dc = delta .- mean(delta[cov])
            comp = _proj(B[cov, :], dc[cov])
            sqrt(n_w * dot(comp, comp) / count(cov) / sigma^2)
        end
        pilots = (coh=rawpil(G.Bcoh), evt=rawpil(G.Bevt), cmb=rawpil(G.Bcmb))
        size_pt = (coh=_noncentral_size(pilots.coh * CR.coh, alpha),
                   evt=_noncentral_size(pilots.evt * CR.evt, alpha),
                   cmb=_noncentral_size(pilots.cmb * CR.cmb, alpha))
        boot = nothing
        push!(notes, "bootstrap disabled: pilots are RAW projected dispersion, NOT covariance-corrected — the reported worst-case sizes are upward-biased (Paper C eq-pilot); set bootstrap>0")
    end

    if psi_hat != 1.0
        dir = psi_hat > 1 ?
            "clustering shrinks the non-centrality here; the iid size is an upper bound" :
            "clustering WORSENS the distortion here (psi < 1)"
        push!(notes, @sprintf("psi_hat = %.3f (AR(1) rho = %.3f): %s (direction computed, not assumed)", psi_hat, rho_ar1, dir))
    end
    if psi_hat > 0 && max(psi_driven / psi_hat, psi_hat / psi_driven) > 1.5
        push!(notes, @sprintf("estimator-driven cross-check psi = %.2f diverges from parametric %.2f (Paper C sec-feasible)", psi_driven, psi_hat))
    end
    N < 40 && push!(notes, @sprintf("few clusters (G = %d): CR3/jackknife or wild bootstrap refinements advisable (Rem. sec-clusters)", N))

    eta_dag = _eta_dagger(alpha, delta)
    eta_cmb = pilots.cmb * CR.cmb                      # combined-class worst-case
    verdict = size_pt.cmb <= alpha + delta ? :CERTIFIED : :FLAGGED
    design = _design_summary_codes(uid, tid, N, T; xt=Dt)
    statistic = (Gamma=Gamma, Gamma_coh=G.coh, Gamma_evt=G.evt, Gamma_cmb=G.cmb,
                 neg_share=neg_share, psi_hat=psi_hat, psi_driven=psi_driven,
                 rho_ar1=rho_ar1, Gamma_CR=CR.unr, Gamma_coh_CR=CR.coh,
                 Gamma_evt_CR=CR.evt, Gamma_cmb_CR=CR.cmb, beta=beta, sigma=sigma,
                 att_bar=att_bar, N1=N1, n_w=n_w, n_cohorts=length(cohorts),
                 pilot_coh=pilots.coh, pilot_evt=pilots.evt, pilot_cmb=pilots.cmb,
                 size_coh=size_pt.coh, size_evt=size_pt.evt, size_cmb=size_pt.cmb,
                 size_realized=_noncentral_size(eta_real_cr, alpha),
                 eta_real_cr=eta_real_cr, boot=boot)
    return AdequacyReport(:twfe_heterogeneity, design, statistic, eta_cmb, eta_dag,
                          eta_dag / CR.cmb, size_pt.cmb, verdict,
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

"Estimator-driven cross-check: psi from the CR1 sandwich over homoskedastic sigma^2."
function _psi_driven(Dt::Vector{Float64}, resid::Vector{Float64},
                     uid::Vector{Int}, n_w::Float64, sigma2::Float64, G::Int)
    meat = zeros(G)
    for k in eachindex(Dt)
        meat[uid[k]] += Dt[k] * resid[k]
    end
    return (G / (G - 1)) * sum(abs2, meat) / (n_w * sigma2)
end
