# =============================================================================
# Diffuse-regime leverage / variance diagnostics (superseded arXiv 2607.05215)
# No current-paper reference case exists for this companion, so the harness rests on
# (a) exact finite-sample identities verified against dense-matrix computation
# (b) the paper's tabulated asymptotic size predictions (Tables 3 and 4)
# =============================================================================

const PD = PanelAdequacy

"Dense FE dummy matrix for verification."
function dense_dummies(uid, tid, N, T)
    n = length(uid)
    D = zeros(n, N + T)
    for k in 1:n
        D[k, uid[k]] = 1.0
        D[k, N + tid[k]] = 1.0
    end
    return D
end

@testset "Diffuse companion — leverage / variance" begin

    @testset "normal-distribution helpers" begin
        @test PD._normcdf(0.0) == 0.5
        @test PD._normcdf(1.959964) ≈ 0.975 atol = 1e-6
        @test PD._norminv(0.975) ≈ 1.959964 atol = 1e-6
        @test PD._norminv(PD._normcdf(0.6180339)) ≈ 0.6180339 atol = 1e-10
    end

    @testset "implied sizes (Thm 3.1 / Cor 3.1; Table 4 predictions)" begin
        # naive/HC0 over-rejection 2(1 - Phi(z sqrt(1-rho))), paper Table 4 column pred_naive
        @test PD._size_hc0(0.052, 0.05) ≈ 0.056 atol = 1e-3
        @test PD._size_hc0(0.104, 0.05) ≈ 0.064 atol = 1e-3
        @test PD._size_hc0(0.25, 0.05) ≈ 0.090 atol = 1e-3
        @test PD._size_hc0(0.50, 0.05) ≈ 0.166 atol = 1e-3
        # HC3 under-rejection 2(1 - Phi(z / sqrt(1-rho))): below 1% at rho = 0.5 (paper §1)
        @test PD._size_hc3(0.50, 0.05) < 0.01
        @test PD._size_hc3(0.0, 0.05) ≈ 0.05 atol = 1e-12
        # breakdown saturation: HC0 size hits alpha + delta at rho† = 1 - (z_{1-(a+d)/2}/z_{1-a/2})^2
        @test PD._rho_dagger(0.05, 0.05) ≈ 0.2957 atol = 1e-3
        @test PD._size_hc0(PD._rho_dagger(0.05, 0.05), 0.05) ≈ 0.10 atol = 1e-10
    end

    @testset "FE leverage diagonal" begin
        # balanced panel: (P_K)_ii = 1/T + 1/N - 1/n exactly; sum = d_K
        N, T = 6, 5
        uid = repeat(1:N, inner=T); tid = repeat(1:T, outer=N)
        p = fe_leverage(uid, tid)
        @test all(isapprox.(p, 1/T + 1/N - 1/(N*T); atol=1e-10))
        @test sum(p) ≈ N + T - 1 atol = 1e-8

        # unbalanced panel: match dense projection diagonal
        uidu = Int[]; tidu = Int[]
        for i in 1:7, t in 1:5
            (2i + t) % 6 == 0 && continue
            push!(uidu, i); push!(tidu, t)
        end
        D = dense_dummies(uidu, tidu, 7, 5)
        P_dense = D * pinv(D' * D) * D'
        pu = fe_leverage(uidu, tidu)
        @test maximum(abs.(pu .- diag(P_dense))) < 1e-8
        @test sum(pu) ≈ 7 + 5 - 1 atol = 1e-8
    end

    @testset "exact HC identities under uniform full leverage" begin
        # x with two-way-demeaned part of CONSTANT magnitude: xt_it = c*s_i*r_t,
        # s, r balanced sign vectors -> full leverage H_ii = 1/T + 1/N uniform,
        # sum H_ii = N + T = d_K + 1, HC1 == HC2 exactly, and the SE ratios of
        # The companion's Table 3 identities hold exactly.
        N, T = 6, 4
        s = [1, 1, 1, -1, -1, -1]; r = [1, 1, -1, -1]; c = 0.7
        uid = repeat(1:N, inner=T); tid = repeat(1:T, outer=N)
        n = N * T
        x = [c * s[uid[k]] * r[tid[k]] + 0.3 * uid[k] + 0.5 * tid[k] for k in 1:n]
        y = [1.5 * x[k] + sin(3.7k) for k in 1:n]

        rep = leverage_report(y, x, uid, tid)
        st = rep.statistic
        Hbar = 1 / T + 1 / N
        @test st.max_leverage ≈ Hbar atol = 1e-8
        @test st.leverage_spread ≈ 1.0 atol = 1e-8
        @test st.se_hc1 ≈ st.se_hc2 rtol = 1e-8           # Remark 3.4, uniform leverage
        @test st.se_hc0 / st.se_hc2 ≈ sqrt(1 - Hbar) atol = 1e-8
        @test st.se_hc3 / st.se_hc2 ≈ 1 / sqrt(1 - Hbar) atol = 1e-8
        @test st.se_df ≈ st.se_naive * sqrt(n / (n - (N + T - 1) - 1)) rtol = 1e-12
    end

    @testset "against dense full-regression computation (unbalanced)" begin
        uid = Int[]; tid = Int[]
        for i in 1:9, t in 1:6
            (i + 3t) % 8 == 0 && continue
            push!(uid, i); push!(tid, t)
        end
        n = length(uid); N, T = 9, 6
        x = [sin(1.3k) + 0.2 * uid[k] - 0.1 * tid[k] for k in 1:n]
        y = [0.8 * x[k] + cos(2.1k) for k in 1:n]

        # dense reference: full regression of y on [D x]
        D = dense_dummies(uid, tid, N, T)
        A = hcat(D, x)
        Ainv = pinv(A' * A)
        coefs = Ainv * (A' * y)
        beta_dense = coefs[end]
        H_dense = diag(A * Ainv * A')
        u_dense = y .- A * coefs
        rss = sum(abs2, u_dense)
        d_K = N + T - 1
        dof = n - d_K - 1
        xt_dense = (I - D * pinv(D' * D) * D') * x
        tau2 = sum(abs2, xt_dense)
        se_df_dense = sqrt(rss / dof / tau2)
        hc0 = sum(abs2.(xt_dense) .* abs2.(u_dense))
        hc2 = sum(abs2.(xt_dense) .* abs2.(u_dense) ./ (1 .- H_dense))
        hc3 = sum(abs2.(xt_dense) .* abs2.(u_dense) ./ (1 .- H_dense) .^ 2)

        rep = leverage_report(y, x, uid, tid)
        st = rep.statistic
        @test st.beta ≈ beta_dense atol = 1e-8
        @test st.max_leverage ≈ maximum(H_dense) atol = 1e-8
        @test st.se_df ≈ se_df_dense atol = 1e-10
        @test st.se_hc0 ≈ sqrt(hc0) / tau2 atol = 1e-10
        @test st.se_hc2 ≈ sqrt(hc2) / tau2 atol = 1e-10
        @test st.se_hc3 ≈ sqrt(hc3) / tau2 atol = 1e-10
        @test rep.design.d_K == d_K
    end

    @testset "verdict logic and rendering" begin
        # low saturation, strong signal -> CERTIFIED, no HC3 regime note
        N, T = 50, 20
        uid = repeat(1:N, inner=T); tid = repeat(1:T, outer=N)
        n = N * T
        x = [sin(0.9k) + 0.1 * uid[k] for k in 1:n]
        y = [2.0 * x[k] + 0.3 * cos(1.7k) for k in 1:n]
        rep = leverage_report(y, x, uid, tid)
        @test rep.design.rho < 0.1
        @test rep.verdict === :CERTIFIED
        @test !any(occursin("HC3 over-correcting", note) for note in rep.notes)
        @test rep.implied_size ≈ PD._size_hc0(rep.design.rho, 0.05) atol = 1e-12

        # heavy saturation (rho = 11/32 > rho† = 0.296) -> FLAGGED + HC3 regime note
        N2, T2 = 8, 4
        uid2 = repeat(1:N2, inner=T2); tid2 = repeat(1:T2, outer=N2)
        n2 = N2 * T2
        x2 = [cos(1.1k) + 0.2 * tid2[k] for k in 1:n2]
        y2 = [x2[k] + 0.5 * sin(2.3k) for k in 1:n2]
        rep2 = leverage_report(y2, x2, uid2, tid2)
        @test rep2.design.rho > PD._rho_dagger(0.05, 0.05)
        @test rep2.verdict === :FLAGGED
        @test rep2.implied_size > 0.10
        @test any(occursin("HC3 over-correcting", note) for note in rep2.notes)
        @test rep2.breakdown ≈ PD._rho_dagger(0.05, 0.05) atol = 1e-12

        out = sprint(show, MIME("text/plain"), rep2)
        @test occursin("Leverage / Variance (diffuse-regime companion)", out)
        @test occursin("SE(beta)", out)
        @test occursin("VERDICT: FLAGGED", out)

        # degenerate inputs are rejected
        @test_throws ArgumentError leverage_report(y2[1:5], x2[1:5], uid2[1:5], tid2[1:5])  # dof <= 0
        xcoll = [Float64(uid2[k]) for k in 1:n2]  # x in the FE span: no within variation
        @test_throws ArgumentError leverage_report(y2, xcoll, uid2, tid2)
    end

end
