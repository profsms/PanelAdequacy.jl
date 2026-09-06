# =============================================================================
# Module B — measurement-error adequacy.
# Reference cases: V-Dem, the repeated-report twins application, and the
# extended simulation range. Threshold machinery must be exact-inversion /
# quadratic — NEVER the discarded linear surrogate. Breakdown expectations are
# the FIXED-POINT values lambda† = t*/(t* + eta†) (paper Def. def-breakdown);
# cluster expectations are the by-unit CRVE psi_hat of Remark rem-cluster,
# matching the current V-Dem and twins tables.
# =============================================================================

"Extract a V-Dem spec (y, x, x_sd complete cases) from the parsed CSV columns."
function vdem_spec(cols, xcol::Symbol, sdcol::Symbol)
    ly = tryparse.(Float64, cols[:ly])
    xv = tryparse.(Float64, cols[xcol])
    sv = tryparse.(Float64, cols[sdcol])
    keep = findall(k -> ly[k] !== nothing && xv[k] !== nothing && sv[k] !== nothing,
                   eachindex(ly))
    return (unit = cols[:iso][keep], time = cols[:year][keep],
            y = Float64[ly[k] for k in keep], x = Float64[xv[k] for k in keep],
            sd = Float64[sv[k] for k in keep])
end

@testset "Module B — measurement error" begin

    @testset "threshold constants (measurement-error article)" begin
        @test PD._eta_dagger(0.05, 0.05) ≈ 0.652 atol = 5e-4
        @test PD._eta_dagger(0.05, 0.01) ≈ 0.295 atol = 5e-4
        @test PD._eta_quad(0.05, 0.05) ≈ 0.661 atol = 5e-4
        @test PD._eta_quad(0.05, 0.01) ≈ 0.296 atol = 1e-3   # paper rounds 0.29546
        z = PD._norminv(0.975)
        @test z * PD._normpdf(z) ≈ 0.11455 atol = 2e-5
        # quadratic form is mildly anti-conservative: eta_quad > eta_dagger
        @test PD._eta_quad(0.05, 0.05) > PD._eta_dagger(0.05, 0.05)
        # exact size: even in eta, equals alpha at eta = 0, increasing in |eta|
        @test PD._noncentral_size(0.0, 0.05) ≈ 0.05 atol = 1e-12
        @test PD._noncentral_size(-1.3, 0.05) ≈ PD._noncentral_size(1.3, 0.05) atol = 1e-14
        @test PD._noncentral_size(PD._eta_dagger(0.05, 0.05), 0.05) ≈ 0.10 atol = 1e-10
        # Current simulation endpoints: Design 4 iid and Design 5 clustered.
        @test PD._noncentral_size(3.58, 0.05) ≈ 0.9474 atol = 5e-5
        @test PD._noncentral_size(3.67, 0.05) ≈ 0.9564 atol = 5e-5
        eta_dag = PD._eta_dagger(0.05, 0.05)
        z_beta = PD._norminv(0.95)
        @test breakdown_reliability(2.0, 1.0, 1.0) ≈
              2.0 / (2.0 + eta_dag) atol = 1e-12
        @test certified_breakdown_reliability(2.0, 1.0, 1.0) ≈
              (2.0 + z_beta) / (2.0 + z_beta + eta_dag) atol = 1e-12
        @test certified_breakdown_reliability(2.0, 1.0, 1.0) >
              breakdown_reliability(2.0, 1.0, 1.0)
    end

    @testset "reliability helpers" begin
        @test reliability_from_interval([0.1, 0.2], [0.3, 0.5]) ≈ [0.1, 0.15]
        # Observed-scale reliability: Var(nu) = (1-r) Var(X*)
        @test reliability_from_ratio(0.8, 2.0) ≈ sqrt(0.2) * 2.0
        # The old formula is still available when the supplied SD is latent signal.
        @test reliability_from_ratio(0.8, 2.0; scale=:signal) ≈ sqrt(0.25) * 2.0
        xrep = [-2.0, -1.0, 1.0, 2.0]
        zrep = [-1.8, -1.2, 0.9, 2.1]
        @test reliability_from_repeats(xrep, zrep) ≈ dot(xrep, zrep) / dot(xrep, xrep)
        @test reliability_from_repeats(xrep, zrep; method=:equal_variance) ≈
              1 - dot(xrep - zrep, xrep - zrep) / (2dot(xrep, xrep))
        @test_throws ArgumentError reliability_from_ratio(1.2, 2.0)
        @test_throws ArgumentError reliability_from_ratio(0.8, -1.0)
        @test_throws ArgumentError reliability_from_interval([0.3], [0.2])
        @test_throws ArgumentError reliability_from_repeats([1.0], [1.0])
        @test_throws ArgumentError reliability_from_repeats([1.0, 2.0], [1.0])
    end

    @testset "primitive threshold: exact inversion is the default" begin
        rho, c2, beta0, sigma = 0.2, 3.0, 0.7, 1.4
        eta_dag = PD._eta_dagger(0.05, 0.05)
        expected = beta0^2 * c2^2 * (1 - rho) / (sigma^2 * eta_dag^2) - c2
        @test tau2_crit(rho, c2, beta0, sigma) ≈ expected atol=1e-12
        z = PD._norminv(0.975)
        quad = beta0^2 * c2^2 * (1 - rho) * z * PD._normpdf(z) /
               (sigma^2 * 0.05) - c2
        @test tau2_crit(rho, c2, beta0, sigma; method=:quadratic) ≈ quad atol=1e-12
        @test tau2_crit(rho, c2, beta0, sigma) >
              tau2_crit(rho, c2, beta0, sigma; method=:quadratic)
    end

    @testset "reference case 2a: V-Dem two-pole (spec §7.2, eiv_vdem_results static rows)" begin
        cols = read_simple_csv(joinpath(TESTDATA, "eiv_vdem_panel.csv"))

        # --- aggregate polyarchy: CERTIFIED ---
        s = vdem_spec(cols, :v2x_polyarchy, :v2x_polyarchy_sd)
        rep = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd, pilot=:point)
        @test rep.design.n == 8930
        @test rep.design.d_K == 221
        st = rep.statistic
        @test st.lambda_hat ≈ 0.8937 atol = 1e-3
        @test st.beta_star ≈ 0.06096 rtol = 2e-3
        @test st.beta_corr ≈ 0.06821 rtol = 2e-3
        @test st.sigma ≈ 0.32956 rtol = 2e-3
        @test rep.design.tau_star2 ≈ 127.826 rtol = 2e-3
        @test rep.eta ≈ 0.24884 rtol = 5e-3
        @test rep.implied_size ≈ 0.05712 atol = 5e-4
        @test rep.threshold ≈ 0.652 atol = 5e-4
        @test rep.breakdown ≈ 0.762 atol = 2e-3     # fixed point t*/(t*+eta†); paper Table 3
        # point verdict is exactly equivalent to lambda_hat >= breakdown
        @test (rep.statistic.lambda_hat >= rep.breakdown) ==
              (rep.verdict === :POINT_PASS)
        @test rep.verdict === :POINT_PASS   # point pilot: a point pass, NOT a certificate
        # formal (conservative) certificate also passes for polyarchy
        repc = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd)
        @test repc.verdict === :CERTIFIED
        @test repc.statistic.eta_upper > repc.eta
        @test repc.statistic.breakdown_certified ≈ 0.8513 atol = 2e-3
        @test repc.statistic.reliability_lower ≈ repc.statistic.lambda_hat
        @test repc.statistic.false_certification_bound == 0.05
        # A noisy reliability pilot must enter through a lower bound and gets
        # its own error budget; the two errors add without independence.
        repl = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd,
                            reliability_lower=0.80, gamma=0.025,
                            gamma_lambda=0.025)
        @test repl.verdict === :FLAGGED
        @test repl.statistic.reliability_lower == 0.80
        @test repl.statistic.false_certification_bound == 0.05
        # cluster-robust (country CRVE): paper Table 3 psi_hat = 19.2, still certified
        repcr = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd,
                             pilot=:point, cluster=:crve)
        @test repcr.statistic.psi_hat ≈ 19.18 rtol = 1e-2
        @test repcr.implied_size ≈ 0.050 atol = 1e-3
        @test repcr.verdict === :POINT_PASS
        @test haskey(repcr.statistic.cluster, :projection_ratio)
        @test repcr.statistic.cluster.projection_ratio ≈ 0.007 atol = 8e-4

        # --- legislative constraints: FLAGGED ---
        s = vdem_spec(cols, :v2xlg_legcon, :v2xlg_legcon_sd)
        rep = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd, pilot=:point)
        @test rep.design.n == 8529
        @test rep.statistic.lambda_hat ≈ 0.4985 atol = 1e-3
        @test rep.eta ≈ 1.0763 rtol = 5e-3
        @test rep.implied_size ≈ 0.1896 atol = 1e-3
        @test rep.breakdown ≈ 0.621 atol = 2e-3     # fixed point; paper Table 3
        @test rep.verdict === :FLAGGED
        # the paper's middle case: flagged iid, point pass under clustering
        repcr = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd,
                             pilot=:point, cluster=:crve)
        @test repcr.statistic.psi_hat ≈ 24.05 rtol = 1e-2
        @test repcr.implied_size ≈ 0.0555 atol = 1e-3
        @test repcr.verdict === :POINT_PASS
        @test (repcr.statistic.lambda_hat >= repcr.breakdown) ==
              (repcr.verdict === :POINT_PASS)

        # THE naive-pilot danger (Prop. prop-pilot(i) / Design 3a, on real data):
        # the attenuated pilot CERTIFIES this genuinely-failing specification
        repn = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd, pilot=:naive)
        @test repn.eta ≈ 1.0763 * 0.4985 rtol = 1e-2   # eta understated by factor lambda
        @test repn.verdict === :POINT_PASS              # the exact error the diagnostic prevents
        @test any(occursin("ANTI-CONSERVATIVE", n) for n in repn.notes)

        # --- judicial constraints: FLAGGED decisively, exact size 1.00 ---
        s = vdem_spec(cols, :v2x_jucon, :v2x_jucon_sd)
        rep = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd, pilot=:point)
        @test rep.design.n == 8889
        @test rep.statistic.lambda_hat ≈ 0.3788 atol = 1e-3
        @test rep.eta ≈ 13.93 rtol = 1e-2
        @test rep.implied_size ≈ 1.0 atol = 1e-6
        @test rep.breakdown ≈ 0.929 atol = 2e-3     # fixed point; paper Table 3
        @test rep.verdict === :FLAGGED
        @test any(occursin("quadratic", n) for n in rep.notes)   # far-out honesty note
        # flag SURVIVES clustering: psi_hat = 25.2 but eta_CR = 2.77, size 79%
        repcr = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd,
                             pilot=:point, cluster=:crve)
        @test repcr.statistic.psi_hat ≈ 25.21 rtol = 1e-2
        @test repcr.eta ≈ 2.774 rtol = 1e-2
        @test repcr.implied_size ≈ 0.792 atol = 3e-3
        @test repcr.verdict === :FLAGGED
    end

    @testset "reference case 2b: gate-1 headline (spec §7.2, vdem_gate1.csv)" begin
        cols = read_simple_csv(joinpath(TESTDATA, "vdem_gate1.csv"))
        ly = parse.(Float64, cols[:ly])
        x = parse.(Float64, cols[:poly])
        sd = parse.(Float64, cols[:poly_sd])
        rep = eiv_adequacy(ly, x, cols[:iso], cols[:year]; sigma_nu=sd, pilot=:point)
        st = rep.statistic
        @test st.lambda_hat ≈ 0.868 atol = 1.5e-3
        @test st.noise_ratio ≈ 0.153 atol = 3e-3
        @test st.beta_star ≈ 0.072 atol = 1e-3
        @test st.beta_corr ≈ 0.083 atol = 1e-3
    end

    @testset "repeated-report twins application" begin
        d = load_dataset("twins")
        needed = (:DLHRWAGE, :DEDUC1, :DEDUC2, :DTEN, :DMARRIED, :DUNCOV)
        keep = [all(!ismissing(getproperty(d, nm)[i]) for nm in needed)
                for i in eachindex(d.DLHRWAGE)]
        yraw = Float64.(d.DLHRWAGE[keep])
        xraw = Float64.(d.DEDUC1[keep])
        zraw = Float64.(d.DEDUC2[keep])
        W = hcat(ones(sum(keep)), Float64.(d.DTEN[keep]),
                 Float64.(d.DMARRIED[keep]), Float64.(d.DUNCOV[keep]))
        y = yraw - W * (W \ yraw)
        x = xraw - W * (W \ xraw)
        z = zraw - W * (W \ zraw)
        tau2 = dot(x, x)
        bstar = dot(x, y) / tau2
        u = y - bstar * x
        n, d_K = length(y), size(W, 2)
        sigma = sqrt(dot(u, u) / (n - d_K - 1))

        lambda_cov = reliability_from_repeats(x, z)
        lambda_equal = reliability_from_repeats(x, z; method=:equal_variance)
        @test n == 147
        @test bstar ≈ 0.0908758941551 atol = 1e-12
        @test sigma / sqrt(tau2) ≈ 0.0219814978882 atol = 1e-12
        @test lambda_cov ≈ 0.5749036782323 atol = 1e-12
        @test lambda_equal ≈ 0.5514169767299 atol = 1e-12

        rcov = eiv_adequacy_summary(bstar, sigma, tau2, n, d_K;
                                    reliability=lambda_cov, pilot=:point)
        req = eiv_adequacy_summary(bstar, sigma, tau2, n, d_K;
                                   reliability=lambda_equal, pilot=:point)
        @test rcov.breakdown ≈ 0.863710397039 atol = 1e-7
        @test req.breakdown ≈ rcov.breakdown atol = 1e-14
        @test rcov.statistic.beta_corr ≈ 0.1580715128394 atol = 1e-12
        @test req.statistic.beta_corr ≈ 0.1648043096058 atol = 1e-12
        # The covariance reliability correction is the reverse-direction IV
        # ratio algebraically; it is corroboration, not external validation.
        @test rcov.statistic.beta_corr ≈ dot(x, y) / dot(x, z) atol = 1e-12
        @test rcov.eta ≈ 3.056917186720 atol = 1e-12
        @test req.eta ≈ 3.363210998044 atol = 1e-12
        @test rcov.implied_size ≈ 0.863669337612 atol = 1e-12
        @test req.implied_size ≈ 0.919728454662 atol = 1e-12
        @test rcov.verdict === :FLAGGED && req.verdict === :FLAGGED

        # Rouse's correlated-report sensitivity uses rounded published inputs.
        rouse = eiv_adequacy_summary(0.071, 1.0, 1 / 0.016^2, 445, 4;
                                     reliability=0.748, pilot=:point)
        @test rouse.breakdown ≈ 0.871831787842 atol = 1e-7
        @test rouse.statistic.beta_corr ≈ 0.0949197860963 atol = 1e-12
        @test rouse.eta ≈ 1.494986631016 atol = 1e-12
        @test rouse.implied_size ≈ 0.321249033980 atol = 1e-12
        @test rouse.verdict === :FLAGGED

        # Summary-form report renders without N/T.
        out = sprint(show, MIME("text/plain"), rcov)
        @test occursin("n=147", out) && occursin("d_K=4", out)
        @test !occursin("N=0", out)
    end

    @testset "controls enter regression df and exact noise trace" begin
        N, T = 6, 4
        unit = repeat(1:N, inner=T)
        time = repeat(1:T, outer=N)
        n = length(unit)
        k = collect(1:n)
        z = sin.(0.37 .* k) .+ 0.05 .* unit
        x = cos.(0.61 .* k) .+ 0.4 .* z .+ 0.1 .* unit
        y = 1.2 .* x .- 0.7 .* z .+ sin.(1.13 .* k)
        sd = 0.03 .+ 0.002 .* k

        # Dense reference nuisance projection: intercept, N-1 unit dummies,
        # T-1 time dummies, and the supplied control.
        K = ones(n, 1 + (N - 1) + (T - 1) + 1)
        col = 2
        for j in 2:N
            K[:, col] .= unit .== j
            col += 1
        end
        for j in 2:T
            K[:, col] .= time .== j
            col += 1
        end
        K[:, col] .= z
        Q = Matrix(qr(K).Q)[:, 1:size(K, 2)]
        M = I - Q * Q'
        xt = M * x
        yt = M * y
        tau2 = dot(xt, xt)
        beta = dot(xt, yt) / tau2
        dof = n - size(K, 2) - 1
        sigma = sqrt(sum(abs2, yt .- beta .* xt) / dof)
        hK = vec(sum(abs2, Q; dims=2))
        lambda = 1 - dot(1 .- hK, abs2.(sd)) / tau2

        rep = eiv_adequacy(y, x, unit, time;
                           controls=z, sigma_nu=sd, pilot=:point)
        @test rep.statistic.beta_star ≈ beta atol=1e-11
        @test rep.statistic.sigma ≈ sigma atol=1e-11
        @test rep.statistic.lambda_hat ≈ lambda atol=1e-11

        s = 0.04
        reps = eiv_adequacy(y, x, unit, time;
                            controls=z, sigma_nu=s, pilot=:point)
        lambda_s = 1 - s^2 * (n - size(K, 2)) / tau2
        @test reps.statistic.lambda_hat ≈ lambda_s atol=1e-11
    end

    @testset "input validation and edge cases" begin
        n0 = 40
        uid = repeat(1:10, inner=4); tid = repeat(1:4, outer=10)
        x = [sin(0.8k) + 0.1 * uid[k] for k in 1:n0]
        y = [x[k] + 0.2 * cos(1.9k) for k in 1:n0]

        # exactly one noise source required
        @test_throws ArgumentError eiv_adequacy(y, x, uid, tid)
        @test_throws ArgumentError eiv_adequacy(y, x, uid, tid;
                                                sigma_nu=0.1, reliability=0.9)
        @test_throws ArgumentError eiv_adequacy(y, x, uid, tid; reliability=1.2)
        @test_throws ArgumentError eiv_adequacy(y, x, uid, tid; codelow=x)  # needs both
        @test_throws ArgumentError eiv_adequacy(y, x, uid, tid; sigma_nu=[0.1, 0.2])
        @test_throws ArgumentError eiv_adequacy(y, x, uid, tid; sigma_nu=-0.1)
        @test_throws ArgumentError eiv_adequacy(y, x, uid, tid;
                                                reliability=0.9, cluster=:bogus)
        @test_throws ArgumentError eiv_adequacy(y, x, uid, tid;
                                                reliability=0.9,
                                                reliability_lower=0.0)
        @test_throws ArgumentError eiv_adequacy(y, x, uid, tid;
                                                reliability=0.9,
                                                gamma_lambda=0.96)

        # lambda <= 0 (noise swamps signal): hard FLAG with exact size 1
        rephard = eiv_adequacy(y, x, uid, tid; sigma_nu=100.0)
        @test rephard.verdict === :FLAGGED
        @test rephard.implied_size == 1.0
        @test any(occursin("exceeds", n) for n in rephard.notes)

        # binary treatment: misclassification is nonclassical — warn
        xb = Float64.([(uid[k] > 5) && (tid[k] >= 3) for k in 1:n0])  # staggered-style dummy
        repb = eiv_adequacy(y, xb, uid, tid; reliability=0.9, pilot=:point)
        @test any(occursin("MISCLASSIFICATION", n) for n in repb.notes)

        # lambda = 1 (no noise): eta = 0, certified at any pilot
        rep1 = eiv_adequacy(y, x, uid, tid; reliability=1.0, pilot=:point)
        @test rep1.eta == 0.0
        @test rep1.verdict === :POINT_PASS
    end

    @testset "report rendering (spec §2.2 format)" begin
        rep = eiv_adequacy(; beta_star=0.06, sigma=0.33, tau_star2=127.8,
                           n=8930, d_K=221, reliability=0.898)
        out = sprint(show, MIME("text/plain"), rep)
        @test occursin("Measurement Error", out)
        @test occursin("Within reliability lambda_hat = 0.898", out)
        @test occursin("Threshold (delta=0.05) = 0.652", out)
        @test occursin("corrected beta0", out)
        @test occursin("Certified breakdown", out)
        @test occursin("VERDICT:", out)
    end

end
