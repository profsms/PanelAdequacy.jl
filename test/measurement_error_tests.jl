# =============================================================================
# Module B — measurement-error adequacy (Paper B: paper_b_fe_eiv_JoE.tex)
# Reference cases: spec §7.2 (V-Dem two-pole + gate-1 headline) and the PSID
# application (Paper B §7.2). Threshold machinery must be exact-inversion /
# quadratic — NEVER the discarded linear surrogate. Breakdown expectations are
# the FIXED-POINT values lambda† = t*/(t* + eta†) (paper Def. def-breakdown);
# cluster expectations are the by-unit CRVE psi_hat of Remark rem-cluster,
# matching Table tab-vdem / tab-psid of the paper.
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

@testset "Module B — measurement error (Paper B)" begin

    @testset "threshold constants (Paper B, Remark rem-exact-cv)" begin
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
    end

    @testset "reliability helpers" begin
        @test reliability_from_interval([0.1, 0.2], [0.3, 0.5]) ≈ [0.1, 0.15]
        # PSID pathway: sigma_nu = sqrt((1-r)/r) * within-sd
        @test reliability_from_ratio(0.8, 2.0) ≈ sqrt(0.25) * 2.0
        @test_throws ArgumentError reliability_from_ratio(1.2, 2.0)
    end

    @testset "reference case 2a: V-Dem two-pole (spec §7.2, eiv_vdem_results static rows)" begin
        cols = read_simple_csv(joinpath(TESTDATA, "eiv_vdem_panel.csv"))

        # --- aggregate polyarchy: CERTIFIED ---
        s = vdem_spec(cols, :v2x_polyarchy, :v2x_polyarchy_sd)
        rep = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd, pilot=:point)
        @test rep.design.n == 8930
        @test rep.design.d_K == 221
        st = rep.statistic
        @test st.lambda_hat ≈ 0.8984 atol = 1e-3
        @test st.beta_star ≈ 0.06096 rtol = 2e-3
        @test st.beta_corr ≈ 0.06785 rtol = 2e-3
        @test st.sigma ≈ 0.32956 rtol = 2e-3
        @test rep.design.tau_star2 ≈ 127.826 rtol = 2e-3
        @test rep.eta ≈ 0.23640 rtol = 5e-3
        @test rep.implied_size ≈ 0.05643 atol = 5e-4
        @test rep.threshold ≈ 0.652 atol = 5e-4
        @test rep.breakdown ≈ 0.762 atol = 2e-3     # fixed point t*/(t*+eta†); paper Table 3
        # point verdict is exactly equivalent to lambda_hat >= breakdown
        @test (rep.statistic.lambda_hat >= rep.breakdown) ==
              (rep.verdict === :CERTIFIED)
        @test rep.verdict === :CERTIFIED
        # formal (conservative) certificate also passes for polyarchy
        repc = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd)
        @test repc.verdict === :CERTIFIED
        @test repc.statistic.eta_upper > repc.eta
        # cluster-robust (country CRVE): paper Table 3 psi_hat = 19.2, still certified
        repcr = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd,
                             pilot=:point, cluster=:crve)
        @test repcr.statistic.psi_hat ≈ 19.18 rtol = 1e-2
        @test repcr.implied_size ≈ 0.050 atol = 1e-3
        @test repcr.verdict === :CERTIFIED

        # --- legislative constraints: FLAGGED ---
        s = vdem_spec(cols, :v2xlg_legcon, :v2xlg_legcon_sd)
        rep = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd, pilot=:point)
        @test rep.design.n == 8529
        @test rep.statistic.lambda_hat ≈ 0.5472 atol = 1e-3
        @test rep.eta ≈ 0.8853 rtol = 5e-3
        @test rep.implied_size ≈ 0.1435 atol = 1e-3
        @test rep.breakdown ≈ 0.621 atol = 2e-3     # fixed point; paper Table 3
        @test rep.verdict === :FLAGGED
        # the paper's middle case: flagged iid, CERTIFIED under country clustering
        repcr = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd,
                             pilot=:point, cluster=:crve)
        @test repcr.statistic.psi_hat ≈ 24.05 rtol = 1e-2
        @test repcr.implied_size ≈ 0.054 atol = 1e-3
        @test repcr.verdict === :CERTIFIED
        @test (repcr.statistic.lambda_hat >= repcr.breakdown) ==
              (repcr.verdict === :CERTIFIED)

        # THE naive-pilot danger (Prop. prop-pilot(i) / Design 3a, on real data):
        # the attenuated pilot CERTIFIES this genuinely-failing specification
        repn = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd, pilot=:naive)
        @test repn.eta ≈ 0.8853 * 0.5472 rtol = 1e-2   # eta understated by factor lambda
        @test repn.verdict === :CERTIFIED               # the exact error Paper B prevents
        @test any(occursin("ANTI-CONSERVATIVE", n) for n in repn.notes)

        # --- judicial constraints: FLAGGED decisively, exact size 1.00 ---
        s = vdem_spec(cols, :v2x_jucon, :v2x_jucon_sd)
        rep = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd, pilot=:point)
        @test rep.design.n == 8889
        @test rep.statistic.lambda_hat ≈ 0.4125 atol = 1e-3
        @test rep.eta ≈ 12.10 rtol = 1e-2
        @test rep.implied_size ≈ 1.0 atol = 1e-6
        @test rep.breakdown ≈ 0.929 atol = 2e-3     # fixed point; paper Table 3
        @test rep.verdict === :FLAGGED
        @test any(occursin("quadratic", n) for n in rep.notes)   # far-out honesty note
        # flag SURVIVES clustering: psi_hat = 25.2 but eta_CR = 2.4, size 67%
        repcr = eiv_adequacy(s.y, s.x, s.unit, s.time; sigma_nu=s.sd,
                             pilot=:point, cluster=:crve)
        @test repcr.statistic.psi_hat ≈ 25.21 rtol = 1e-2
        @test repcr.eta ≈ 2.41 rtol = 1e-2
        @test repcr.implied_size ≈ 0.674 atol = 3e-3
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

    @testset "PSID application via summary-form API (Paper B §app-psid)" begin
        # regression output recorded in results/eiv_psid_summary.csv
        bstar, sigma, tau2 = 0.7332110, 4.2465084, 83.1775587
        n, d_K = 4165, 595 + 7 - 1

        # self-consistent breakdown (fixed point; paper main text: 0.71)
        @test breakdown_reliability(bstar, sigma, tau2) ≈ 0.707 atol = 2e-3
        # cluster-robust breakdown at the paper's person-cluster psi = 2.33: 0.61
        @test breakdown_reliability(bstar, sigma, tau2; psi=2.330) ≈ 0.613 atol = 2e-3

        # within reliability 0.65 (Bound-Krueger first difference): FLAGGED
        rep = eiv_adequacy(; beta_star=bstar, sigma=sigma, tau_star2=tau2,
                           n=n, d_K=d_K, reliability=0.65, pilot=:point)
        @test rep.eta ≈ 0.848 atol = 2e-3            # paper: |eta| = 0.85
        @test rep.implied_size ≈ 0.1356 atol = 1e-3  # paper: 13.6%
        @test rep.verdict === :FLAGGED
        # ... but CERTIFIED under person clustering (paper Table 4: size 8.6%)
        repcr = eiv_adequacy(; beta_star=bstar, sigma=sigma, tau_star2=tau2,
                             n=n, d_K=d_K, reliability=0.65, pilot=:point,
                             psi=2.330)
        @test repcr.eta ≈ 0.556 atol = 2e-3
        @test repcr.implied_size ≈ 0.086 atol = 1e-3
        @test repcr.verdict === :CERTIFIED
        @test repcr.breakdown ≈ 0.613 atol = 2e-3

        # level reliability 0.82: point pass, formal certificate FAILS (paper ddagger note)
        repp = eiv_adequacy(; beta_star=bstar, sigma=sigma, tau_star2=tau2,
                            n=n, d_K=d_K, reliability=0.82, pilot=:point)
        @test repp.implied_size ≈ 0.0638 atol = 1e-3  # paper: 6.4%
        @test repp.verdict === :CERTIFIED
        repf = eiv_adequacy(; beta_star=bstar, sigma=sigma, tau_star2=tau2,
                            n=n, d_K=d_K, reliability=0.82)   # default: conservative
        @test repf.verdict === :FLAGGED
        @test repf.statistic.eta_upper ≈ 0.707 atol = 2e-3

        # summary-form report renders without N/T
        out = sprint(show, MIME("text/plain"), repf)
        @test occursin("n=4165", out) && occursin("d_K=601", out)
        @test !occursin("N=0", out)
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
        @test_throws ArgumentError eiv_adequacy(y, x, uid, tid;
                                                reliability=0.9, cluster=:bogus)

        # lambda <= 0 (noise swamps signal): hard FLAG with exact size 1
        rephard = eiv_adequacy(y, x, uid, tid; sigma_nu=100.0)
        @test rephard.verdict === :FLAGGED
        @test rephard.implied_size == 1.0
        @test any(occursin("exceeds", n) for n in rephard.notes)

        # binary treatment: misclassification is nonclassical — warn (Paper B §5.3)
        xb = Float64.([(uid[k] > 5) && (tid[k] >= 3) for k in 1:n0])  # staggered-style dummy
        repb = eiv_adequacy(y, xb, uid, tid; reliability=0.9, pilot=:point)
        @test any(occursin("MISCLASSIFICATION", n) for n in repb.notes)

        # lambda = 1 (no noise): eta = 0, certified at any pilot
        rep1 = eiv_adequacy(y, x, uid, tid; reliability=1.0, pilot=:point)
        @test rep1.eta == 0.0
        @test rep1.verdict === :CERTIFIED
    end

    @testset "report rendering (spec §2.2 format)" begin
        rep = eiv_adequacy(; beta_star=0.06, sigma=0.33, tau_star2=127.8,
                           n=8930, d_K=221, reliability=0.898)
        out = sprint(show, MIME("text/plain"), rep)
        @test occursin("Measurement Error (Paper B)", out)
        @test occursin("Within reliability lambda_hat = 0.898", out)
        @test occursin("Threshold (delta=0.05) = 0.652", out)
        @test occursin("corrected beta0", out)
        @test occursin("VERDICT:", out)
    end

end
