# =============================================================================
# Module C — staggered-DiD / TWFE-heterogeneity adequacy (Paper C: paper_c_twfe_v1.tex)
# Reference cases: spec §7.3-§7.6, pinned to the verified audit pipeline
# (results/crgamma_audit_results.csv — the numbers in Paper C's Table tab-audit).
# =============================================================================

"Read a vendored adoption panel (uid, tid, ft, y; ft empty = never-treated)."
function read_panel(name)
    cols = read_simple_csv(joinpath(TESTDATA, name))
    ft = [isempty(s) ? missing : parse(Float64, s) for s in cols[:ft]]
    return (unit = parse.(Int, cols[:uid]), time = parse.(Int, cols[:tid]),
            ft = ft, y = parse.(Float64, cols[:y]))
end

@testset "Module C — TWFE heterogeneity (Paper C)" begin

    @testset "block design: Gamma = 0 exactly (Prop. prop-gamma0, spec §7.3)" begin
        N, T, g = 10, 6, 4
        unit = repeat(1:N, inner=T); time = repeat(1:T, outer=N)
        ft = [u <= 4 ? g : missing for u in unit]
        rep = twfe_design(unit, time, ft)
        @test rep.statistic.Gamma < 1e-8
        @test rep.statistic.neg_share == 0.0
        @test rep.verdict === :CERTIFIED
        @test any(occursin("block", n) for n in rep.notes)
    end

    @testset "three-cohort no-reservoir design (Paper C §sec-sim: Gamma = 1.54, 11% negative)" begin
        N, T = 48, 12
        unit = repeat(1:N, inner=T); time = repeat(1:T, outer=N)
        ft = [u <= 16 ? 3 : (u <= 32 ? 7 : 11) for u in unit]
        rep = twfe_design(unit, time, ft)
        @test rep.statistic.Gamma ≈ 1.54 atol = 0.02
        @test rep.statistic.neg_share ≈ 0.11 atol = 0.01
        @test rep.verdict === :INCONCLUSIVE          # design-only: needs c/sigma pilot
        # breakdown = (c/sigma)† = eta†/Gamma (Cor. cor-cv)
        @test rep.breakdown ≈ PD._eta_dagger(0.05, 0.05) / rep.statistic.Gamma rtol = 1e-10
    end

    @testset "reference case 3: design statistics (spec §7.3)" begin
        castle = read_panel("castle_panel.csv")
        rep = twfe_design(castle.unit, castle.time, castle.ft)
        @test rep.design.n == 550
        @test rep.design.N == 50 && rep.design.T == 11
        @test rep.statistic.Gamma ≈ 0.211415 rtol = 1e-4
        @test rep.statistic.neg_share ≈ 0.0 atol = 1e-12
        @test rep.statistic.N1 == 95
        @test rep.statistic.n_w ≈ 34.7127 rtol = 1e-4

        divorce = read_panel("divorce_panel.csv")
        repd = twfe_design(divorce.unit, divorce.time, divorce.ft)
        @test repd.design.n == 1377
        @test repd.statistic.Gamma ≈ 0.856801 rtol = 1e-4
        @test repd.statistic.neg_share ≈ 0.072917 atol = 1e-4
        @test repd.statistic.N1 == 576
    end

    @testset "reference cases 4-6: inference layer (audit pipeline values)" begin
        castle = read_panel("castle_panel.csv")
        divorce = read_panel("divorce_panel.csv")

        # --- castle, iid: CERTIFIED at size 5.0%; bare TWFE beta = 0.082 (spec §7.6) ---
        rep = twfe_adequacy(castle.y, castle.unit, castle.time, castle.ft; cluster=:iid)
        st = rep.statistic
        @test st.beta ≈ 0.081812 rtol = 1e-3          # matches published ~8%
        @test st.sigma ≈ 0.186992 rtol = 1e-3
        @test st.att_bar ≈ 0.109355 rtol = 1e-3
        @test st.cohort_sd_raw ≈ 0.053850 rtol = 1e-3
        @test rep.eta ≈ -0.014088 atol = 1e-4         # realized eta (iid)
        @test rep.implied_size ≈ 0.050023 atol = 1e-4
        @test st.eta_worst_raw ≈ 0.358708 rtol = 1e-3
        # castle shrinkage: cohort dispersion is ALL sampling noise -> shrunk c = 0
        @test st.cohort_sd_shrunk == 0.0
        @test st.eta_worst == 0.0
        @test rep.verdict === :CERTIFIED

        # --- castle, clustered (AR(1) route): stays certified, psi computed not assumed ---
        repc = twfe_adequacy(castle.y, castle.unit, castle.time, castle.ft)
        stc = repc.statistic
        @test stc.rho_ar1 ≈ 0.226384 rtol = 1e-3
        @test stc.psi_hat ≈ 1.335767 rtol = 1e-3
        @test stc.Gamma_CR ≈ 0.182924 rtol = 1e-3
        @test repc.eta ≈ -0.012190 atol = 1e-4
        @test repc.implied_size ≈ 0.050017 atol = 1e-4
        @test repc.verdict === :CERTIFIED
        # estimator-driven cross-check diverges here (3.37 vs 1.34): must be surfaced
        @test stc.psi_driven ≈ 3.372112 rtol = 1e-2
        @test any(occursin("cross-check", n) for n in repc.notes)

        # --- divorce, iid: FLAGGED, realized size 60% ---
        repd = twfe_adequacy(divorce.y, divorce.unit, divorce.time, divorce.ft; cluster=:iid)
        std_ = repd.statistic
        @test std_.sigma ≈ 0.198134 rtol = 1e-3
        @test std_.att_bar ≈ -0.077839 rtol = 1e-3
        @test std_.cohort_sd_raw ≈ 0.305575 rtol = 1e-3
        @test repd.eta ≈ 2.223177 rtol = 1e-3
        @test repd.implied_size ≈ 0.603821 atol = 1e-3
        @test std_.eta_worst_raw ≈ 12.405 rtol = 2e-3
        # divorce shrinkage: heterogeneity GENUINE (spec §7.5, paper-final panel)
        @test std_.cohort_sd_shrunk ≈ 0.30107 rtol = 2e-3
        @test std_.shrink_factor ≈ 0.985 atol = 2e-3
        @test repd.verdict === :FLAGGED

        # --- divorce, clustered: still FLAGGED at 42% (the claimable number) ---
        repdc = twfe_adequacy(divorce.y, divorce.unit, divorce.time, divorce.ft)
        stdc = repdc.statistic
        @test stdc.rho_ar1 ≈ 0.292743 rtol = 1e-3
        @test stdc.psi_hat ≈ 1.615212 rtol = 1e-3
        @test stdc.Gamma_CR ≈ 0.674163 rtol = 1e-3
        @test repdc.eta ≈ 1.749280 rtol = 1e-3
        @test repdc.implied_size ≈ 0.416671 atol = 1e-3
        @test repdc.verdict === :FLAGGED
        # psi > 1 realized here: iid alarm was an upper bound — note must say so,
        # without asserting a universal direction
        @test any(occursin("upper bound", n) for n in repdc.notes)
        @test any(occursin("API", n) || occursin("Paper C", n) for n in repdc.notes)
    end

    @testset "exchangeable psi identity (Thm. thm-cluster(a): psi = 1 - rho_c exactly)" begin
        castle = read_panel("castle_panel.csv")
        uid, tid, N, T = PD._integer_codes(castle.unit, castle.time)
        D = PD._treatment_indicator(castle.time, castle.ft)
        Dt = PD._twoway_demean(D, uid, tid, N, T)
        for rho_c in (0.3, 0.5, 0.8)
            psi = PD._psi_parametric(Dt, uid, tid, rho_c; kind=:exchangeable)
            @test psi ≈ 1 - rho_c atol = 1e-9        # d_i'1 = 0 makes this exact
        end
        @test PD._psi_parametric(Dt, uid, tid, 0.0; kind=:ar1) ≈ 1.0 atol = 1e-9
    end

    @testset "user-supplied cohort effects override the internal pilot" begin
        N, T = 30, 8
        unit = repeat(1:N, inner=T); time = repeat(1:T, outer=N)
        ft = [u <= 8 ? 3 : (u <= 16 ? 6 : missing) for u in unit]
        y = [0.4 * sin(1.1k) + 0.05 * unit[k] for k in eachindex(unit)]
        # constant supplied cohort effects -> zero dispersion, zero realized deviation
        rep = twfe_adequacy(y, unit, time, ft; cluster=:iid,
                            cohort_effects=[0.5, 0.5], cohort_ses=[0.1, 0.1])
        @test rep.statistic.cohort_sd_raw == 0.0
        @test rep.eta ≈ 0.0 atol = 1e-10
        @test rep.verdict === :CERTIFIED
        @test !any(occursin("order-of-magnitude", n) for n in rep.notes)
        # internal pilot fires the honesty note instead
        rep2 = twfe_adequacy(y, unit, time, ft; cluster=:iid)
        @test any(occursin("order-of-magnitude", n) for n in rep2.notes)
    end

    @testset "validation and rendering" begin
        unit = repeat(1:6, inner=4); time = repeat(1:4, outer=6)
        # never-treated everywhere -> no treated cells
        @test_throws ArgumentError twfe_design(unit, time, fill(missing, 24))
        # first_treat varying within unit
        badft = [k % 3 == 0 ? 2 : 3 for k in 1:24]
        @test_throws ArgumentError twfe_design(unit, time, badft)
        # adoption date not an observed period
        @test_throws ArgumentError twfe_design(unit, time,
                                               [u <= 3 ? 2.5 : missing for u in unit])

        castle = read_panel("castle_panel.csv")
        out = sprint(show, MIME("text/plain"),
                     twfe_adequacy(castle.y, castle.unit, castle.time, castle.ft))
        @test occursin("TWFE Heterogeneity (Paper C)", out)
        @test occursin("Gamma", out)
        @test occursin("VERDICT: CERTIFIED", out)
        outd = sprint(show, MIME("text/plain"),
                      twfe_design(castle.unit, castle.time, castle.ft))
        @test occursin("negative-weight share", outd)
    end

end
