# =============================================================================
# Module C — staggered-DiD / TWFE-heterogeneity adequacy (current Paper C)
# Reference cases pinned to the audit pipeline (Tables 4-5). Design statistics
# are deterministic; the covariance-aware pilot and its sizes depend on the wild
# bootstrap (fixed seed) and are checked with wider tolerance.
# =============================================================================

"Read a vendored adoption panel (uid, tid, ft, y; ft empty = never-treated)."
function read_panel(name)
    cols = read_simple_csv(joinpath(TESTDATA, name))
    ft = [isempty(s) ? missing : parse(Float64, s) for s in cols[:ft]]
    return (unit = parse.(Int, cols[:uid]), time = parse.(Int, cols[:tid]),
            ft = ft, y = parse.(Float64, cols[:y]))
end

@testset "Module C — TWFE heterogeneity (Paper C)" begin

    @testset "block design: Gamma = 0 exactly (Prop. prop-gamma0)" begin
        N, T, g = 10, 6, 4
        unit = repeat(1:N, inner=T); time = repeat(1:T, outer=N)
        ft = [u <= 4 ? g : missing for u in unit]
        rep = twfe_design(unit, time, ft)
        @test rep.statistic.Gamma < 1e-8
        @test rep.statistic.Gamma_cmb < 1e-8
        @test rep.statistic.Gamma_gt < 1e-8
        @test rep.statistic.neg_share == 0.0
        @test rep.verdict === :CERTIFIED
        @test any(occursin("block", n) for n in rep.notes)
    end

    @testset "three-cohort no-reservoir design (Gamma = 1.54, 11% negative)" begin
        N, T = 48, 12
        unit = repeat(1:N, inner=T); time = repeat(1:T, outer=N)
        ft = [u <= 16 ? 3 : (u <= 32 ? 7 : 11) for u in unit]
        rep = twfe_design(unit, time, ft)
        @test rep.statistic.Gamma ≈ 1.54 atol = 0.02
        @test rep.statistic.neg_share ≈ 0.11 atol = 0.01
        # restricted ladder is nested: coh, evt <= cmb <= gt = unrestricted
        st = rep.statistic
        @test st.Gamma_coh <= st.Gamma_cmb + 1e-9
        @test st.Gamma_evt <= st.Gamma_cmb + 1e-9
        @test st.Gamma_cmb <= st.Gamma_gt + 1e-9
        @test st.Gamma_gt ≈ st.Gamma atol = 1e-10
        @test rep.verdict === :INCONCLUSIVE
        @test rep.breakdown ≈ PD._eta_dagger(0.05, 0.05) / st.Gamma_gt rtol = 1e-10
    end

    @testset "castle design statistics (Table 4)" begin
        castle = read_panel("castle_panel.csv")
        rep = twfe_design(castle.unit, castle.time, castle.ft)
        st = rep.statistic
        @test rep.design.n == 550 && rep.design.N == 50 && rep.design.T == 11
        @test st.Gamma ≈ 0.2114 rtol = 1e-3
        @test st.Gamma_cmb ≈ 0.198 rtol = 5e-3
        @test st.Gamma_gt ≈ st.Gamma atol = 1e-10
        @test st.Gamma_evt ≈ 0.168 rtol = 5e-3
        @test st.Gamma_coh ≈ 0.142 rtol = 5e-3
        @test st.neg_share ≈ 0.0 atol = 1e-12
        @test st.N1 == 95
        @test st.n_w ≈ 34.7127 rtol = 1e-4
    end

    @testset "divorce design statistics (always-treated dropped; Table 4)" begin
        divorce = read_panel("divorce_panel.csv")
        rep = twfe_design(divorce.unit, divorce.time, divorce.ft)
        st = rep.statistic
        @test rep.design.N == 49                 # 51 - 2 always-treated
        @test rep.design.n == 1323
        @test st.Gamma ≈ 0.6433 rtol = 2e-3
        @test st.Gamma_cmb ≈ 0.562 rtol = 5e-3
        @test st.Gamma_gt ≈ st.Gamma atol = 1e-10
        @test st.Gamma_evt ≈ 0.468 rtol = 5e-3
        @test st.Gamma_coh ≈ 0.381 rtol = 5e-3
        @test st.neg_share ≈ 0.0115 atol = 2e-3
        @test st.N1 == 522
        @test any(occursin("always-treated", n) for n in rep.notes)
    end

    @testset "castle inference: projected-norm upper bound certifies" begin
        castle = read_panel("castle_panel.csv")
        rep = twfe_adequacy(castle.y, castle.unit, castle.time, castle.ft;
                            bootstrap=299, seed=20260715)
        st = rep.statistic
        @test st.beta ≈ 0.081812 rtol = 1e-3          # published ~8% homicide increase
        @test st.sigma ≈ 0.186992 rtol = 1e-3
        @test st.rho_ar1 ≈ 0.2264 rtol = 1e-2
        @test st.psi_hat ≈ 3.3721 rtol = 1e-3
        @test st.psi_direct ≈ 3.3721 rtol = 1e-3
        @test st.psi_ar1 ≈ 1.3358 rtol = 1e-3
        @test st.Gamma_cmb_CR ≈ 0.1079 rtol = 5e-3
        # covariance correction floors the near-zero envelope; the fitted
        # group-time direction is likewise essentially null.
        @test st.pilot_cmb ≈ 0.0 atol = 1e-6
        @test st.size_cmb ≈ 0.05 atol = 2e-3
        @test st.Gamma_gt ≈ st.Gamma atol = 1e-10
        @test st.pilot_gt ≈ 0.74787382 atol = 1e-7
        @test rep.eta ≈ 0.08610202 atol = 1e-7
        @test st.size_gt ≈ 0.05085 atol = 2e-3
        @test st.K_upper_gt < rep.threshold
        # HC3 is a boundary diagnostic here; it can land on either side of eta†
        # across Julia/OS combinations while still remaining near the threshold.
        @test st.K_upper_gt_hc3 ≈ rep.threshold atol = 0.02
        @test st.eta_directional ≈ -0.0134423 atol = 1e-5
        @test st.size_directional ≈ 0.0500207 atol = 1e-5
        @test st.directional_alignment ≈ -0.164229 atol = 1e-5
        @test st.sign_reversal_rms ≈ 0.386972 atol = 1e-5
        @test st.size_realized ≈ st.size_directional atol = 1e-12
        @test rep.verdict === :CERTIFIED
        @test st.boot !== nothing && st.boot.norm !== nothing
    end

    @testset "divorce point envelope is large but norm bound is inconclusive" begin
        divorce = read_panel("divorce_panel.csv")
        rep = twfe_adequacy(divorce.y, divorce.unit, divorce.time, divorce.ft;
                            bootstrap=299, seed=20260715)
        st = rep.statistic
        @test st.sigma ≈ 0.1961 rtol = 1e-3
        @test st.psi_hat ≈ 2.91718 rtol = 2e-3
        @test st.psi_direct ≈ 2.91718 rtol = 2e-3
        @test st.psi_ar1 ≈ 1.61958 rtol = 2e-3
        @test st.Gamma_cmb_CR ≈ 0.32925 rtol = 5e-3
        @test st.eta_directional ≈ 1.629204 atol = 1e-5
        @test st.size_directional ≈ 0.370579 atol = 1e-5
        @test st.directional_alignment ≈ 0.567753 atol = 1e-5
        @test st.sign_reversal_rms ≈ 0.0245369 atol = 1e-5
        @test st.pilot_cmb ≈ 5.29888 rtol = 1e-3
        @test st.size_cmb ≈ 0.414877 atol = 2e-3
        @test st.Gamma_gt ≈ st.Gamma atol = 1e-10
        @test st.pilot_gt ≈ 6.59941945 atol = 1e-7
        @test rep.eta ≈ 2.48546963 atol = 1e-7
        @test st.size_gt > 0.65
        @test st.K_lower_gt < rep.threshold < st.K_upper_gt
        @test st.size_coh > 0.10 && st.size_evt > 0.10 # every subspace exceeds the bound
        @test st.size_cmb >= st.size_coh - 1e-9        # nesting: combined dominates
        @test rep.verdict === :INCONCLUSIVE
        @test any(occursin("point pilot", n) for n in rep.notes)
        @test any(occursin("fixed-T", n) for n in rep.notes)
    end

    @testset "minimum-wage headline reproduces the JAE point diagnostics" begin
        mw = read_panel("minimum_wage_panel.csv")
        rep = twfe_adequacy(mw.y, mw.unit, mw.time, mw.ft;
                            controls=:never, bootstrap=19, seed=20260828)
        st = rep.statistic
        @test st.beta ≈ -0.03660863 atol = 1e-7
        @test st.Gamma ≈ 0.26964309 atol = 1e-7
        @test st.Gamma_cmb ≈ 0.25728013 atol = 1e-7
        @test st.Gamma_gt ≈ st.Gamma atol = 1e-10
        @test st.neg_share ≈ 0.0 atol = 1e-12
        @test st.psi_direct ≈ 1.38489247 atol = 1e-7
        @test st.pilot_cmb ≈ 6.46943999 atol = 1e-7
        @test st.size_cmb ≈ 0.29304460 atol = 1e-7
        @test st.pilot_gt ≈ 6.46015317 atol = 1e-7
        @test st.size_gt ≈ 0.31599274 atol = 1e-7
        @test st.att_target ≈ -0.05158099 atol = 1e-7
        @test st.K_lower_gt > rep.threshold
        @test st.eta_directional ≈ 1.37026379 atol = 1e-7
        @test st.size_directional ≈ 0.27812971 atol = 1e-7
        @test st.directional_alignment ≈ 0.92476056 atol = 1e-7
        @test st.sign_reversal_rms ≈ 0.13576699 atol = 1e-7
        @test rep.verdict === :FLAGGED
    end

    @testset "exchangeable psi identity (Thm. thm-cluster(a): psi = 1 - rho_c)" begin
        castle = read_panel("castle_panel.csv")
        uid, tid, N, T = PD._integer_codes(castle.unit, castle.time)
        D = PD._treatment_indicator(castle.time, castle.ft)
        Dt = PD._twoway_demean(D, uid, tid, N, T)
        for rho_c in (0.3, 0.5, 0.8)
            psi = PD._psi_parametric(Dt, uid, tid, rho_c; kind=:exchangeable)
            @test psi ≈ 1 - rho_c atol = 1e-9
        end
        @test PD._psi_parametric(Dt, uid, tid, 0.0; kind=:ar1) ≈ 1.0 atol = 1e-9
    end

    @testset "validation and rendering" begin
        unit = repeat(1:6, inner=4); time = repeat(1:4, outer=6)
        @test_throws ArgumentError twfe_design(unit, time, fill(missing, 24))
        badft = [k % 3 == 0 ? 2 : 3 for k in 1:24]
        @test_throws ArgumentError twfe_design(unit, time, badft)
        @test_throws ArgumentError twfe_design(unit, time,
                                               [u <= 3 ? 2.5 : missing for u in unit])

        castle = read_panel("castle_panel.csv")
        out = sprint(show, MIME("text/plain"),
                     twfe_adequacy(castle.y, castle.unit, castle.time, castle.ft; bootstrap=99))
        @test occursin("TWFE Heterogeneity", out)
        @test occursin("restricted ladder", out)
        @test occursin("Point worst-case envelopes", out)
        @test occursin("Boundary-robust group-time K interval", out)
        @test occursin("Directional plug-in", out)
        @test !occursin("Realized-profile", out)
        @test occursin("VERDICT: FORMALLY CERTIFIED", out)
        outd = sprint(show, MIME("text/plain"),
                      twfe_design(castle.unit, castle.time, castle.ft))
        @test occursin("negative-weight share", outd)
        @test_throws ArgumentError twfe_adequacy(castle.y, castle.unit, castle.time,
                                                 castle.ft; cluster=:bad)
        @test_throws ArgumentError twfe_adequacy(castle.y, castle.unit, castle.time,
                                                 castle.ft; controls=:bad)
        @test_throws ArgumentError twfe_adequacy(castle.y, castle.unit, castle.time,
                                                 castle.ft; psi=-1.0)
        @test_throws ArgumentError twfe_adequacy(castle.y, castle.unit, castle.time,
                                                 castle.ft; gamma=0.6)
    end

end
