# =============================================================================
# Paper A — exact inference under concentrated identifying variation
# =============================================================================
using Test
using Random
using Statistics
using DelimitedFiles

@testset "Paper A — cycle inference" begin

    @testset "kappa = 1 on designs that are a single cycle" begin
        # one four-cycle: the contrast spans the whole cycle space
        @test cycle_capture([1.0, 3.0, 0.0, 2.0], [1, 2, 2, 1], [1, 1, 2, 2];
                            method=:greedy) ≈ 1.0 atol = 1e-12
        @test cycle_capture([1.0, 3.0, 0.0, 2.0], [1, 2, 2, 1], [1, 1, 2, 2];
                            method=:structured) ≈ 1.0 atol = 1e-12
        # one digon (a repeated cell)
        @test cycle_capture([2.0, 5.0], [1, 1], [1, 1]; method=:greedy) ≈ 1.0 atol = 1e-12
    end

    @testset "sparse (mobility-network) packing" begin
        # two-edge movers between firm pairs: the AKM geometry
        rng = MersenneTwister(21)
        nw, nf = 400, 20
        a = Int[]; b = Int[]
        for w in 1:nw
            f1 = rand(rng, 1:nf); f2 = rand(rng, 1:nf)
            append!(a, [w, w]); append!(b, [f1, f2])
        end
        x = randn(rng, length(a))
        cs = cycle_contrasts(x, a, b; method=:sparse)
        @test length(cs) > 0
        @test 0 < cs.kappa <= 1 + 1e-12
        # every contrast still annihilates both fixed effects exactly
        for c in 1:length(cs)
            du = Dict{Int,Float64}(); dt = Dict{Int,Float64}()
            for (k, e) in enumerate(cs.rows[c])
                du[a[e]] = get(du, a[e], 0.0) + cs.signs[c][k]
                dt[b[e]] = get(dt, b[e], 0.0) + cs.signs[c][k]
            end
            @test all(==(0.0), values(du))
            @test all(==(0.0), values(dt))
        end
        allr = vcat(cs.rows...)
        @test length(unique(allr)) == length(allr)
        # extreme pairing beats naive greedy on this geometry
        @test cs.kappa > cycle_capture(x, a, b; method=:greedy)
        @test cs.kappa === cycle_contrasts(x, a, b; method=:sparse).kappa

        # Every worker has exactly two observations here, so the sparse system
        # is compatible with worker-block symmetry and the API verifies it.
        comp = support_compatibility(cs, a)
        @test comp.compatible
        @test signflip_test(randn(rng, length(x)), cs; nflips=49,
                            rng=MersenneTwister(1), blocks=a).effective_C > 0
    end

    @testset "custom multiway contrasts and block compatibility" begin
        # A paired-stayer contrast that annihilates worker, firm and year FE.
        worker = [1, 1, 2, 2]
        firm = [10, 10, 20, 20]
        year = [2000, 2001, 2000, 2001]
        x = [0.0, 0.0, 0.0, 1.0]
        rows = [[1, 2, 3, 4]]
        weights = [[-1.0, 1.0, 1.0, -1.0]]
        cs = contrast_system(x, [worker, firm, year], rows, weights;
                             blocks=worker)
        @test cs.kappa ≈ 1.0 atol=1e-12
        @test support_compatibility(cs, worker).compatible
        @test maximum(abs, multiway_demean(x, worker, firm, year) .-
                           twoway_demean(x, worker, year)) < 1e-10

        # Ordinary four-cycles split multi-period worker blocks and therefore
        # cannot silently inherit worker-cluster exactness.
        N, T = 5, 4
        u = repeat(1:N, inner=T); t = repeat(1:T, outer=N)
        xx = sin.(0.7 .* collect(1.0:length(u))); yy = cos.(1.1 .* xx)
        cs2 = cycle_contrasts(xx, u, t)
        @test !support_compatibility(cs2, u).compatible
        @test_throws ArgumentError signflip_test(yy, cs2; blocks=u, nflips=49)
    end

    @testset "cluster diagnostics aggregate at the dependence-block level" begin
        rng = MersenneTwister(2026)
        N, T = 12, 2
        worker = repeat(1:N, inner=T)
        year = repeat(1:T, outer=N)
        x = randn(rng, N * T)
        y = 0.4 .* x .+ randn(rng, N * T)
        rep = cycle_report(y, x, worker, year; blocks=worker,
                           interval=false, nflips=49, rng=MersenneTwister(1))

        xt = twoway_demean(x, worker, year)
        block_mass = [sum(abs2, xt[worker .== g]) for g in 1:N]
        @test rep.statistic.concentration_level === :block
        @test rep.statistic.concentration_blocks == N
        @test rep.statistic.lambda_n ≈ maximum(block_mass) / sum(block_mass)

        yt = twoway_demean(y, worker, year)
        u = yt .- rep.statistic.beta_ols .* xt
        cluster_score = [sum((xt .* u)[worker .== g]) for g in 1:N]
        score_share = abs2.(cluster_score) ./ sum(abs2, cluster_score)
        @test rep.statistic.score_lambda_n ≈ maximum(score_share)
        @test rep.statistic.score_H_n ≈ sum(abs2, score_share)
    end

    @testset "only treatment-loaded supports determine orbit granularity" begin
        # The first digon has zero loading; it cannot create an effective sign.
        x = [1.0, 1.0, 0.0, 2.0]
        u = [1, 1, 2, 2]; t = [1, 1, 2, 2]
        cs = cycle_contrasts(x, u, t; method=:greedy)
        out = signflip_test([0.2, -0.1, 0.4, -0.3], cs;
                            nflips=99, rng=MersenneTwister(1))
        @test out.C == 2
        @test out.effective_C == 1
        @test out.full_enumeration_floor == 1.0
    end

    @testset "contrasts annihilate BOTH fixed effects exactly" begin
        rng = MersenneTwister(11)
        N, T = 8, 6
        unit = repeat(1:N, inner=T); time = repeat(1:T, outer=N)
        x = randn(rng, N * T)
        for method in (:greedy, :structured, :sparse)
            cs = cycle_contrasts(x, unit, time; method=method)
            @test length(cs) > 0
            for c in 1:length(cs)
                du = zeros(N); dt = zeros(T)
                for (k, e) in enumerate(cs.rows[c])
                    du[unit[e]] += cs.signs[c][k]
                    dt[time[e]] += cs.signs[c][k]
                end
                # exact, not approximate: this is what makes the test exact
                @test all(==(0.0), du)
                @test all(==(0.0), dt)
            end
            # supports are edge-disjoint (needed for sign-flip exchangeability)
            allr = vcat(cs.rows...)
            @test length(unique(allr)) == length(allr)
            # capture cannot exceed the total within variation
            @test 0 < cs.kappa <= 1 + 1e-12
        end
    end

    @testset "loadings are exact: v'x == v'xtilde" begin
        rng = MersenneTwister(3)
        N, T = 6, 5
        unit = repeat(1:N, inner=T); time = repeat(1:T, outer=N)
        x = randn(rng, N * T)
        cs = cycle_contrasts(x, unit, time)
        xt = twoway_demean(x, unit, time)
        for c in 1:length(cs)
            b_raw = sum(cs.signs[c][k] * x[e] for (k, e) in enumerate(cs.rows[c])) /
                    sqrt(length(cs.rows[c]))
            b_res = sum(cs.signs[c][k] * xt[e] for (k, e) in enumerate(cs.rows[c])) /
                    sqrt(length(cs.rows[c]))
            @test b_raw ≈ b_res atol = 1e-9
            @test cs.loadings[c] ≈ b_raw atol = 1e-12
        end
    end

    @testset "packing is deterministic (cross-language parity requirement)" begin
        rng = MersenneTwister(5)
        N, T = 7, 7
        unit = repeat(1:N, inner=T); time = repeat(1:T, outer=N)
        x = float.(rand(rng, 1:9, N * T))      # discrete: many exact ties
        k1 = cycle_capture(x, unit, time)
        k2 = cycle_capture(x, unit, time)
        @test k1 === k2
        @test cycle_capture(x, unit, time; method=:greedy) ===
              cycle_capture(x, unit, time; method=:greedy)
    end

    @testset "structured packing dominates naive greedy" begin
        rng = MersenneTwister(9)
        N, T = 10, 8
        unit = repeat(1:N, inner=T); time = repeat(1:T, outer=N)
        x = randn(rng, N * T)
        @test cycle_capture(x, unit, time; method=:structured) >
              cycle_capture(x, unit, time; method=:greedy)
    end

    @testset "real Spec-G panel: V_n and lambda_n match the article" begin
        path = joinpath(@__DIR__, "..", "data", "f_score_panel.csv")
        raw, hdr = readdlm(path, ','; header=true)
        cols = Dict(strip(String(h)) => i for (i, h) in enumerate(vec(hdr)))
        uid = String.(raw[:, cols["uid"]])
        cy = [string(raw[i, cols["country"]], ":", raw[i, cols["year"]])
              for i in axes(raw, 1)]
        fs = Float64.(raw[:, cols["fscore"]])
        cs = cycle_contrasts(fs, uid, cy)
        # V_n is a property of the design, not of the packing: exact match
        @test cs.V_n ≈ 461.693976 atol = 1e-5
        xt = twoway_demean(fs, uid, cy)
        @test maximum(abs2, xt) / cs.V_n ≈ 0.036 atol = 5e-4   # article: lambda_n = 0.036
        # kappa is a heuristic lower bound and tie-break dependent; the article's
        # run reports 0.795. Ours is deterministic and must be at least as good.
        @test cs.kappa >= 0.79
        @test cycle_capture(fs, uid, cy; method=:greedy) < cs.kappa
    end

    @testset "sign-flip test is EXACT under a non-Gaussian concentrated design" begin
        # This is the property the module exists for: the conventional t-test
        # fails badly here, the exact test does not.
        rng = MersenneTwister(7)
        N, T = 10, 4
        unit = repeat(1:N, inner=T); time = repeat(1:T, outer=N)
        n = N * T
        x = randn(rng, n); x[1] *= 12.0          # concentrate the variation
        reps, nflips = 1500, 299
        rej_c = 0; rej_t = 0
        for _ in 1:reps
            eps = [(rand(rng, Bool) ? 1.0 : -1.0) * randn(rng)^2 for _ in 1:n]
            eps[1] *= 8.0
            t = signflip_test(eps, x, unit, time; beta0=0.0, nflips=nflips, rng=rng)
            t.p <= 0.05 && (rej_c += 1)
            abs(leverage_report(eps, x, unit, time).statistic.t_hc2) > 1.96 && (rej_t += 1)
        end
        size_cycle = rej_c / reps
        size_t = rej_t / reps
        @test size_cycle <= 0.08              # exact: holds its nominal level
        @test size_t > 0.20                   # conventional: badly over-rejects
    end

    @testset "confidence set inverts the test, and is affine-exact" begin
        rng = MersenneTwister(2)
        N, T = 8, 5
        unit = repeat(1:N, inner=T); time = repeat(1:T, outer=N)
        n = N * T
        x = randn(rng, n)
        y = 0.7 .* x .+ randn(rng, n)
        itv = signflip_interval(y, x, unit, time; alpha=0.05, ngrid=2001,
                                nflips=999, rng=MersenneTwister(4))
        @test itv.lo < itv.beta_tilde < itv.hi     # centred on the contrast estimate
        # the p-value at beta_tilde is maximal (the statistic vanishes there)
        p_at = signflip_test(y, x, unit, time; beta0=itv.beta_tilde, nflips=999,
                             rng=MersenneTwister(4)).p
        @test p_at > 0.05
        # a value far outside the set is rejected
        p_far = signflip_test(y, x, unit, time; beta0=itv.hi + 10 * (itv.hi - itv.lo),
                              nflips=999, rng=MersenneTwister(4)).p
        @test p_far <= 0.05
    end

    @testset "report, verdicts and the 2^(1-C) floor" begin
        rng = MersenneTwister(13)
        N, T = 25, 10                              # big enough that lambda_n is small
        unit = repeat(1:N, inner=T); time = repeat(1:T, outer=N)
        n = N * T
        x = randn(rng, n); y = randn(rng, n)
        rep = cycle_report(y, x, unit, time; nflips=499, rng=MersenneTwister(1))
        @test rep.statistic.lambda_n < 0.10        # genuinely diffuse
        @test rep.verdict === :POINT_PASS
        @test rep.statistic.min_pvalue ≈ exp2(1 - rep.statistic.effective_C)
        @test rep.statistic.se_price ≈ 1 / sqrt(rep.statistic.kappa)
        @test isfinite(rep.statistic.score_lambda_n)
        @test rep.statistic.score_n_eff > 0
        @test occursin("Concentrated Identifying Variation", sprint(show,
              MIME"text/plain"(), rep))

        # concentrated design -> FLAGGED, and it says why
        xc = randn(rng, n); xc[1] *= 30.0
        repc = cycle_report(y, xc, unit, time; nflips=499, rng=MersenneTwister(1),
                            interval=false)
        @test repc.verdict === :FLAGGED
        @test any(occursin("CONCENTRATION WARNING", s) for s in repc.notes)

        # too few supports for any level-alpha test to exist
        rep2 = cycle_report([1.0, 3.0, 0.0, 2.0], [1.0, 3.0, 0.0, 2.0],
                            [1, 2, 2, 1], [1, 1, 2, 2]; nflips=99, interval=false)
        @test rep2.statistic.C == 1
        @test rep2.statistic.effective_C == 1
        @test rep2.statistic.min_pvalue == 1.0
        @test rep2.verdict === :INCONCLUSIVE       # 2^(1-1) = 1 > alpha
        @test occursin("may repair", rep2.statistic.reason)

        # Binary-treatment granularity is structural: disjoint supports imply
        # no more treatment-loaded contrasts than treated observations.
        unit_b = repeat(1:6, inner=2)
        time_b = ones(Int, 12)
        y_b = [sin(k) + 0.1 * cos(2k) for k in 1:12]
        x_low = repeat([1.0, 0.0], 6)
        x_low[7:end] .= 0.0                    # n1 = 3, same design as x_high
        x_high = repeat([1.0, 0.0], 6)         # n1 = 6

        pre_low = PanelAdequacy.applicable(x_low, unit_b, time_b)
        pre_high = PanelAdequacy.applicable(x_high, unit_b, time_b)
        @test !pre_low.ok
        @test occursin("n1 = 3 treated", pre_low.reason)
        @test occursin("Not repairable by repacking", pre_low.reason)
        @test pre_high.ok

        low = cycle_report(y_b, x_low, unit_b, time_b;
                           method=:greedy, interval=false)
        high = cycle_report(y_b, x_high, unit_b, time_b;
                            method=:greedy, interval=false)
        @test low.verdict === :INCONCLUSIVE
        @test low.statistic.n_treated == 3
        @test !low.statistic.binary_floor_ok
        @test occursin("floor 2^(1-3) = 0.25", low.statistic.reason)
        @test high.verdict !== :INCONCLUSIVE
        @test high.statistic.effective_C == 6
        @test high.statistic.binary_floor_ok

        # Complement coding gives the same structural conclusion: the tighter
        # cap is min(n1,n0), not the arbitrary label attached to one category.
        x_mostly = 1 .- x_low
        pre_mostly = PanelAdequacy.applicable(x_mostly, unit_b, time_b)
        @test !pre_mostly.ok
        @test occursin("n1 = 9 treated and n0 = 3", pre_mostly.reason)
        mostly = cycle_report(y_b, x_mostly, unit_b, time_b;
                              method=:greedy, interval=false)
        @test mostly.verdict === :INCONCLUSIVE
        @test mostly.statistic.effective_C == 3

        row = adequacy_row(x_high, unit_b, time_b; method=:greedy)
        @test row.n_treated == 6
        @test row.binary_floor_ok
        continuous = adequacy_row(collect(1.0:12.0), unit_b, time_b;
                                  method=:greedy)
        @test continuous.n_treated === nothing
        @test continuous.binary_floor_ok
    end
end
