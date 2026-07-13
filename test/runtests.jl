using Test
using LinearAlgebra
using PanelAdequacy

# Reference data now ships INSIDE the package (julia/data/), so the harness is
# self-contained — it runs from the package repo alone, no sibling testdata/.
# Root ../../testdata remains the canonical source these are vendored from.
const TESTDATA = normpath(joinpath(@__DIR__, "..", "data"))

"Minimal CSV reader for the vendored reference files (plain numeric/ISO fields)."
function read_simple_csv(path)
    lines = readlines(path)
    header = Symbol.(split(lines[1], ','))
    cols = Dict(h => String[] for h in header)
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        vals = split(ln, ',')
        for (h, v) in zip(header, vals)
            push!(cols[h], v)
        end
    end
    return cols
end

"Dense two-way projection M = I - D(D'D)^- D' for verification on small panels."
function dense_demean(x, uid, tid, N, T)
    n = length(x)
    D = zeros(n, N + T)
    for k in 1:n
        D[k, uid[k]] = 1.0
        D[k, N + tid[k]] = 1.0
    end
    M = I - D * pinv(D' * D) * D'
    return M * x
end

include("leverage_tests.jl")
include("measurement_error_tests.jl")
include("staggered_tests.jl")

@testset "bundled datasets API" begin
    @test Set(datasets()) == Set(["vdem_gate1", "eiv_vdem_panel",
        "psid_wages_panel", "castle_panel", "divorce_panel", "f_score_panel"])
    @test isfile(datapath("castle_panel"))
    @test isfile(datapath("castle_panel.csv"))
    @test_throws ArgumentError datapath("nope")
    d = load_dataset("eiv_vdem_panel")
    @test haskey(d, :iso) && haskey(d, :v2x_polyarchy)
    @test eltype(d.iso) == String                      # non-numeric stays String
    @test Missing <: eltype(d.v2x_polyarchy)           # numeric-with-blanks -> Union{Missing,Float64}
    # the bundled dataset drives the diagnostic directly (article-reproducibility)
    keep = .!ismissing.(d.ly) .& .!ismissing.(d.v2x_polyarchy) .& .!ismissing.(d.v2x_polyarchy_sd)
    rep = eiv_adequacy(Float64.(d.ly[keep]), Float64.(d.v2x_polyarchy[keep]),
                       d.iso[keep], d.year[keep];
                       sigma_nu=Float64.(d.v2x_polyarchy_sd[keep]), pilot=:point)
    @test rep.statistic.lambda_hat ≈ 0.8984 atol = 1e-3

    # Paper A empirical application (Piotroski F-Score / Visegrad panel) drives
    # leverage_report directly from the bundled object, reproducing Table 2/3.
    fp = load_dataset("f_score_panel")
    @test length(fp.uid) == 217 && length(unique(fp.uid)) == 19
    @test eltype(fp.uid) == String && eltype(fp.country) == String
    fs = Float64.(fp.fscore); logr = log1p.(Float64.(fp.ret))
    rB = leverage_report(logr, fs, fp.uid, Float64.(fp.year))          # Spec B (log return)
    @test rB.design.n == 217 && rB.design.d_K == 33
    @test rB.statistic.beta ≈ -0.00147 atol = 5e-5                     # paper -0.00147
    @test rB.statistic.se_hc2 ≈ 0.0153 atol = 5e-4                     # LO SE
    @test rB.design.tau_star2 ≈ 559 atol = 1                          # pooled tau^2 ~ 559
    mE = fp.country .== "Poland"
    rE = leverage_report(logr[mE], fs[mE], fp.uid[mE], Float64.(fp.year[mE]))  # Spec E
    @test rE.design.n == 122
    @test abs(rE.statistic.beta / rE.statistic.se_hc2) ≈ 0.61 atol = 0.02      # headline |t|
end

@testset "PanelAdequacy shared infrastructure" begin

    @testset "union-find d_K" begin
        # balanced 5x4 panel: connected, d_K = N + T - 1
        uid = repeat(1:5, inner=4); tid = repeat(1:4, outer=5)
        d = design_summary(uid, tid)
        @test d.N == 5 && d.T == 4 && d.n == 20
        @test d.ncomponents == 1
        @test d.d_K == 5 + 4 - 1
        @test d.rho == d.d_K / 20

        # two disconnected blocks: units 1-3 x times 1-2, units 4-6 x times 3-4
        uid2 = vcat(repeat(1:3, inner=2), repeat(4:6, inner=2))
        tid2 = vcat(repeat(1:2, outer=3), repeat(3:4, outer=3))
        d2 = design_summary(uid2, tid2)
        @test d2.ncomponents == 2
        @test d2.d_K == 6 + 4 - 2

        # single unit, many periods
        d3 = design_summary(fill(1, 7), 1:7)
        @test d3.d_K == 1 + 7 - 1 && d3.ncomponents == 1

        # non-integer identifiers (strings/mixed) must work
        d4 = design_summary(["a", "a", "b", "b"], [2001, 2002, 2001, 2002])
        @test d4.N == 2 && d4.T == 2 && d4.d_K == 3
    end

    @testset "two-way demeaning vs dense projection" begin
        # unbalanced panel: drop scattered cells from a 8x6 grid
        rng_state = 20260710
        uid = Int[]; tid = Int[]
        for i in 1:8, t in 1:6
            (i + 2t) % 7 == 0 && continue   # deterministic unbalancedness
            push!(uid, i); push!(tid, t)
        end
        n = length(uid)
        @test n < 48                        # actually unbalanced
        x = [sin(0.7k + rng_state % 13) + 0.3k^2 / n for k in 1:n]

        xt = twoway_demean(x, uid, tid)
        xt_dense = dense_demean(x, uid, tid, 8, 6)
        @test maximum(abs.(xt .- xt_dense)) < 1e-8

        # idempotence: demeaning a demeaned vector is a no-op
        @test maximum(abs.(twoway_demean(xt, uid, tid) .- xt)) < 1e-8

        # cluster-centering identity (Paper C, Lemma 1): within-unit sums of the
        # two-way-within-transformed vector are zero, balanced or not
        for i in 1:8
            @test abs(sum(xt[uid .== i])) < 1e-8
        end
    end

    @testset "block-design within treatment is uniform on treated cells" begin
        # single adoption date g=4, treated units 1-4, never-treated 5-10, T=6
        N, T, g = 10, 6, 4
        uid = repeat(1:N, inner=T); tid = repeat(1:T, outer=N)
        D = Float64.([(i <= 4) && (t >= g) for i in 1:N for t in 1:T])
        Dt = twoway_demean(D, uid, tid)
        treated = D .== 1.0
        w = Dt[treated]
        # Gamma = 0 precondition (Paper C, Prop. 3): Dt constant on treated cells
        @test maximum(abs.(w .- w[1])) < 1e-8
    end

    @testset "reference case 1: V-Dem design primitive (spec §7.1)" begin
        path = joinpath(TESTDATA, "vdem_gate1.csv")
        @test isfile(path)
        cols = read_simple_csv(path)
        unit = cols[:iso]
        time = parse.(Float64, cols[:year])
        poly = parse.(Float64, cols[:poly])
        poly_sd = parse.(Float64, cols[:poly_sd])

        d = design_summary(unit, time; x=poly)
        @test d.n == 8704
        @test d.N == 174
        @test d.T == 60
        @test d.d_K == 233
        @test d.ncomponents == 1
        @test isapprox(d.rho, 0.0268; atol=5e-4)

        # Gate-1 headline reliability lambda_hat = 0.868 (spec §4 reference data)
        # — pure infrastructure arithmetic: lambda = 1 - mean(sd^2)(n-d_K)/tau*^2
        a_hat = sum(abs2, poly_sd) / d.n * (d.n - d.d_K)
        lambda_hat = 1 - a_hat / d.tau_star2
        @test isapprox(lambda_hat, 0.868; atol=1e-3)
    end

    @testset "AdequacyReport construction and display" begin
        d = design_summary(repeat(1:5, inner=4), repeat(1:4, outer=5))
        r = AdequacyReport(:measurement_error, d, (lambda_hat=0.868,),
                           0.28, 0.652, 0.82, 0.053, :CERTIFIED, 0.05, 0.05,
                           ["corrected, conservative pilot used"])
        out = sprint(show, MIME("text/plain"), r)
        @test occursin("Panel Adequacy Report — Measurement Error (Paper B)", out)
        @test occursin("lambda_hat = 0.868", out)
        @test occursin("|eta| = 0.280", out)
        @test occursin("Threshold (delta=0.05) = 0.652", out)
        @test occursin("Implied size of nominal 5% test: 5.3%", out)
        @test occursin("VERDICT: CERTIFIED at delta=0.05", out)
        @test occursin("Note: corrected, conservative pilot", out)

        # invalid pathology / verdict must be rejected
        @test_throws ArgumentError AdequacyReport(:conformal, d, NamedTuple(),
            nothing, nothing, nothing, nothing, :CERTIFIED, 0.05, 0.05, String[])
        @test_throws ArgumentError AdequacyReport(:leverage, d, NamedTuple(),
            nothing, nothing, nothing, nothing, :MAYBE, 0.05, 0.05, String[])

        # design-only report (Module C pre-outcome idiom) renders without eta
        r2 = AdequacyReport(:twfe_heterogeneity, d, (Gamma=0.21, neg_share=0.0),
                            nothing, nothing, nothing, nothing, :INCONCLUSIVE,
                            0.05, 0.05, String["design statistic only; no outcome supplied"])
        out2 = sprint(show, MIME("text/plain"), r2)
        @test occursin("Gamma = 0.210", out2)
        @test occursin("VERDICT: INCONCLUSIVE", out2)
    end

end
