# =============================================================================
# Module D — stepped-wedge design (Halkiewicz and Alexeev, arXiv:2610.35853).
# Reference values: test/fixtures/sw_reference.csv, a verbatim copy of the frozen
# Python output 2026-11-07_sw_package_reference.csv (swdesign.py full GLS/TWFE,
# Theorem 2 and Theorem 3 closed forms, Corollary 1, Haines actual cells).
# =============================================================================

const SWREF = read_simple_csv(joinpath(@__DIR__, "fixtures", "sw_reference.csv"))
const SWTOL = 1e-10
_num(s) = isempty(s) ? nothing : parse(Float64, s)
_des(s) = parse.(Float64, split(s, ';'))

function _sw_cov(i)
    est = Symbol(SWREF[:estimator][i])
    kw = Dict{Symbol,Any}(:estimator => est)
    if est === :GLS
        _num(SWREF[:phi][i]) !== nothing && (kw[:phi] = _num(SWREF[:phi][i]))
        if _num(SWREF[:icc][i]) !== nothing
            kw[:icc] = _num(SWREF[:icc][i]); kw[:cac] = _num(SWREF[:cac][i])
            _num(SWREF[:m][i]) !== nothing && (kw[:m] = _num(SWREF[:m][i]))
        end
    end
    return kw
end

@testset "Module D — stepped-wedge design (Paper 1)" begin

    t_of(T, q, phi) = isinf(phi) ? (2T - 1 - q * T) / ((T - 1) * (T + 1)) :
        ((1 - q) + (2T - 1 - q * T) * phi) / ((T - 1) * ((1 + q) + (T + 1) * phi))
    G2(T, t) = (T - 2) * ((T - 3) * (T + 1) * t^2 + 15) / (5T * (3 - (T + 1) * t)^2)

    @testset "Theorem 2 closed form, T = 3..10" begin
        for T in 3:10, q in (0.0, 0.1, 0.25, 0.5, 0.8), phi in (0.0, 0.5, 2.0, 10.0, Inf)
            d = vcat(fill((1 - q) / (T - 1), T - 1), q)
            kw = isinf(phi) ? (estimator=:TWFE,) : (phi=phi,)
            @test sw_gamma(; design=d, kw...)^2 ≈ G2(T, t_of(T, q, phi)) atol = SWTOL
            t = t_of(T, q, phi)
            W = sw_exposure_weights(d; kw...)
            @test maximum(abs.(W .- [6 * (T - e) * (1 - t * e) / (T * (T - 1) * (3 - (T + 1) * t)) for e in 1:T-1])) < SWTOL
        end
    end

    @testset "Theorem 1 (TWFE complete wedge), T = 3..10" begin
        for T in 3:10
            G2tw = (19T^3 - 31T^2 - 2T + 12) / (5T * (T - 2) * (T + 1))
            @test sw_gamma(; design=vcat(ones(T - 1), 0.0), estimator=:TWFE)^2 ≈ G2tw atol = SWTOL
        end
    end

    @testset "Theorem 3: ETI cost unimodal in q" begin
        for T in (4, 5, 7), phi in (0.5, 2.0, Inf)
            kw = isinf(phi) ? (estimator=:TWFE,) : (phi=phi,)
            v = [sw_eti_cost(vcat(fill((1 - q) / (T - 1), T - 1), q); kw...).eti_cost for q in 0:0.01:0.9]
            s = filter(!=(0), sign.(diff(v)))
            @test s[1] < 0
            @test count(s[2:end] .!= s[1:end-1]) <= 1
        end
    end

    @testset "reference values vs Python (1e-10)" begin
        n = length(SWREF[:case])
        kinds = SWREF[:kind]
        cases = unique(SWREF[:case][i] for i in 1:n if kinds[i] in ("balanced", "integer", "nested") &&
                                                     SWREF[:quantity][i] != "phi_prime")
        worst = 0.0
        for cid in cases
            idx = [i for i in 1:n if SWREF[:case][i] == cid]
            i0 = idx[1]
            d = _des(SWREF[:design][i0])
            kw = _sw_cov(i0)
            W = sw_exposure_weights(d; kw...)
            ec = sw_eti_cost(d; kw...)
            for i in idx
                q = SWREF[:quantity][i]; val = parse(Float64, SWREF[:value][i])
                got = q == "Gamma" ? sw_gamma(W) : q == "kappa" ? sw_radius(W) :
                      q == "it_ratio" ? ec.it_ratio : q == "eti_cost" ? ec.eti_cost :
                      q == "var_it" ? ec.var_it : q == "var_eti" ? ec.var_eti :
                      W[parse(Int, replace(q, "W_" => ""))]
                worst = max(worst, abs(got - val))
            end
        end
        @test worst < SWTOL
        for i in 1:n
            SWREF[:quantity][i] == "phi_prime" || continue
            @test sw_phi_nested(_num(SWREF[:icc][i]), _num(SWREF[:cac][i]), _num(SWREF[:m][i])) ≈
                  parse(Float64, SWREF[:value][i]) atol = SWTOL
        end
    end

    @testset "paper numbers and design search (T = 5, G = 24)" begin
        @test round(sw_radius(; design=[6, 6, 6, 6, 0], phi=2), digits=3) == 0.133
        @test round(sw_eti_cost([6, 6, 6, 6, 0]; phi=2).eti_cost, digits=2) == 2.27
        s = sw_design_search(24, 5; cap=6, max_wait=12, phi=2)
        for k in (0, 4, 6, 8, 12)
            i = findfirst(==("search_eti_phi2.0_cap6_wl$k"), SWREF[:case])
            row = s.frontier[findfirst(==(Float64(k)), s.frontier[:, 1]), :]
            @test row[7] ≈ parse(Float64, SWREF[:value][i]) atol = SWTOL
            @test row[2:6] == _des(SWREF[:design][i])
        end
        @test round.(s.frontier[[findfirst(==(Float64(k)), s.frontier[:, 1]) for k in (4, 6, 8, 12)], 7], digits=2) ==
              [1.52, 1.38, 1.29, 1.23]
        i = findfirst(==("search_radius_phi2_complete"), SWREF[:case])
        b = sw_design_search(24, 5; max_wait=0, phi=2, objective=:radius, max_it_ratio=1).best
        @test b[1, 8] ≈ parse(Float64, SWREF[:value][i]) atol = SWTOL
        @test b[1, 1:5] == _des(SWREF[:design][i])
    end

    @testset "Haines actual cells: kappa* = 0.1244 from the bundled data" begin
        h = load_dataset("haines2017_trial1")
        wards = sort(unique(Int.(h.ward)))
        start = Dict(w => Int(h.start[findfirst(==(w), Int.(h.ward))]) for w in wards)
        order = sort(wards, by = w -> (start[w], w))
        sizes = [count((Int.(h.ward) .== w) .& (Int.(h.period) .== t)) for w in order, t in 1:7]
        idx = [i for i in eachindex(SWREF[:case]) if SWREF[:case][i] == "haines_actual"]
        icc = _num(SWREF[:icc][idx[1]])
        W = sw_exposure_weights([2, 2, 2, 2, 2, 2, 0]; icc=icc, cac=1.0, sizes=sizes)
        for i in idx
            q = SWREF[:quantity][i]; val = parse(Float64, SWREF[:value][i])
            q == "kappa" && @test sw_radius(W) ≈ val atol = SWTOL
            startswith(q, "W_") && @test W[parse(Int, q[3:end])] ≈ val atol = SWTOL
        end
        @test round(sw_radius(W), digits=4) == 0.1244
        u = load_dataset("dots99_uganda")
        @test length(u.facility) == 2224 && length(unique(u.facility)) == 18
    end

    @testset "reductions and validation" begin
        sz = fill(20.0, 24, 5); d = [5, 5, 5, 5, 4]
        @test sw_exposure_weights(d; icc=0.05, cac=0.5, sizes=sz) ≈ sw_exposure_weights(d; icc=0.05, cac=0.5, m=20) atol = SWTOL
        @test sw_exposure_weights(d; icc=0.05, cac=0.5, m=20) ≈ sw_exposure_weights(d; phi=sw_phi_nested(0.05, 0.5, 20)) atol = SWTOL
        @test sw_eti_cost(d; estimator=:TWFE, sizes=sz).eti_cost ≈ sw_eti_cost(d; estimator=:TWFE).eti_cost atol = SWTOL
        @test_throws ArgumentError sw_exposure_weights([0, 6, 6, 6, 6]; phi=2)
        @test_throws ArgumentError sw_exposure_weights([6, 6]; phi=2)
        @test_throws ArgumentError sw_exposure_weights([6, 6, 6, 6, 0]; phi=2, icc=0.1, m=10)
        @test_throws ArgumentError sw_exposure_weights([6, 6, 6, 6, 0])
        @test_throws ArgumentError sw_eti_cost([24, 0, 0, 0, 0]; phi=2)
        @test_throws ArgumentError sw_gamma([0.5, 0.5]; target=[1.0, 1.0])
    end
end
