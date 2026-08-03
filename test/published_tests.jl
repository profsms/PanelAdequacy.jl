@testset "published Paper A capture values" begin
    fixtures = joinpath(@__DIR__, "fixtures")

    parse_ints(values) = parse.(Int, values)
    parse_floats(values) = parse.(Float64, values)

    match = read_simple_csv(joinpath(fixtures, "kss_match.csv"))
    match_unit = parse_ints(match[:unit])
    match_time = parse_ints(match[:time])
    match_x = parse_floats(match[:x])
    match_greedy = cycle_contrasts(match_x, match_unit, match_time;
                                   method=:greedy)
    match_designed = cycle_contrasts(match_x, match_unit, match_time;
                                     method=:sparse)
    @info "published capture lock" design="KSS match" greedy=match_greedy.kappa designed=match_designed.kappa
    @test match_greedy.kappa ≈ 0.2638214123104785 atol=1e-12
    @test match_designed.kappa ≈ 0.5110281257551292 atol=1e-12
    @test match_greedy.kappa >= 0.262 - 5e-4
    @test match_designed.kappa >= 0.509 - 5e-4

    wage = read_simple_csv(joinpath(fixtures, "kss_wage.csv"))
    wage_unit = parse_ints(wage[:worker])
    wage_firm = parse_ints(wage[:firm])
    wage_year = parse_ints(wage[:year])
    wage_x = parse_floats(wage[:x])
    support_data = read_simple_csv(joinpath(fixtures, "kss_wage_supports.csv"))
    wage_rows = [[parse(Int, support_data[name][i])
                  for name in (:early_1999, :early_2001,
                               :late_1999, :late_2001)]
                 for i in eachindex(support_data[:early_1999])]
    wage_weights = [Float64[1, -1, -1, 1] for _ in wage_rows]
    wage_system = contrast_system(wage_x,
        [wage_unit, wage_firm, wage_year], wage_rows, wage_weights)
    @info "published capture lock" design="KSS wage" designed=wage_system.kappa
    @test wage_system.kappa ≈ 0.6112378951948423 atol=1e-12

    fscore = load_dataset("f_score_panel")
    fscore_time = string.(fscore.country, ":", fscore.year)
    fscore_x = Float64.(fscore.fscore)
    fscore_greedy = cycle_contrasts(fscore_x, fscore.uid, fscore_time;
                                    method=:greedy)
    fscore_designed = cycle_contrasts(fscore_x, fscore.uid, fscore_time;
                                      method=:structured)
    @info "published capture lock" design="F-score" greedy=fscore_greedy.kappa designed=fscore_designed.kappa
    @test fscore_greedy.kappa ≈ 0.3741655924479094 atol=1e-12
    @test fscore_designed.kappa ≈ 0.827387880261079 atol=1e-12
    @test fscore_greedy.kappa >= 0.365 - 5e-4
    @test fscore_designed.kappa >= 0.795 - 5e-4

    dense = read_simple_csv(joinpath(fixtures, "calibrated_dense.csv"))
    dense_unit = parse_ints(dense[:unit])
    dense_time = dense[:time]
    dense_x = parse_floats(dense[:x])
    dense_greedy = cycle_contrasts(dense_x, dense_unit, dense_time;
                                   method=:greedy)
    dense_designed = cycle_contrasts(dense_x, dense_unit, dense_time;
                                     method=:structured)
    @info "published capture lock" design="calibrated dense" greedy=dense_greedy.kappa designed=dense_designed.kappa
    @test dense_greedy.kappa ≈ 0.4388570425010178 atol=1e-12
    @test dense_designed.kappa ≈ 0.834126282995861 atol=1e-12
    @test dense_greedy.kappa >= 0.251 - 5e-4
    @test dense_designed.kappa >= 0.828 - 5e-4

    grunfeld = load_dataset("grunfeld_panel")
    grunfeld_y = Float64.(grunfeld.invest)
    grunfeld_value = Float64.(grunfeld.value)
    grunfeld_capital = Float64.(grunfeld.capital)
    grunfeld_firm = grunfeld.firm
    grunfeld_year = Float64.(grunfeld.year)
    grunfeld_design = design_summary(grunfeld_firm, grunfeld_year;
        x=grunfeld_capital, controls=grunfeld_value)
    grunfeld_score = score_concentration(grunfeld_y, grunfeld_capital,
        grunfeld_firm, grunfeld_year; controls=grunfeld_value)
    grunfeld_system = cycle_contrasts(grunfeld_capital, grunfeld_firm,
        grunfeld_year; method=:structured, controls=grunfeld_value)
    @info "published capture lock" design="Grunfeld capital | value" designed=grunfeld_system.kappa
    @test something(grunfeld_design.lambda_n) ≈ 0.20649852072450153 atol=1e-12
    @test something(grunfeld_design.n_eff) ≈ 16.559732274829294 atol=1e-11
    @test grunfeld_score.lambda_score ≈ 0.738792811202239 atol=1e-12
    @test grunfeld_score.n_eff_score ≈ 1.8035450833677689 atol=1e-12
    @test length(grunfeld_system) == 32
    @test grunfeld_system.kappa ≈ 0.626959467924497 atol=1e-12
    @test grunfeld_system.max_share ≈ 0.3522385481988268 atol=1e-12
    for c in eachindex(grunfeld_system.rows)
        r = grunfeld_system.rows[c]
        q = grunfeld_system.signs[c] ./ sqrt(length(r))
        @test abs(dot(q, grunfeld_value[r])) <= 2e-12
        @test maximum(abs(sum(q[k] for k in eachindex(r)
                              if grunfeld_firm[r[k]] == f))
                      for f in unique(grunfeld_firm[r])) <= 2e-12
        @test maximum(abs(sum(q[k] for k in eachindex(r)
                              if grunfeld_year[r[k]] == t))
                      for t in unique(grunfeld_year[r])) <= 2e-12
    end

    for (x, unit, time, greedy, designed) in (
        (match_x, match_unit, match_time, match_greedy, match_designed),
        (fscore_x, fscore.uid, fscore_time, fscore_greedy, fscore_designed),
        (dense_x, dense_unit, dense_time, dense_greedy, dense_designed),
    )
        d = design_summary(unit, time; x=x)
        @test 0 < something(d.lambda_n) <= 1
        @test 1 <= something(d.n_eff) <= d.n
        @test 0 <= greedy.kappa <= designed.kappa <= 1 + 1e-12
    end

    # Canonical label ordering makes selected supports invariant to input rows.
    function label_supports(cs, unit, time)
        sort([join(sort([string(unit[e], ":", time[e]) for e in rows]), "|")
              for rows in cs.rows])
    end
    perm = reverse(eachindex(dense_x))
    permuted = cycle_contrasts(dense_x[perm], dense_unit[perm], dense_time[perm];
                               method=:structured)
    @test permuted.kappa ≈ dense_designed.kappa atol=1e-12
    @test label_supports(permuted, dense_unit[perm], dense_time[perm]) ==
          label_supports(dense_designed, dense_unit, dense_time)
end
