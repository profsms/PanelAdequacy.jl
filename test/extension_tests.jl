using Test
using DataFrames
using GLM
using FixedEffectModels
using PanelAdequacy

unit = repeat(1:12, inner=6)
time = repeat(1:6, outer=12)
x = [sin(0.31 * i + 0.7 * t) + 0.03 * i * t for i in 1:12 for t in 1:6]
y = [0.2 * unit[k] - 0.1 * time[k] + 0.55 * x[k] + 0.08 * cos(k)
     for k in eachindex(x)]
z = [cos(0.23 * i - 0.4 * t) + 0.02 * i for i in 1:12 for t in 1:6]
df = DataFrame(y=y, x=x, z=z, unit=unit, time=time)
raw = leverage_report(y, x, unit, time)
raw_controlled = leverage_report(y, x, unit, time; controls=z)

@testset "GLM extension" begin
    model = GLM.lm(@formula(y ~ x), df)
    report = leverage_report(model, unit, time; coefficient=:x)
    @test report.statistic.beta ≈ raw.statistic.beta atol=1e-12
    @test report.statistic.lambda_n ≈ raw.statistic.lambda_n atol=1e-12
    @test score_concentration(model, unit, time; coefficient=:x) ==
          score_concentration(y, x, unit, time)
    controlled = GLM.lm(@formula(y ~ x + z), df)
    controlled_report = leverage_report(controlled, unit, time; coefficient=:x)
    @test controlled_report.statistic.beta ≈
          raw_controlled.statistic.beta atol=1e-12
    @test controlled_report.statistic.lambda_n ≈
          raw_controlled.statistic.lambda_n atol=1e-12
end

@testset "FixedEffectModels extension" begin
    model = FixedEffectModels.reg(df, @formula(y ~ x + fe(unit) + fe(time)))
    report = leverage_report(model, df; y=:y, x=:x, unit=:unit, time=:time)
    @test report.statistic.beta ≈ raw.statistic.beta atol=1e-12
    @test report.statistic.lambda_n ≈ raw.statistic.lambda_n atol=1e-12
    @test_throws ArgumentError leverage_report(model, df[1:end-1, :];
        y=:y, x=:x, unit=:unit, time=:time)
    controlled = FixedEffectModels.reg(
        df, @formula(y ~ x + z + fe(unit) + fe(time)))
    controlled_report = leverage_report(controlled, df; y=:y, x=:x,
        unit=:unit, time=:time, controls=[:z])
    @test controlled_report.statistic.beta ≈
          raw_controlled.statistic.beta atol=1e-12
    @test controlled_report.statistic.lambda_n ≈
          raw_controlled.statistic.lambda_n atol=1e-12
end
