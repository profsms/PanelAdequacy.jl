module PanelAdequacyFixedEffectModelsExt

using PanelAdequacy
using FixedEffectModels

function _column(data, name::Symbol)
    hasproperty(data, name) || throw(ArgumentError(
        "estimation-sample table has no column :$name"))
    return getproperty(data, name)
end

function _vectors(model::FixedEffectModels.FixedEffectModel, data;
                  y::Symbol, x::Symbol, unit::Symbol, time::Symbol,
                  controls::AbstractVector{Symbol}=Symbol[])
    values = (_column(data, y), _column(data, x),
              _column(data, unit), _column(data, time))
    n = length(first(values))
    all(v -> length(v) == n, values) || throw(ArgumentError(
        "y, x, unit, and time columns must have equal length"))
    n == FixedEffectModels.nobs(model) || throw(ArgumentError(
        "the supplied table has $n rows but the fitted model used $(FixedEffectModels.nobs(model)); pass the exact estimation sample after missing-value filtering"))
    Z = isempty(controls) ? zeros(Float64, n, 0) :
        hcat((Float64.(_column(data, name)) for name in controls)...)
    return (values..., Z)
end

function PanelAdequacy.leverage_report(model::FixedEffectModels.FixedEffectModel,
                                       data; y::Symbol, x::Symbol,
                                       unit::Symbol, time::Symbol,
                                       controls::AbstractVector{Symbol}=Symbol[], kwargs...)
    yv, xv, uv, tv, Z = _vectors(model, data; y=y, x=x, unit=unit,
                                 time=time, controls=controls)
    return PanelAdequacy.leverage_report(yv, xv, uv, tv; controls=Z, kwargs...)
end

function PanelAdequacy.score_concentration(model::FixedEffectModels.FixedEffectModel,
                                           data; y::Symbol, x::Symbol,
                                           unit::Symbol, time::Symbol,
                                           controls::AbstractVector{Symbol}=Symbol[])
    yv, xv, uv, tv, Z = _vectors(model, data; y=y, x=x, unit=unit,
                                 time=time, controls=controls)
    return PanelAdequacy.score_concentration(yv, xv, uv, tv; controls=Z)
end

function PanelAdequacy.cycle_report(model::FixedEffectModels.FixedEffectModel,
                                    data; y::Symbol, x::Symbol,
                                    unit::Symbol, time::Symbol,
                                    controls::AbstractVector{Symbol}=Symbol[], kwargs...)
    yv, xv, uv, tv, Z = _vectors(model, data; y=y, x=x, unit=unit,
                                 time=time, controls=controls)
    return PanelAdequacy.cycle_report(yv, xv, uv, tv; controls=Z, kwargs...)
end

function PanelAdequacy.eiv_adequacy(model::FixedEffectModels.FixedEffectModel,
                                    data; y::Symbol, x::Symbol,
                                    unit::Symbol, time::Symbol,
                                    controls::AbstractVector{Symbol}=Symbol[], kwargs...)
    isempty(controls) || throw(ArgumentError(
        "the eiv_adequacy model adapter currently requires a single non-FE regressor"))
    yv, xv, uv, tv, _ = _vectors(model, data; y=y, x=x, unit=unit,
                                 time=time, controls=controls)
    return PanelAdequacy.eiv_adequacy(yv, xv, uv, tv; kwargs...)
end

function PanelAdequacy.adequacy_row(model::FixedEffectModels.FixedEffectModel,
                                   data; y::Symbol, x::Symbol,
                                   unit::Symbol, time::Symbol,
                                   controls::AbstractVector{Symbol}=Symbol[], kwargs...)
    yv, xv, uv, tv, Z = _vectors(model, data; y=y, x=x, unit=unit,
                                 time=time, controls=controls)
    return PanelAdequacy.adequacy_row(xv, uv, tv; y=yv, controls=Z, kwargs...)
end

end
