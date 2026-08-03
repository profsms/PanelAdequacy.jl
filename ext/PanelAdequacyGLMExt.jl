module PanelAdequacyGLMExt

using PanelAdequacy
using GLM
using StatsModels

const SupportedLinearModel = Union{
    GLM.LinearModel,
    StatsModels.TableRegressionModel{<:GLM.LinearModel},
}

function _inner_model(model)
    model isa GLM.LinearModel ? model : getproperty(model, :model)
end

function _assert_unweighted(model)
    inner = _inner_model(model)
    response = getproperty(inner, :rr)
    weights = if hasproperty(response, :weights)
        getproperty(response, :weights)
    elseif hasproperty(response, :wts)
        getproperty(response, :wts)
    else
        return
    end
    isempty(weights) && return
    all(isone, weights) || throw(ArgumentError(
        "GLM adapter supports only unweighted linear models; use the vector API for weighted designs"))
end

function _xy(model, coefficient)
    _assert_unweighted(model)
    y = Float64.(GLM.response(model))
    X = Matrix{Float64}(GLM.modelmatrix(model))
    names = String.(GLM.coefnames(model))
    length(names) == size(X, 2) || throw(ArgumentError(
        "could not align GLM coefficient names with its model matrix"))

    index = if coefficient isa Integer
        1 <= coefficient <= size(X, 2) || throw(ArgumentError(
            "coefficient index must lie in 1:$(size(X, 2))"))
        Int(coefficient)
    elseif coefficient isa Symbol || coefficient isa AbstractString
        matches = findall(==(String(coefficient)), names)
        length(matches) == 1 || throw(ArgumentError(
            "coefficient $(repr(coefficient)) does not identify exactly one GLM column; available names are $(join(names, ", "))"))
        only(matches)
    elseif coefficient === nothing
        candidates = findall(name -> lowercase(name) ∉ ("(intercept)", "intercept"), names)
        length(candidates) == 1 || throw(ArgumentError(
            "the GLM has $(length(candidates)) non-intercept columns; pass coefficient by name or index"))
        only(candidates)
    else
        throw(ArgumentError("coefficient must be nothing, an integer, a symbol, or a string"))
    end
    nuisance = [j for j in axes(X, 2)
                if j != index && lowercase(names[j]) ∉ ("(intercept)", "intercept")]
    controls = isempty(nuisance) ? zeros(Float64, length(y), 0) : X[:, nuisance]
    return y, X[:, index], controls
end

function PanelAdequacy.leverage_report(model::SupportedLinearModel,
                                       unit::AbstractVector,
                                       time::AbstractVector;
                                       coefficient=nothing, kwargs...)
    y, x, controls = _xy(model, coefficient)
    return PanelAdequacy.leverage_report(y, x, unit, time;
                                         controls=controls, kwargs...)
end

function PanelAdequacy.score_concentration(model::SupportedLinearModel,
                                           unit::AbstractVector,
                                           time::AbstractVector;
                                           coefficient=nothing)
    y, x, controls = _xy(model, coefficient)
    return PanelAdequacy.score_concentration(y, x, unit, time;
                                             controls=controls)
end

function PanelAdequacy.cycle_report(model::SupportedLinearModel,
                                    unit::AbstractVector,
                                    time::AbstractVector;
                                    coefficient=nothing, kwargs...)
    y, x, controls = _xy(model, coefficient)
    return PanelAdequacy.cycle_report(y, x, unit, time;
                                      controls=controls, kwargs...)
end

function PanelAdequacy.eiv_adequacy(model::SupportedLinearModel,
                                    unit::AbstractVector,
                                    time::AbstractVector;
                                    coefficient=nothing, kwargs...)
    y, x, controls = _xy(model, coefficient)
    size(controls, 2) == 0 || throw(ArgumentError(
        "the eiv_adequacy model adapter currently requires a single non-intercept regressor"))
    return PanelAdequacy.eiv_adequacy(y, x, unit, time; kwargs...)
end

function PanelAdequacy.adequacy_row(model::SupportedLinearModel,
                                   unit::AbstractVector,
                                   time::AbstractVector;
                                   coefficient=nothing, kwargs...)
    y, x, controls = _xy(model, coefficient)
    return PanelAdequacy.adequacy_row(x, unit, time; y=y,
                                     controls=controls, kwargs...)
end

end
