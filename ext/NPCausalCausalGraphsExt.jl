module NPCausalCausalGraphsExt

using CausalGraphs
using DataFrames
using NPCausal
using Statistics

import NPCausal: admg_estimate_causal, admg_nps_tmle_a, admg_nested_anipw_a
import NPCausal: admg_id_plugin_a, admg_estimate_id

function _npc_levels(a)
    avals = collect(a isa AbstractVector ? a : [a])
    isempty(avals) && error("`a` must be a scalar or length-two vector.")
    length(avals) > 2 && error("Use scalar for E[Y(a)] or length-two for ACE.")
    avals
end

function _npc_index(avals, a)
    idx = findfirst(x -> x == a, avals)
    idx === nothing &&
        error("Treatment level `$a` was not observed; NPCausal cannot estimate E[Y($a)].")
    idx
end

function _npc_mean_result(raw, idx::Int)
    est = raw.means.Estimate[idx]
    contrib = raw.ifvals[:, idx]
    eif = contrib .- est
    (EYa = est,
     lower_ci = raw.means.CI_Lower[idx],
     upper_ci = raw.means.CI_Upper[idx],
     standard_error = raw.means.StdError[idx],
     EIF = eif)
end

function _combine_npc_ate(a, raw, observed_avals)
    avals = _npc_levels(a)
    idx1 = _npc_index(observed_avals, avals[1])
    out1 = _npc_mean_result(raw, idx1)
    if length(avals) == 1
        return Dict{Symbol,Any}(
            :NPCausalOnestep => out1,
            :NPCausalRaw => raw,
        )
    end

    idx0 = _npc_index(observed_avals, avals[2])
    out0 = _npc_mean_result(raw, idx0)
    contrib = raw.ifvals[:, idx1] .- raw.ifvals[:, idx0]
    ace = mean(contrib)
    se = std(contrib) / sqrt(length(contrib))
    eif = contrib .- ace
    Dict{Symbol,Any}(
        :NPCausalOnestep => (ACE = ace,
                             lower_ci = ace - 1.96 * se,
                             upper_ci = ace + 1.96 * se,
                             standard_error = se,
                             EIF = eif),
        :NPCausalOnestep_Y1 => out1,
        :NPCausalOnestep_Y0 => out0,
        :NPCausalRaw => raw,
    )
end

function admg_nps_tmle_a(; kwargs...)
    CausalGraphs.nps_tmle_a(; kwargs...)
end

function admg_nested_anipw_a(; kwargs...)
    CausalGraphs.nested_anipw_a(; kwargs...)
end

function admg_id_plugin_a(; kwargs...)
    CausalGraphs.id_plugin_a(; kwargs...)
end

function admg_estimate_id(; kwargs...)
    CausalGraphs.estimate_id(; kwargs...)
end

function _admg_combine_levels(a, primary::Symbol, call_one)
    avals = _npc_levels(a)
    out1 = call_one(avals[1])
    if length(avals) == 1
        x = getproperty(out1, primary)
        return Dict{Symbol,Any}(
            primary => (EYa=x.estimated_psi, lower_ci=x.lower_ci,
                        upper_ci=x.upper_ci, EIF=x.EIF),
            Symbol(primary, "_Ya") => x,
            :raw_Ya => out1,
        )
    end

    out0 = call_one(avals[2])
    x1 = getproperty(out1, primary)
    x0 = getproperty(out0, primary)
    ace = x1.estimated_psi - x0.estimated_psi
    eif = x1.EIF - x0.EIF
    se = sqrt(mean(eif .^ 2) / length(eif))
    Dict{Symbol,Any}(
        primary => (ACE=ace, lower_ci=ace - 1.96 * se,
                    upper_ci=ace + 1.96 * se, EIF=eif),
        Symbol(primary, "_Y1") => x1,
        Symbol(primary, "_Y0") => x0,
        :raw_Y1 => out1,
        :raw_Y0 => out0,
    )
end

function admg_estimate_causal(; a, data::DataFrame,
                                graph::Union{CausalGraphs.ADMG,Nothing}=nothing,
                                vertices=nothing,
                                di_edges=CausalGraphs.Edge[],
                                bi_edges=CausalGraphs.Edge[],
                                multivariate_variables=Dict{Symbol,Vector{Symbol}}(),
                                treatment, outcome,
                                sample_weights=nothing, kwargs...)
    g = CausalGraphs.with_graph(graph, vertices, di_edges, bi_edges, multivariate_variables)
    id = CausalGraphs.identify(g, treatment, outcome)

    if id.strategy == :not_identified
        error("The effect of $treatment on $outcome is not identified in this ADMG.")
    elseif id.strategy == :a_fixable
        sample_weights === nothing ||
            error("The NPCausal ATE bridge currently does not support `sample_weights`.")
        A = Symbol(treatment)
        Y = Symbol(outcome)
        predictors = CausalGraphs.replace_vector(
            CausalGraphs.markov_pillow(g, A; treatment=A),
            g.multivariate_variables,
        )
        all(p -> hasproperty(data, p), predictors) ||
            error("Data are missing at least one NPCausal adjustment variable.")
        X = isempty(predictors) ? DataFrame(_npc_intercept = ones(nrow(data))) : data[:, predictors]
        raw = NPCausal.ate(data[!, Y], data[!, A], X; kwargs...)
        observed_avals = sort(collect(unique(data[!, A])))
        return _combine_npc_ate(a, raw, observed_avals)
    elseif id.strategy == :p_fixable
        return _admg_combine_levels(a, :TMLE, aval ->
            admg_nps_tmle_a(a=aval, data=data, graph=g,
                            treatment=treatment, outcome=outcome,
                            sample_weights=sample_weights; kwargs...))
    elseif id.strategy == :nested_fixable
        return _admg_combine_levels(a, :ANIPW, aval ->
            admg_nested_anipw_a(a=aval, data=data, graph=g,
                                treatment=treatment, outcome=outcome,
                                id_result=id, sample_weights=sample_weights; kwargs...))
    else
        return CausalGraphs.estimate_id(a=a, data=data, graph=g,
                                        treatment=treatment, outcome=outcome,
                                        sample_weights=sample_weights; kwargs...)
    end
end

end
