module NPCausal

export ate, att, ctseff, ivlate, ivbds, ipsi, superlearner
export admg_estimate_causal, admg_nps_tmle_a, admg_nested_anipw_a
export admg_id_plugin_a, admg_estimate_id

include("ate.jl")
include("att.jl")
include("ctseff.jl")
include("ivlate.jl")
include("ivbds.jl")
include("ipsi.jl")
include("ensemble.jl")

function _requires_causalgraphs(name)
    error("`$name` requires CausalGraphs.jl. Install it with " *
          "`using Pkg; Pkg.add(url=\"https://github.com/xiangao/CausalGraphs.jl\")`, " *
          "then load both packages with `using NPCausal, CausalGraphs`.")
end

"""
    admg_estimate_causal(; a, data, graph, treatment, outcome, kwargs...)

Estimate a graph-identified ADMG causal effect. Load `CausalGraphs.jl` together
with `NPCausal.jl` to enable this optional extension.

The extension uses `CausalGraphs.identify` for routing. Backdoor/a-fixable
effects are estimated with `NPCausal.ate`; p-fixable effects use the NPS TMLE
implementation; nested-fixable effects use ANIPW; and general discrete ID
functionals use the finite-support ID plug-in with EIF confidence intervals.
"""
admg_estimate_causal(args...; kwargs...) = _requires_causalgraphs("admg_estimate_causal")

"""
    admg_nps_tmle_a(; a, data, graph, treatment, outcome, kwargs...)

Estimate `E[outcome(a)]` for a p-fixable/front-door/NPS ADMG effect. Requires
the optional `CausalGraphs.jl` extension.
"""
admg_nps_tmle_a(args...; kwargs...) = _requires_causalgraphs("admg_nps_tmle_a")

"""
    admg_nested_anipw_a(; a, data, graph, treatment, outcome, id_result, kwargs...)

Estimate `E[outcome(a)]` for a nested-fixable ADMG effect using ANIPW. Requires
the optional `CausalGraphs.jl` extension.
"""
admg_nested_anipw_a(args...; kwargs...) = _requires_causalgraphs("admg_nested_anipw_a")

"""
    admg_id_plugin_a(; a, data, graph, treatment, outcome, kwargs...)

Estimate `E[outcome(a)]` from a symbolic Pearl-Shpitser ID expression using a
finite-support plug-in estimator with EIF confidence intervals. Requires the
optional `CausalGraphs.jl` extension.
"""
admg_id_plugin_a(args...; kwargs...) = _requires_causalgraphs("admg_id_plugin_a")

"""
    admg_estimate_id(; a, data, graph, treatment, outcome, kwargs...)

Run the general ADMG ID route through `CausalGraphs.jl` and estimate the
resulting finite-support ID functional from `NPCausal.jl`.
"""
admg_estimate_id(args...; kwargs...) = _requires_causalgraphs("admg_estimate_id")

end # module NPCausal
