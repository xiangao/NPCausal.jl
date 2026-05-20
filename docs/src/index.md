# NPCausal.jl

`NPCausal.jl` is a Julia implementation of the nonparametric causal estimators I
use from `npcausal`. The package focuses on influence-function estimators with
cross-fitting.

## Included estimators

- `ate` for average treatment effects
- `att` for average treatment effects on the treated
- `ctseff` for continuous treatment effects
- `ivlate` and `ivbds` for instrumental-variable settings
- `ipsi` for incremental propensity score interventions
- optional ADMG estimators with `CausalGraphs.jl`: `admg_estimate_causal`,
  `admg_nps_tmle_a`, `admg_nested_anipw_a`, and `admg_estimate_id`

## ADMG Estimation with CausalGraphs.jl

When `NPCausal.jl` is loaded together with
[`CausalGraphs.jl`](https://github.com/xiangao/CausalGraphs.jl),
`admg_estimate_causal(...)` uses `CausalGraphs.identify` for routing and
estimates the supported target from the `NPCausal` namespace.

```julia
using Pkg
Pkg.add(url="https://github.com/xiangao/CausalGraphs.jl")

using NPCausal, CausalGraphs, DataFrames

res = NPCausal.admg_estimate_causal(
    a = [1, 0],
    data = data,
    graph = graph,
    treatment = :A,
    outcome = :Y,
)

r = x -> round(x, sigdigits=4)
(ACE = r(res[:TMLE].ACE),
 lower_ci = r(res[:TMLE].lower_ci),
 upper_ci = r(res[:TMLE].upper_ci))
```

The route covers backdoor/a-fixable effects through `ate`,
p-fixable/front-door effects through NPS TMLE, nested-fixable effects through
ANIPW, and finite-support discrete ID-algorithm functionals through
`:IDPlugin`.

```@docs
admg_estimate_causal
admg_nps_tmle_a
admg_nested_anipw_a
admg_id_plugin_a
admg_estimate_id
```

## Tutorials

- [Getting Started](vignettes/01_getting_started.md)
- [Advanced Estimators](vignettes/02_advanced_estimators.md)
