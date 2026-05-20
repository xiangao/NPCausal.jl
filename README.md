# NPCausal.jl

`NPCausal.jl` is a Julia implementation of the estimators I use from
`npcausal`: influence-function based causal estimators with cross-fitting.
When loaded together with [`CausalGraphs.jl`](https://github.com/xiangao/CausalGraphs.jl),
it can also estimate graph-identified ADMG effects after `CausalGraphs.jl`
handles identification.

## Why Julia?
The R package is useful, but repeated nuisance fitting can be slow when the
sample is large. This version keeps the same basic estimators but lets Julia do
the repeated work. By default it uses `MLJ.jl` and `EvoTrees.jl`, and the
cross-fitting loop can use Julia threads.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/xiangao/NPCausal.jl")
```

## Tutorials

Full documentation: **https://xiangao.github.io/NPCausal.jl/**

| Tutorial | Description |
|----------|-------------|
| [Getting Started](https://xiangao.github.io/NPCausal.jl/vignettes/01_getting_started/) | ATE and ATT estimation with cross-fitting |
| [Advanced Estimators](https://xiangao.github.io/NPCausal.jl/vignettes/02_advanced_estimators/) | Continuous treatment, IV, and policy intervention estimators |

## ADMG Estimation with CausalGraphs.jl

`NPCausal.jl` exposes optional ADMG estimators when `CausalGraphs.jl` is also
installed and loaded:

```julia
using Pkg
Pkg.add(url="https://github.com/xiangao/CausalGraphs.jl")

using NPCausal, CausalGraphs, DataFrames

graph = make_graph(
    vertices = [:A, :M, :Y],
    di_edges = [(:A, :M), (:M, :Y)],
    bi_edges = [(:A, :Y)],
)

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

The ADMG route supports backdoor/a-fixable effects via `ate`,
p-fixable/front-door effects via NPS TMLE, nested-fixable effects via ANIPW,
and finite-support discrete ID-algorithm functionals via `:IDPlugin`.

## Basic Usage

```julia
using NPCausal
using DataFrames

# Generate dummy data
n = 1000
X = DataFrame(x1 = randn(n), x2 = randn(n))
a = rand([0, 1, 2], n) # Categorical treatment
y = X.x1 .+ X.x2 .* (a .== 1) .+ 2 .* (a .== 2) .+ randn(n)

# Estimate ATE using 5-fold cross-fitting and EvoTrees
results = ate(y, a, X; nsplits=5)

# View Average Treatment Effects
println(results.means)

# View Contrasts (e.g., E[Y(1)] - E[Y(0)])
println(results.contrasts)
```
