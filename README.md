# NPCausal.jl

A blazing-fast, modern Julia implementation of `npcausal`.

This package provides nonparametric estimation of causal effects using the theory of influence functions, cross-fitting, and machine learning.
When loaded together with [`CausalGraphs.jl`](https://github.com/xiangao/CausalGraphs.jl),
it can also estimate graph-identified ADMG effects after `CausalGraphs.jl`
handles identification.

## Why Julia?
The original `npcausal` package in R relies on `SuperLearner` for cross-fitted estimation of nuisance parameters. While theoretically sound, `SuperLearner` running sequentially in R can be a major bottleneck on large datasets.

`NPCausal.jl` solves this by utilizing:
1. **`MLJ.jl` and `EvoTrees.jl`**: By default, it uses high-performance gradient boosting directly in Julia.
2. **Native Multithreading**: The cross-fitting process is parallelized using `Threads.@threads`, operating with zero memory copying and linear scaling across cores.

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
