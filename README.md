# NPCausal.jl

A blazing-fast, modern Julia implementation of `npcausal`.

This package provides nonparametric estimation of causal effects using the theory of influence functions, cross-fitting, and machine learning.

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
