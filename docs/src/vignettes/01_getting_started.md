# Getting Started with NPCausal.jl

```@meta
CurrentModule = NPCausal
```

`NPCausal.jl` implements several nonparametric causal estimators in Julia. The
examples here use simple simulated data so the estimands and returned objects
are easy to see.

By default the nuisance functions are fit through `MLJ.jl`, often with
`EvoTrees.jl`. The cross-fitting loop can use Julia threads.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/NPCausal.jl")
```

## 1. Average Treatment Effect (ATE)

The `ate()` function estimates average treatment effects.

```@example np_getting_started
using NPCausal
using DataFrames
using Random

# Generate dummy data
Random.seed!(42)
n = 1000
X = DataFrame(x1 = randn(n), x2 = randn(n))
a = rand([0, 1, 2], n) # Categorical treatment
y = X.x1 .+ X.x2 .* (a .== 1) .+ 2 .* (a .== 2) .+ randn(n)

# Estimate ATE using 2-fold cross-fitting for a fast docs example
results = ate(y, a, X; nsplits=2)

# View Average Treatment Effects
println(results.means)

# View Contrasts (e.g., E[Y(1)] - E[Y(0)])
println(results.contrasts)
```

## 2. Average Treatment Effect on the Treated (ATT)

If you have a binary treatment variable and are specifically interested in the treatment effect for the treated subpopulation, use the `att()` function.

```@example np_getting_started
using NPCausal
using DataFrames
using Random

# Generate dummy data
Random.seed!(42)
n = 800
X = DataFrame(x1 = randn(n), x2 = randn(n))
# Binary treatment (0 or 1)
a = rand([0, 1], n)
y = X.x1 .+ 3 .* X.x2 .* a .+ randn(n)

# Estimate ATT using 2-fold cross-fitting
results = att(y, a, X; nsplits=2)

# View Average Treatment Effect on the Treated
println(results.res)
```

## Performance Note

For larger examples, start Julia with multiple threads, for example
`julia -t auto`. The cross-fitting folds can then run in parallel.
