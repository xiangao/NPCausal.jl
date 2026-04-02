# Getting Started with NPCausal.jl

`NPCausal.jl` is a modern, high-performance Julia implementation of the popular nonparametric causal inference methods developed by Edward Kennedy. 

By default, this package leverages **`MLJ.jl`** alongside the extremely fast gradient boosted trees package **`EvoTrees.jl`**. It also utilizes native Julia multithreading to perform the cross-fitting loop in parallel, virtually eliminating the bottleneck associated with ensembled nuisance parameter estimation.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/NPCausal.jl")
```

## 1. Average Treatment Effect (ATE)

The `ate()` function provides doubly robust estimation of the Average Treatment Effect.

```julia
using NPCausal
using DataFrames
using Random

# Generate dummy data
Random.seed!(42)
n = 10000
X = DataFrame(x1 = randn(n), x2 = randn(n))
a = rand([0, 1, 2], n) # Categorical treatment
y = X.x1 .+ X.x2 .* (a .== 1) .+ 2 .* (a .== 2) .+ randn(n)

# Estimate ATE using 5-fold cross-fitting
results = ate(y, a, X; nsplits=5)

# View Average Treatment Effects
println(results.means)

# View Contrasts (e.g., E[Y(1)] - E[Y(0)])
println(results.contrasts)
```

## 2. Average Treatment Effect on the Treated (ATT)

If you have a binary treatment variable and are specifically interested in the treatment effect for the treated subpopulation, use the `att()` function.

```julia
using NPCausal
using DataFrames
using Random

# Generate dummy data
Random.seed!(42)
n = 5000
X = DataFrame(x1 = randn(n), x2 = randn(n))
# Binary treatment (0 or 1)
a = rand([0, 1], n)
y = X.x1 .+ 3 .* X.x2 .* a .+ randn(n)

# Estimate ATT using 5-fold cross-fitting
results = att(y, a, X; nsplits=5)

# View Average Treatment Effect on the Treated
println(results.res)
```

## Performance Note

For optimal performance, start Julia with multiple threads (e.g., `julia -t auto`). `NPCausal.jl` will automatically distribute the cross-validation folds across available CPU threads, performing the nuisance estimation in parallel.
