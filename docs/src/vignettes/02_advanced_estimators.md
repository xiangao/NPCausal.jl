# Advanced Estimators in NPCausal.jl

```@meta
CurrentModule = NPCausal
```

In addition to basic ATE and ATT functions, `NPCausal.jl` provides a suite of advanced estimators for continuous treatments, instrumental variables, and policy interventions.

## 1. Continuous Treatment Effects (`ctseff`)

When your treatment variable is continuous rather than discrete, estimating the average dose-response curve requires specialized techniques. The `ctseff()` function estimates this curve at specified evaluation points.

```julia
using NPCausal
using DataFrames
using Random

# Generate dummy data with a continuous treatment
Random.seed!(42)
n = 800
X = DataFrame(x1 = randn(n), x2 = randn(n))
a = randn(n) .+ X.x1 # Continuous treatment
y = a.^2 .+ X.x2 .+ randn(n)

# Estimate continuous effect curve across a grid of evaluation points
eval_pts = collect(-2.0:0.5:2.0)
results = ctseff(y, a, X, eval_pts)

# View curve estimates and confidence bands
println(results.res)
```

## 2. Instrumental Variables (`ivlate` and `ivbds`)

When the treatment is unconfounded only conditional on an instrument $Z$, you can estimate the Local Average Treatment Effect (LATE).

```julia
# Z is an instrumental variable, A is the treatment, Y is the outcome
results_late = ivlate(y, a, z, X; nsplits=5)

println(results_late.res)
```

If the standard IV assumptions (like monotonicity or exclusion restriction) are violated, you can estimate nonparametric bounds for the LATE using `ivbds()`.

```julia
results_bounds = ivbds(y, a, z, X; nsplits=5)

println(results_bounds.res)
```

## 3. Incremental Propensity Score Interventions (`ipsi`)

Instead of fixing treatment to a specific value, what if we just shifted the odds of receiving treatment by a factor $\delta$? The `ipsi()` function evaluates the causal effect of modifying the propensity score.

```julia
# Evaluate the effect of shifting the propensity score odds by factors of 0.5, 1.0, and 2.0
delta_values = [0.5, 1.0, 2.0]
results_ipsi = ipsi(y, a, X, delta_values; nsplits=5)

println(results_ipsi.res)
```
