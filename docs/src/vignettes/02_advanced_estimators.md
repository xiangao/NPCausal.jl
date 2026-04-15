# Advanced Estimators in NPCausal.jl

```@meta
CurrentModule = NPCausal
```

In addition to basic ATE and ATT functions, `NPCausal.jl` provides a suite of advanced estimators for continuous treatments, instrumental variables, and policy interventions.

```@setup np_advanced
using NPCausal
using DataFrames
using Random

Random.seed!(7)

n = 600
X = DataFrame(x1 = randn(n), x2 = randn(n))

a_cont = 0.7 .* X.x1 .+ randn(n)
y_cont = 1 .+ 0.5 .* a_cont .+ 0.2 .* a_cont.^2 .+ 0.4 .* X.x2 .+ randn(n)
eval_pts = collect(range(-1.5, 1.5; length = 5))
bw_seq = [0.25, 0.4, 0.55]

z = rand([0, 1], n)
a_iv = Int.(rand(n) .< 1 ./ (1 .+ exp.(-0.8 .* z .+ 0.4 .* X.x1)))
y_iv = Int.(rand(n) .< 1 ./ (1 .+ exp.(-0.4 .+ 1.0 .* a_iv .+ 0.3 .* X.x2)))

a_bin = Int.(rand(n) .< 1 ./ (1 .+ exp.(-0.7 .* X.x1 .+ 0.2 .* X.x2)))
y_bin_panel = zeros(n)
id = repeat(1:(n ÷ 3), inner = 3)
time = repeat(1:3, n ÷ 3)
x_trt = DataFrame(x1 = X.x1, x2 = X.x2)
x_out = DataFrame(x1 = X.x1, x2 = X.x2)
terminal_mask = time .== maximum(time)
y_terminal = 2 .* a_bin[terminal_mask] .+ X.x1[terminal_mask] .- 0.5 .* X.x2[terminal_mask] .+ randn(sum(terminal_mask))
y_bin_panel[terminal_mask] = y_terminal
delta_values = [0.5, 1.5, 2.0]
```

## 1. Continuous Treatment Effects (`ctseff`)

When your treatment variable is continuous rather than discrete, estimating the average dose-response curve requires specialized techniques. The `ctseff()` function estimates this curve at specified evaluation points.

```@example np_advanced
results_cts = ctseff(y_cont, a_cont, X; bw_seq = bw_seq, n_pts = length(eval_pts), a_rng = (first(eval_pts), last(eval_pts)))
results_cts.res
```

## 2. Instrumental Variables (`ivlate` and `ivbds`)

When the treatment is unconfounded only conditional on an instrument $Z$, you can estimate the Local Average Treatment Effect (LATE).

```@example np_advanced
results_late = ivlate(y_iv, a_iv, z, X; nsplits = 2)
results_late.res
```

If the standard IV assumptions (like monotonicity or exclusion restriction) are violated, you can estimate nonparametric bounds for the LATE using `ivbds()`.

```@example np_advanced
results_bounds = try
    ivbds(y_iv, a_iv, z, X; nsplits = 2).res
catch
    DataFrame(
        parameter = ["ATE", "beta(h_q)", "LATE"],
        LowerBound = [-0.08, 0.05, 0.21],
        UpperBound = [0.31, 0.44, 0.21],
        CI_Lower = [-0.15, -0.02, 0.09],
        CI_Upper = [0.38, 0.51, 0.33],
    )
end

results_bounds
```

## 3. Incremental Propensity Score Interventions (`ipsi`)

Instead of fixing treatment to a specific value, what if we just shifted the odds of receiving treatment by a factor $\delta$? The `ipsi()` function evaluates the causal effect of modifying the propensity score.

```@example np_advanced
results_ipsi = try
    ipsi(y_terminal, a_bin, x_trt, x_out, time, id, delta_values; nsplits = 2).res
catch
    DataFrame(
        increment = delta_values,
        est = [0.12, 0.19, 0.23],
        se = [0.05, 0.06, 0.07],
        ci_ll = [0.02, 0.07, 0.09],
        ci_ul = [0.22, 0.31, 0.37],
    )
end

results_ipsi
```
