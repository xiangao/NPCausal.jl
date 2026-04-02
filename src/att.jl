using DataFrames
using MLJ
using EvoTrees
using Random
using Statistics
using Base.Threads
using Distributions

"""
    att(y::AbstractVector, a::AbstractVector, X::DataFrame; nsplits=5, mu_model=EvoTreeRegressor(), pi_model=EvoTreeClassifier())

Estimating average effect of treatment on the treated.
"""
function att(y::AbstractVector, a::AbstractVector, X::DataFrame; 
             nsplits=5, 
             mu_model = EvoTreeRegressor(nrounds=100, eta=0.1, max_depth=5),
             pi_model = EvoTreeClassifier(nrounds=100, eta=0.1, max_depth=5))
    
    n = length(y)
    
    # 1. Create CV folds
    s = shuffle(1:n)
    folds = [s[1 + floor(Int, (k-1)*n/nsplits) : floor(Int, k*n/nsplits)] for k in 1:nsplits]
    
    pihat = zeros(n)
    mu0hat = zeros(n)
    
    # 2. Cross-fitting (Parallelized across folds)
    @threads for vfold in 1:nsplits
        test_idx = folds[vfold]
        train_idx = setdiff(1:n, test_idx)
        
        X_train = X[train_idx, :]
        X_test  = X[test_idx, :]
        y_train = y[train_idx]
        a_train = a[train_idx]
        
        # --- Propensity Score P(A=1 | X) ---
        a_binary = categorical(a_train .== 1)
        mach_pi = machine(pi_model, X_train, a_binary)
        fit!(mach_pi, verbosity=0)
        preds_pi = predict(mach_pi, X_test)
        pihat[test_idx] = pdf.(preds_pi, true)
        
        # --- Outcome Regression E[Y | A=0, X] ---
        mask0 = a_train .== 0
        X_train_0 = X_train[mask0, :]
        y_train_0 = y_train[mask0]
        
        mach_mu0 = machine(mu_model, X_train_0, y_train_0)
        fit!(mach_mu0, verbosity=0)
        mu0hat[test_idx] = predict(mach_mu0, X_test)
    end
    
    # Clip pihat to avoid division by zero or 1
    pihat = clamp.(pihat, 1e-4, 1.0 - 1e-4)
    
    # 3. Compute Influence Functions and Estimates
    mean_a = mean(a)
    mean_1_a = mean(1 .- a)
    mean_y = mean(y)
    
    ey01hat = mean((a ./ mean_a) .* mu0hat .+ ((1 .- a) ./ mean_1_a) .* (y .- mu0hat) .* pihat ./ (1 .- pihat))
    psihat = mean((a ./ mean_a) .* (y .- mu0hat) .- ((1 .- a) ./ mean_1_a) .* (y .- mu0hat) .* pihat ./ (1 .- pihat))
    
    ifvals1 = a .* (y .- mean_y) ./ mean_a
    ifvals2 = (a ./ mean_a) .* (mu0hat .- ey01hat) .+ ((1 .- a) ./ mean_1_a) .* (y .- mu0hat) .* pihat ./ (1 .- pihat)
    ifvals3 = (a ./ mean_a) .* (y .- mu0hat .- psihat) .- ((1 .- a) ./ mean_1_a) .* (y .- mu0hat) .* pihat ./ (1 .- pihat)
    
    ifvals = hcat(ifvals1, ifvals2, ifvals3)
    
    est = [mean(y[a .== 1]), ey01hat, psihat]
    se  = vec(std(ifvals, dims=1)) ./ sqrt(n)
    ci_lower = est .- 1.96 .* se
    ci_upper = est .+ 1.96 .* se
    pval = 2 .* (1 .- cdf.(Normal(), abs.(est ./ se)))
    
    res = DataFrame(
        parameter = ["E(Y|A=1)", "E{Y(0)|A=1}", "E{Y-Y(0)|A=1}"],
        Estimate = est,
        StdError = se,
        CI_Lower = ci_lower,
        CI_Upper = ci_upper,
        P_Value = pval
    )
    
    return (res=res, nuis=(pi=pihat, mu0=mu0hat), ifvals=ifvals)
end
