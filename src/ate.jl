using DataFrames
using MLJ
using EvoTrees
using Random
using Statistics
using Base.Threads

"""
    ate(y::Vector, a::Vector, X::DataFrame; nsplits=5, model=EvoTreeRegressor())

Fast, multi-threaded Doubly Robust ATE estimator via Cross-Fitting.
"""
function ate(y::AbstractVector, a::AbstractVector, X::DataFrame; 
             nsplits=5, 
             mu_model = EvoTreeRegressor(nrounds=100, eta=0.1, max_depth=5),
             pi_model = EvoTreeClassifier(nrounds=100, eta=0.1, max_depth=5))
    
    n = length(y)
    avals = unique(a)
    sort!(avals)
    n_avals = length(avals)
    
    # 1. Create CV folds
    # Randomly shuffle and assign to folds
    s = shuffle(1:n)
    folds = [s[1 + floor(Int, (k-1)*n/nsplits) : floor(Int, k*n/nsplits)] for k in 1:nsplits]
    
    muhat = zeros(n, n_avals)
    pihat = zeros(n, n_avals)
    
    # 2. Cross-fitting (Parallelized across folds)
    @info "Starting cross-fitting across $nsplits folds with $(Threads.nthreads()) threads..."
    
    @threads for vfold in 1:nsplits
        test_idx = folds[vfold]
        train_idx = setdiff(1:n, test_idx)
        
        X_train = X[train_idx, :]
        X_test  = X[test_idx, :]
        y_train = y[train_idx]
        a_train = a[train_idx]
        
        for (i, a_val) in enumerate(avals)
            # --- Propensity Score P(A=a_val | X) ---
            # Fit binary classifier for every treatment level (including last)
            a_binary = categorical(a_train .== a_val)
            mach_pi = machine(pi_model, X_train, a_binary)
            fit!(mach_pi, verbosity=0)
            preds = predict(mach_pi, X_test)
            pihat[test_idx, i] = pdf.(preds, true)

            # --- Outcome Regression E[Y | A=a_val, X] ---
            # Filter training data to only those who received this treatment
            mask = a_train .== a_val
            X_train_a = X_train[mask, :]
            y_train_a = y_train[mask]

            mach_mu = machine(mu_model, X_train_a, y_train_a)
            fit!(mach_mu, verbosity=0)

            muhat[test_idx, i] = predict(mach_mu, X_test)
        end
    end
    
    # 3. Normalize propensities to sum to 1 per observation, then clamp
    pihat .= max.(pihat, 1e-6)
    pihat .= pihat ./ sum(pihat, dims=2)
    pihat .= clamp.(pihat, 0.01, 1.0 - 0.01)

    # 4. Compute Influence Functions and Estimates
    ifvals = zeros(n, n_avals)
    for (i, a_val) in enumerate(avals)
        indicator = (a .== a_val)
        # IF = I(A=a)/pi * (Y - mu) + mu
        ifvals[:, i] .= (indicator ./ pihat[:, i]) .* (y .- muhat[:, i]) .+ muhat[:, i]
    end
    
    est = mean(ifvals, dims=1)[1, :]
    se  = std(ifvals, dims=1)[1, :] ./ sqrt(n)
    ci_lower = est .- 1.96 .* se
    ci_upper = est .+ 1.96 .* se
    
    # Format Results
    res1 = DataFrame(
        parameter = ["E{Y($a_val)}" for a_val in avals],
        Estimate = est,
        StdError = se,
        CI_Lower = ci_lower,
        CI_Upper = ci_upper
    )
    
    # Compute Contrasts (ATE)
    res2 = DataFrame(parameter=String[], Estimate=Float64[], StdError=Float64[], CI_Lower=Float64[], CI_Upper=Float64[])
    for i in 1:n_avals
        for j in (i+1):n_avals
            diff_if = ifvals[:, j] .- ifvals[:, i]
            d_est = mean(diff_if)
            d_se = std(diff_if) / sqrt(n)
            push!(res2, (
                parameter = "E{Y($(avals[j]))} - E{Y($(avals[i]))}",
                Estimate = d_est,
                StdError = d_se,
                CI_Lower = d_est - 1.96 * d_se,
                CI_Upper = d_est + 1.96 * d_se
            ))
        end
    end
    
    return (means=res1, contrasts=res2, ifvals=ifvals, nuis=(pi=pihat, mu=muhat))
end
