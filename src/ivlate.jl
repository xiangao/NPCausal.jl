using DataFrames
using MLJ
using EvoTrees
using Random
using Statistics
using Base.Threads
using Distributions

expit(x) = exp(x) / (1 + exp(x))
logit(x) = log(x / (1 - x))

function _project_01(x1, y1)
    n = length(x1)
    x2 = zeros(n)
    y2 = zeros(n)
    for i in 1:n
        x = x1[i]
        y = y1[i]
        if y > 1 && 0 < x && x < 1
            y = 1
        elseif x < 0 && y > 1
            x = 0; y = 1
        elseif x < 0 && 0 < y && y < 1
            x = 0
        elseif -x > y && y < 0
            x = 0; y = 0
        elseif x > y && y > -x && y < 2 - x
            v = (x + y) / 2
            x = v; y = v
        elseif y > 2 - x && x > 1
            x = 1; y = 1
        end
        x2[i] = x; y2[i] = y
    end
    return x2, y2
end

"""
    ivlate(y::AbstractVector, a::AbstractVector, z::AbstractVector, X::DataFrame; kwargs...)

Estimating complier average effect of binary treatment using binary instrument.
"""
function ivlate(y::AbstractVector, a::AbstractVector, z::AbstractVector, X::DataFrame; 
             nsplits=5, 
             mu_model = EvoTreeRegressor(nrounds=100, eta=0.1, max_depth=5),
             pi_model = EvoTreeClassifier(nrounds=100, eta=0.1, max_depth=5),
             project01=true)
    
    n = length(y)

    s = shuffle(1:n)
    folds = [s[1 + floor(Int, (k-1)*n/nsplits) : floor(Int, k*n/nsplits)] for k in 1:nsplits]
    
    pihat = zeros(n)
    la1hat = zeros(n); la0hat = zeros(n)
    mu1hat = zeros(n); mu0hat = zeros(n)
    onesided = sum(a[z .== 0]) == 0
    
    @threads for vfold in 1:nsplits
        test_idx = folds[vfold]
        train_idx = setdiff(1:n, test_idx)
        
        X_train = X[train_idx, :]
        X_test  = X[test_idx, :]
        y_train = y[train_idx]
        a_train = a[train_idx]
        z_train = z[train_idx]
        
        # IV Propensity Score E[Z | X]
        z_binary = categorical(z_train .== 1)
        mach_pi = machine(pi_model, X_train, z_binary)
        MLJ.fit!(mach_pi, verbosity=0)
        pihat[test_idx] = pdf.(MLJ.predict(mach_pi, X_test), true)
        
        # Treatment Regression E[A | Z=1, X] and E[A | Z=0, X]
        mask_z1 = z_train .== 1
        mach_la1 = machine(mu_model, X_train[mask_z1, :], a_train[mask_z1])
        MLJ.fit!(mach_la1, verbosity=0)
        la1hat[test_idx] = MLJ.predict(mach_la1, X_test)
        
        if !onesided
            mask_z0 = z_train .== 0
            mach_la0 = machine(mu_model, X_train[mask_z0, :], a_train[mask_z0])
            MLJ.fit!(mach_la0, verbosity=0)
            la0hat[test_idx] = MLJ.predict(mach_la0, X_test)
        else
            la0hat[test_idx] .= 0.0
        end
        
        # Outcome Regression E[Y | Z=1, X] and E[Y | Z=0, X]
        mach_mu1 = machine(mu_model, X_train[mask_z1, :], y_train[mask_z1])
        MLJ.fit!(mach_mu1, verbosity=0)
        mu1hat[test_idx] = MLJ.predict(mach_mu1, X_test)
        
        mask_z0 = z_train .== 0
        mach_mu0 = machine(mu_model, X_train[mask_z0, :], y_train[mask_z0])
        MLJ.fit!(mach_mu0, verbosity=0)
        mu0hat[test_idx] = MLJ.predict(mach_mu0, X_test)
    end
    
    pihat = clamp.(pihat, 1e-4, 1.0 - 1e-4)
    
    if project01
        la0hat, la1hat = _project_01(la0hat, la1hat)
    end
    
    ifvals_out = z .* (y .- mu1hat) ./ pihat .- (1 .- z) .* (y .- mu0hat) ./ (1 .- pihat) .+ mu1hat .- mu0hat
    ifvals_trt = z .* (a .- la1hat) ./ pihat .- (1 .- z) .* (a .- la0hat) ./ (1 .- pihat) .+ la1hat .- la0hat
    ifvals_gam2 = 2 .* (la1hat .- la0hat) .* (z .* (a .- la1hat) ./ pihat .- (1 .- z) .* (a .- la0hat) ./ (1 .- pihat)) .+ (la1hat .- la0hat).^2
    
    psihat = mean(ifvals_out) / mean(ifvals_trt)
    ifvals = (ifvals_out .- psihat .* ifvals_trt) ./ mean(ifvals_trt)
    muhat = mean(ifvals_trt)
    xihat = mean(ifvals_gam2)
    
    muhat2 = muhat * (0.01 < muhat < 0.99) + 0.01 * (muhat <= 0.01) + 0.99 * (muhat >= 0.99)
    q = quantile(la1hat .- la0hat, 1 - muhat2)
    xihat2 = mean(ifvals_trt .* (la1hat .- la0hat .> q))
    sharp2 = (xihat2 - muhat^2) / (muhat - muhat^2)
    
    sharp2 = clamp(sharp2, 0.001, 0.999)
    
    ifvals_sharp2 = (ifvals_trt .* ((la1hat .- la0hat .> q) .+ q) .- xihat2 .- q .* ((la1hat .- la0hat .> q) .- muhat2)) ./ (muhat - muhat^2) .+ (2 * muhat * xihat2 - xihat2 - muhat^2) .* (ifvals_trt .- muhat) ./ ((muhat - muhat^2)^2)
    
    est = [psihat, muhat, sharp2]
    se = [std(ifvals), std(ifvals_trt), std(ifvals_sharp2)] ./ sqrt(n)
    ci_ll = est .- 1.96 .* se
    ci_ul = est .+ 1.96 .* se
    
    ci_ll[3] = expit(logit(sharp2) - 1.96 * std(ifvals_sharp2 ./ (sharp2 - sharp2^2)) / sqrt(n))
    ci_ul[3] = expit(logit(sharp2) + 1.96 * std(ifvals_sharp2 ./ (sharp2 - sharp2^2)) / sqrt(n))
    
    pval = 2 .* (1 .- cdf.(Normal(), abs.(est ./ se)))
    pval[2:3] .= NaN
    
    res = DataFrame(
        parameter = ["LATE", "Strength", "Sharpness"],
        Estimate = est,
        StdError = se,
        CI_Lower = ci_ll,
        CI_Upper = ci_ul,
        P_Value = pval
    )
    
    return (res=res, nuis=(pi=pihat, la1=la1hat, la0=la0hat, mu1=mu1hat, mu0=mu0hat), ifvals=ifvals)
end
