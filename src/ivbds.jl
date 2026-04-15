using DataFrames
using MLJ
using EvoTrees
using Random
using Statistics
using Base.Threads
using Distributions
using Roots

function _project_01_ivbds(x1, y1)
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
    ivbds(y::AbstractVector, a::AbstractVector, z::AbstractVector, X::DataFrame; kwargs...)

Estimating bounds on treatment effects with instrumental variables.
"""
function ivbds(y::AbstractVector, a::AbstractVector, z::AbstractVector, X::DataFrame; 
             nsplits=5, 
             mu_model = EvoTreeRegressor(nrounds=100, eta=0.1, max_depth=5),
             pi_model = EvoTreeClassifier(nrounds=100, eta=0.1, max_depth=5),
             project01=true)
    
    n = length(y)
    s = shuffle(1:n)
    folds = [s[1 + floor(Int, (k-1)*n/nsplits) : floor(Int, k*n/nsplits)] for k in 1:nsplits]
    
    pihat = zeros(n)
    la1hat = zeros(n); la0hat = zeros(n)
    vl1hat = zeros(n); vl0hat = zeros(n)
    vu1hat = zeros(n); vu0hat = zeros(n)
    mu1hat = zeros(n); mu0hat = zeros(n)
    onesided = sum(a[z .== 0]) == 0
    
    vu1 = y .* a .+ 1 .- a
    vu0 = y .* (1 .- a)
    vl1 = y .* a
    vl0 = y .* (1 .- a) .+ a
    
    @threads for vfold in 1:nsplits
        test_idx = folds[vfold]
        train_idx = setdiff(1:n, test_idx)
        
        X_train = X[train_idx, :]
        X_test  = X[test_idx, :]
        y_train = y[train_idx]
        a_train = a[train_idx]
        z_train = z[train_idx]
        
        z_binary = categorical(z_train .== 1)
        mach_pi = machine(pi_model, X_train, z_binary)
        MLJ.fit!(mach_pi, verbosity=0)
        pihat[test_idx] = pdf.(MLJ.predict(mach_pi, X_test), true)
        
        train_z1 = z_train .== 1
        train_z0 = z_train .== 0
        
        mach_la1 = machine(mu_model, X_train[train_z1, :], a_train[train_z1])
        MLJ.fit!(mach_la1, verbosity=0)
        la1hat[test_idx] = MLJ.predict(mach_la1, X_test)
        
        if !onesided
            mach_la0 = machine(mu_model, X_train[train_z0, :], a_train[train_z0])
            MLJ.fit!(mach_la0, verbosity=0)
            la0hat[test_idx] = MLJ.predict(mach_la0, X_test)
        else
            la0hat[test_idx] .= 0.0
        end
        
        mach_mu1 = machine(mu_model, X_train[train_z1, :], y_train[train_z1])
        MLJ.fit!(mach_mu1, verbosity=0)
        mu1hat[test_idx] = MLJ.predict(mach_mu1, X_test)
        
        mach_mu0 = machine(mu_model, X_train[train_z0, :], y_train[train_z0])
        MLJ.fit!(mach_mu0, verbosity=0)
        mu0hat[test_idx] = MLJ.predict(mach_mu0, X_test)
        
        mach_vl1 = machine(mu_model, X_train[train_z1, :], vl1[train_idx][train_z1])
        MLJ.fit!(mach_vl1, verbosity=0)
        vl1hat[test_idx] = MLJ.predict(mach_vl1, X_test)
        
        mach_vl0 = machine(mu_model, X_train[train_z0, :], vl0[train_idx][train_z0])
        MLJ.fit!(mach_vl0, verbosity=0)
        vl0hat[test_idx] = MLJ.predict(mach_vl0, X_test)
        
        mach_vu1 = machine(mu_model, X_train[train_z1, :], vu1[train_idx][train_z1])
        MLJ.fit!(mach_vu1, verbosity=0)
        vu1hat[test_idx] = MLJ.predict(mach_vu1, X_test)
        
        mach_vu0 = machine(mu_model, X_train[train_z0, :], vu0[train_idx][train_z0])
        MLJ.fit!(mach_vu0, verbosity=0)
        vu0hat[test_idx] = MLJ.predict(mach_vu0, X_test)
    end
    
    pihat = clamp.(pihat, 1e-4, 1.0 - 1e-4)
    
    if project01
        la0hat, la1hat = _project_01_ivbds(la0hat, la1hat)
    end
    
    ifvals_atel = z .* (vl1 .- vl1hat) ./ pihat .- (1 .- z) .* (vl0 .- vl0hat) ./ (1 .- pihat) .+ vl1hat .- vl0hat
    ifvals_ateu = z .* (vu1 .- vu1hat) ./ pihat .- (1 .- z) .* (vu0 .- vu0hat) ./ (1 .- pihat) .+ vu1hat .- vu0hat
    ifvals_trt = z .* (a .- la1hat) ./ pihat .- (1 .- z) .* (a .- la0hat) ./ (1 .- pihat) .+ la1hat .- la0hat
    ifvals_out = z .* (y .- mu1hat) ./ pihat .- (1 .- z) .* (y .- mu0hat) ./ (1 .- pihat) .+ mu1hat .- mu0hat
    
    muhat = mean(ifvals_trt)
    muhat2 = muhat * (0.01 < muhat < 0.99) + 0.01 * (muhat <= 0.01) + 0.99 * (muhat >= 0.99)
    q = quantile(la1hat .- la0hat, 1 - muhat2)
    hq = Float64.(la1hat .- la0hat .> q)
    
    ifvals_hql_num = (z .* (vl1 .- vl1hat) ./ pihat .- (1 .- z) .* (vl0 .- vl0hat) ./ (1 .- pihat) .+ vl1hat .- vl0hat) .* hq
    ifvals_hqu_num = (z .* (vu1 .- vu1hat) ./ pihat .- (1 .- z) .* (vu0 .- vu0hat) ./ (1 .- pihat) .+ vu1hat .- vu0hat) .* hq
    
    latehat = mean(ifvals_out) / mean(ifvals_trt)
    bl_ate = mean(ifvals_atel)
    bu_ate = mean(ifvals_ateu)
    bl_hq = mean(ifvals_hql_num) / mean(ifvals_trt)
    bu_hq = mean(ifvals_hqu_num) / mean(ifvals_trt)
    
    bl_ate = clamp(bl_ate, -1.0, 1.0)
    bu_ate = clamp(bu_ate, -1.0, 1.0)
    bl_hq = clamp(bl_hq, -1.0, 1.0)
    bu_hq = clamp(bu_hq, -1.0, 1.0)
    
    ifvals_late = (ifvals_out .- latehat .* ifvals_trt) ./ mean(ifvals_trt)
    ifvals_hql = (ifvals_hql_num .- bl_hq .* ifvals_trt) ./ mean(ifvals_trt)
    ifvals_hqu = (ifvals_hqu_num .- bu_hq .* ifvals_trt) ./ mean(ifvals_trt)
    
    lb = [bl_ate, bl_hq]
    ub = [bu_ate, bu_hq]
    se_l = [std(ifvals_atel), std(ifvals_hql)] ./ sqrt(n)
    se_u = [std(ifvals_ateu), std(ifvals_hqu)] ./ sqrt(n)
    
    cval = zeros(2)
    for j in 1:2
        crit(cn) = abs(cdf(Normal(), cn + (ub[j] - lb[j]) / max(se_l[j], se_u[j])) - cdf(Normal(), -cn) - 0.95)
        cval[j] = find_zero(crit, (1.0, 3.0))
    end
    
    ci_ll = lb .- cval .* se_l
    ci_ul = ub .+ cval .* se_u
    
    push!(lb, latehat)
    push!(ub, latehat)
    push!(ci_ll, latehat - 1.96 * std(ifvals_late) / sqrt(n))
    push!(ci_ul, latehat + 1.96 * std(ifvals_late) / sqrt(n))
    
    res = DataFrame(
        parameter = ["ATE", "beta(h_q)", "LATE"],
        LowerBound = lb,
        UpperBound = ub,
        CI_Lower = ci_ll,
        CI_Upper = ci_ul
    )
    
    return (res=res, nuis=(pi=pihat, la1=la1hat, la0=la0hat, hq=hq, gamhat=la1hat .- la0hat), 
            ifvals=DataFrame(ifvals_atel=ifvals_atel, ifvals_ateu=ifvals_ateu, ifvals_hql=ifvals_hql, ifvals_hqu=ifvals_hqu))
end
