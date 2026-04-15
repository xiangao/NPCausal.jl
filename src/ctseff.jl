using DataFrames
using MLJ
using EvoTrees
using Random
using Statistics
using Base.Threads
using Distributions
using LinearAlgebra

function kern(t)
    return pdf(Normal(0, 1), t)
end

function locpoly(x, y, bw, xout)
    yout = zeros(length(xout))
    for (i, x0) in enumerate(xout)
        w = kern.((x .- x0) ./ bw)
        X_mat = hcat(ones(length(x)), x .- x0)
        W = Diagonal(w)
        try
            beta = (X_mat' * W * X_mat + 1e-8 * I) \ (X_mat' * W * y)
            yout[i] = beta[1]
        catch
            yout[i] = mean(y[w .> 1e-4])
        end
    end
    return yout
end

function interp1d(x, y, xout)
    yout = zeros(length(xout))
    for (i, x0) in enumerate(xout)
        if x0 <= x[1]
            yout[i] = y[1]
        elseif x0 >= x[end]
            yout[i] = y[end]
        else
            idx = searchsortedlast(x, x0)
            if idx == length(x)
                yout[i] = y[end]
            else
                x1, x2 = x[idx], x[idx+1]
                y1, y2 = y[idx], y[idx+1]
                yout[i] = y1 + (y2 - y1) * (x0 - x1) / (x2 - x1)
            end
        end
    end
    return yout
end

"""
    ctseff(y::AbstractVector, a::AbstractVector, X::DataFrame; kwargs...)

Estimating average effect curve for continuous treatment.
"""
function ctseff(y::AbstractVector, a::AbstractVector, X::DataFrame; 
                bw_seq, n_pts = 100, a_rng = (minimum(a), maximum(a)),
                model = EvoTreeRegressor(nrounds=100, eta=0.1, max_depth=5))
    
    n = length(y)
    a_min, a_max = a_rng
    a_vals = collect(range(a_min, a_max, length=n_pts))
    
    mach_pi = machine(model, X, a)
    MLJ.fit!(mach_pi, verbosity=0)
    pimod_vals = MLJ.predict(mach_pi, X)
    
    mach_pi2 = machine(model, X, (a .- pimod_vals).^2)
    MLJ.fit!(mach_pi2, verbosity=0)
    pi2mod_vals = clamp.(MLJ.predict(mach_pi2, X), 1e-4, Inf)
    
    Xa = copy(X)
    Xa.a = a
    mach_mu = machine(model, Xa, y)
    MLJ.fit!(mach_mu, verbosity=0)
    muhat_vals = MLJ.predict(mach_mu, Xa)
    
    muhat_mat = zeros(n, n_pts)
    @threads for j in 1:n_pts
        Xa_new = copy(X)
        Xa_new.a .= a_vals[j]
        muhat_mat[:, j] = MLJ.predict(mach_mu, Xa_new)
    end
    
    a_std = (a .- pimod_vals) ./ sqrt.(pi2mod_vals)
    
    function kde(x_data, x_eval; bw=1.06*std(x_data)*length(x_data)^(-1/5))
        out = zeros(length(x_eval))
        for (i, x0) in enumerate(x_eval)
            out[i] = mean(kern.((x_data .- x0) ./ bw)) / bw
        end
        return out
    end
    
    pihat = kde(a_std, a_std) ./ sqrt.(pi2mod_vals)
    pihat_mat = zeros(n, n_pts)
    @threads for j in 1:n_pts
        a_std_new = (a_vals[j] .- pimod_vals) ./ sqrt.(pi2mod_vals)
        pihat_mat[:, j] = kde(a_std, a_std_new) ./ sqrt.(pi2mod_vals)
    end
    
    mean_pihat_mat = vec(mean(pihat_mat, dims=1))
    varpihat = interp1d(a_vals, mean_pihat_mat, a)
    varpihat_mat = repeat(mean_pihat_mat', n, 1)
    
    muhat = muhat_vals
    mean_muhat_mat = vec(mean(muhat_mat, dims=1))
    mhat = interp1d(a_vals, mean_muhat_mat, a)
    mhat_mat = repeat(mean_muhat_mat', n, 1)
    
    varpihat = clamp.(varpihat, 1e-4, Inf)
    pseudo_out = (y .- muhat) ./ (pihat ./ varpihat) .+ mhat
    
    function w_fn(bw)
        w_avals = zeros(n_pts)
        for (j, a_val) in enumerate(a_vals)
            a_s = (a .- a_val) ./ bw
            kern_s = kern.(a_s) ./ bw
            
            m_a2_k = mean(a_s.^2 .* kern_s)
            m_k = mean(kern_s)
            m_a_k = mean(a_s .* kern_s)
            
            denom = m_k * m_a2_k - m_a_k^2
            if abs(denom) > 1e-10
                w_avals[j] = m_a2_k * (kern(0) / bw) / denom
            else
                w_avals[j] = 0.0
            end
        end
        return w_avals ./ n
    end
    
    function risk_fn(bw)
        w_v = w_fn(bw)
        hats = interp1d(a_vals, w_v, a)
        lp = locpoly(a, pseudo_out, bw, a)
        return mean( ((pseudo_out .- lp) ./ (1.0 .- hats)).^2 )
    end
    
    risk_est = [risk_fn(bw) for bw in bw_seq]
    h_opt = bw_seq[argmin(risk_est)]
    
    est = locpoly(a, pseudo_out, h_opt, a_vals)
    
    se = zeros(n_pts)
    for (j, a_val) in enumerate(a_vals)
        a_s = (a .- a_val) ./ h_opt
        kern_s = kern.(a_s) ./ h_opt
        
        X_mat = hcat(ones(length(a)), a_s)
        W = Diagonal(kern_s)
        beta = (X_mat' * W * X_mat + 1e-8 * I) \ (X_mat' * W * pseudo_out)
        
        m_k = mean(kern_s)
        m_a_k = mean(kern_s .* a_s)
        m_a2_k = mean(kern_s .* a_s.^2)
        Dh = [m_k m_a_k; m_a_k m_a2_k]
        
        kern_mat = repeat(kern.((a_vals .- a_val) ./ h_opt)' ./ h_opt, n, 1)
        g2 = repeat(((a_vals .- a_val) ./ h_opt)', n, 1)
        
        intfn1_mat = kern_mat .* (muhat_mat .- mhat_mat) .* varpihat_mat
        intfn2_mat = g2 .* kern_mat .* (muhat_mat .- mhat_mat) .* varpihat_mat
        
        int1 = zeros(n)
        int2 = zeros(n)
        for i in 1:n_pts-1
            d_a = a_vals[i+1] - a_vals[i]
            int1 .+= d_a .* (intfn1_mat[:, i] .+ intfn1_mat[:, i+1]) ./ 2.0
            int2 .+= d_a .* (intfn2_mat[:, i] .+ intfn2_mat[:, i+1]) ./ 2.0
        end
        
        mat1 = kern_s .* (pseudo_out .- beta[1] .- beta[2] .* a_s) .+ int1
        mat2 = a_s .* kern_s .* (pseudo_out .- beta[1] .- beta[2] .* a_s) .+ int2
        
        if abs(det(Dh)) < 1e-10
            Dh += 1e-8 * I
        end
        
        comb = inv(Dh) * vcat(mat1', mat2')
        cov_mat = cov(comb, dims=2)
        se[j] = sqrt(max(0.0, cov_mat[1, 1]))
    end
    
    ci_ll = est .- 1.96 .* se ./ sqrt(n)
    ci_ul = est .+ 1.96 .* se ./ sqrt(n)
    res = DataFrame(a_vals=a_vals, est=est, se=se, ci_ll=ci_ll, ci_ul=ci_ul)
    
    return (res=res, bw_risk=DataFrame(bw=bw_seq, risk=risk_est))
end
