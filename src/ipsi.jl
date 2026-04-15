using DataFrames
using MLJ
using EvoTrees
using Random
using Statistics
using Base.Threads
using Distributions

"""
    ipsi(y::AbstractVector, a::AbstractVector, x_trt::DataFrame, x_out::DataFrame, time::AbstractVector, id::AbstractVector, delta_seq::AbstractVector; kwargs...)

Estimating effects of incremental propensity score interventions.
"""
function ipsi(y::AbstractVector, a::AbstractVector, x_trt::DataFrame, x_out::DataFrame, time::AbstractVector, id::AbstractVector, delta_seq::AbstractVector;
              nsplits = 5, ci_level = 0.95,
              trt_model = EvoTreeClassifier(nrounds=100, eta=0.1, max_depth=5),
              out_model = EvoTreeRegressor(nrounds=100, eta=0.1, max_depth=5))
    
    ntimes = length(unique(time))
    end_time = maximum(time)
    unique_ids = unique(id)
    n = length(unique_ids)
    
    k = length(delta_seq)

    folds = shuffle(1:n)
    fold_assignments = [folds[1 + floor(Int, (i-1)*n/nsplits) : floor(Int, i*n/nsplits)] for i in 1:nsplits]
    
    id_to_fold_map = Dict{eltype(id), Int}()
    for (fold_idx, ids) in enumerate(fold_assignments)
        for id_val in ids
            id_to_fold_map[id_val] = fold_idx
        end
    end
    slong = [id_to_fold_map[val] for val in id]
    
    # Pre-calculate PS to simplify loops
    ps = zeros(length(a))
    @threads for split in 1:nsplits
        train_mask = slong .!= split
        test_mask = slong .== split
        mach_trt = machine(trt_model, x_trt[train_mask, :], categorical(a[train_mask] .== 1))
        MLJ.fit!(mach_trt, verbosity=0)
        ps[test_mask] = pdf.(MLJ.predict(mach_trt, x_trt[test_mask, :]), true)
    end
    ps = clamp.(ps, 1e-4, 1.0 - 1e-4)
    
    wt = zeros(length(a), k)
    cumwt = zeros(length(a), k)
    vt = zeros(length(a), k)
    rt = zeros(length(a), k)
    
    dat = DataFrame(time=time, id=id, a=a, ps=ps)
    y_dict = Dict(unique_ids[i] => y[i] for i in 1:n)
    dat.y = [t == end_time ? y_dict[id[i]] : NaN for (i, t) in enumerate(time)]
    
    ifvals = zeros(n, k)
    
    for (j, delta) in enumerate(delta_seq)
        wt[:, j] = (delta .* dat.a .+ 1 .- dat.a) ./ (delta .* dat.ps .+ 1 .- dat.ps)
        vt[:, j] = (1 .- delta) .* (dat.a .* (1 .- dat.ps) .- (1 .- dat.a) .* delta .* dat.ps) ./ delta
        
        for id_val in unique_ids
            mask = dat.id .== id_val
            idx_list = findall(mask)
            sort!(idx_list, by = i -> dat.time[i])
            c_wt = 1.0
            for i in idx_list
                c_wt *= wt[i, j]
                cumwt[i, j] = c_wt
            end
        end
    end
    
    for split in 1:nsplits
        train_mask = slong .!= split
        
        for j in 1:k
            delta = delta_seq[j]
            rtp1 = copy(dat.y)
            
            for t_step in 1:ntimes
                t = sort(unique(dat.time), rev=true)[t_step]
                
                mask_t = dat.time .== t
                train_t_mask = mask_t .& train_mask
                
                x_out_t = copy(x_out)
                x_out_t.a = a
                
                if sum(train_t_mask) > 0
                    mach_out = machine(out_model, x_out_t[train_t_mask, :], rtp1[train_t_mask])
                    MLJ.fit!(mach_out, verbosity=0)
                    
                    newx1 = x_out_t[mask_t, :]
                    newx1.a .= 1
                    newx0 = x_out_t[mask_t, :]
                    newx0.a .= 0
                    
                    m1 = MLJ.predict(mach_out, newx1)
                    m0 = MLJ.predict(mach_out, newx0)
                    
                    pi_t = dat.ps[mask_t]
                    rt[mask_t, j] = (delta .* pi_t .* m1 .+ (1 .- pi_t) .* m0) ./ (delta .* pi_t .+ 1 .- pi_t)
                else
                    rt[mask_t, j] .= 0.0
                end
                
                rtp1[mask_t] = rt[mask_t, j]
            end
            
            for (idx_n, id_val) in enumerate(unique_ids)
                if id_to_fold_map[id_val] == split
                    mask_end = (dat.id .== id_val) .& (dat.time .== end_time)
                    val1 = sum((cumwt[:, j] .* dat.y)[mask_end])
                    
                    mask_id = dat.id .== id_val
                    val2 = sum((cumwt[:, j] .* vt[:, j] .* rt[:, j])[mask_id])
                    
                    ifvals[idx_n, j] = val1 + val2
                end
            end
        end
    end
    
    est_eff = vec(mean(ifvals, dims=1))
    sigma = vec(std(ifvals, dims=1))
    
    ci_norm_bounds = abs(quantile(Normal(), (1 - ci_level) / 2))
    eff_ll = est_eff .- ci_norm_bounds .* sigma ./ sqrt(n)
    eff_ul = est_eff .+ ci_norm_bounds .* sigma ./ sqrt(n)
    
    nbs = 10000
    ifvals2 = (ifvals .- est_eff') ./ (sigma')
    maxvals = zeros(nbs)
    @threads for col in 1:nbs
        mult = 2.0 .* rand(Binomial(1, 0.5), n) .- 1.0
        maxvals[col] = maximum(abs.(vec(sum(mult .* ifvals2, dims=1))) ./ sqrt(n))
    end
    calpha = quantile(maxvals, ci_level)
    
    eff_ll2 = est_eff .- calpha .* sigma ./ sqrt(n)
    eff_ul2 = est_eff .+ calpha .* sigma ./ sqrt(n)
    
    se_eff = sigma ./ sqrt(n)

    res = DataFrame(
        increment = delta_seq, est = est_eff, se = se_eff,
        ci_ll = eff_ll2, ci_ul = eff_ul2
    )
    res2 = DataFrame(
        increment = delta_seq, est = est_eff, se = se_eff,
        ci_ll = eff_ll, ci_ul = eff_ul
    )
    
    return (res=res, res_ptwise=res2, calpha=calpha)
end
