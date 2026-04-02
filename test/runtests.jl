using Test
using NPCausal
using DataFrames
using Random

@testset "NPCausal.jl" begin
    Random.seed!(42)
    n = 10000
    X = DataFrame(x1 = randn(n), x2 = randn(n))
    a = rand([0, 1, 2], n)
    
    # y = x1 + x2*a_1 + 2*a_2 + noise
    y = X.x1 .+ X.x2 .* (a .== 1) .+ 2 .* (a .== 2) .+ randn(n)
    
    res = ate(y, a, X; nsplits=5)
    
    @test size(res.means, 1) == 3
    @test size(res.contrasts, 1) == 3
    
    # E[Y(2)] - E[Y(0)] should be approx 2.0
    contrast_2_0 = res.contrasts[res.contrasts.parameter .== "E{Y(2)} - E{Y(0)}", :Estimate][1]
    @test isapprox(contrast_2_0, 2.0, atol=0.2)
    
    # E[Y(1)] - E[Y(0)] should be approx 0.0 (since mean of x2 is 0)
    contrast_1_0 = res.contrasts[res.contrasts.parameter .== "E{Y(1)} - E{Y(0)}", :Estimate][1]
    @test isapprox(contrast_1_0, 0.0, atol=0.2)
end
