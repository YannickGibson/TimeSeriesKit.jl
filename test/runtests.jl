using Test
using Aqua

@testset "TimeSeriesKit.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        using TimeSeriesKit
        Aqua.test_all(TimeSeriesKit)
    end
    
    include("test_core_types.jl")
end

