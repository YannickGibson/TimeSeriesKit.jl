using Test
using Aqua

@testset "TimeSeriesKit.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        using TimeSeriesKit
        Aqua.test_all(TimeSeriesKit)
    end
    
    include("test_core_types.jl")
    include("test_core_utils.jl")
    include("test_core_processes.jl")
    include("test_evaluation_metrics.jl")
end

