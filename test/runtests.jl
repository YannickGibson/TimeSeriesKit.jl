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
    include("test_models_abstract.jl")
    include("test_models_linear.jl")
    include("test_models_ar.jl")
    include("test_models_arima.jl")
    include("test_models_ses.jl")
end

