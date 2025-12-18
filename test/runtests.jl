using Test
using Aqua

@testset "TimeSeriesKit.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        using TimeSeriesKit
        Aqua.test_all(TimeSeriesKit)
    end
    
    include("core/test_core_types.jl")
    include("core/test_core_utils.jl")
    include("core/test_core_processes.jl")
    include("test_evaluation_metrics.jl")
    include("test_evaluation_model_selection.jl")
    include("models/test_models_abstract.jl")
    include("models/test_models_linear.jl")
    include("models/test_models_ar.jl")
    include("models/test_models_arima.jl")
    include("models/test_models_ses.jl")
end

