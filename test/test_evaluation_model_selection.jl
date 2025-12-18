using Test
using TimeSeriesKit
using DataFrames

@testset "Model Selection" begin
    # Create simple test data
    ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    
    @testset "cross_validate - basic functionality" begin
        model = LinearModel()
        result = cross_validate(model, ts, 3)
        
        @test haskey(result, :mean_rmse)
        @test haskey(result, :std_rmse)
        @test haskey(result, :mean_mae)
        @test haskey(result, :std_mae)
        @test haskey(result, :cv_scores)
        
        @test result.mean_rmse >= 0
        @test result.std_rmse >= 0
        @test result.mean_mae >= 0
        @test result.std_mae >= 0
        @test length(result.cv_scores) > 0
    end
    
    @testset "cross_validate - verbose mode" begin
        model = LinearModel()
        # Test that verbose mode runs without error
        result = cross_validate(model, ts, 2, verbose=true)
        @test result.mean_rmse >= 0
    end
    
    @testset "cross_validate - ArgumentError for n_splits < 2" begin
        model = LinearModel()
        @test_throws ArgumentError cross_validate(model, ts, 1)
        @test_throws ArgumentError cross_validate(model, ts, 0)
    end
    
    @testset "cross_validate - short time series" begin
        short_ts = TimeSeries([1.0, 2.0, 3.0])
        model = LinearModel()
        @test_throws ArgumentError cross_validate(model, short_ts, 5)
    end
    
    @testset "cross_validate - AR model" begin
        ar_model = ARModel(p=2)
        result = cross_validate(ar_model, ts, 2)
        
        @test result.mean_rmse >= 0
        @test result.std_rmse >= 0
        @test length(result.cv_scores) > 0
    end
    
    @testset "cross_validate - all folds fail" begin
        # Create a time series that's too short for the model
        tiny_ts = TimeSeries([1.0, 2.0, 3.0, 4.0])
        model = ARModel(p=10)  # p too large for the data
        
        @test_throws ArgumentError cross_validate(model, tiny_ts, 2)
    end
    
    @testset "grid_search - basic functionality" begin
        configs = Dict(
            LinearModel => Dict()
        )
        result = grid_search(configs, ts, 2)
        
        @test haskey(result, :best_model)
        @test haskey(result, :best_score)
        @test haskey(result, :best_params)
        @test haskey(result, :results_df)
        
        @test result.best_model !== nothing
        @test result.best_score >= 0
        @test result.results_df isa DataFrame
        @test nrow(result.results_df) > 0
    end
    
    @testset "grid_search - multiple models" begin
        configs = Dict(
            LinearModel => Dict(),
            ARModel => Dict(:p => [1, 2])
        )
        result = grid_search(configs, ts, 2)
        
        @test result.best_model !== nothing
        @test result.best_score >= 0
        @test nrow(result.results_df) == 3  # 1 LinearModel + 2 ARModel configs
    end
    
    @testset "grid_search - parameter combinations" begin
        configs = Dict(
            LinearModel => Dict(:sliding_window => [2, 3])
        )
        result = grid_search(configs, ts, 2)
        
        @test nrow(result.results_df) == 2
        @test result.best_model !== nothing
        @test result.best_params isa Dict
    end
    
    @testset "grid_search - verbose mode" begin
        configs = Dict(
            LinearModel => Dict()
        )
        result = grid_search(configs, ts, 2, verbose=true)
        @test result.best_model !== nothing
    end
    
    @testset "grid_search - no valid configurations" begin
        # Use invalid parameters that will fail
        configs = Dict(
            ARModel => Dict(:p => [100])  # p too large for data
        )
        @test_throws ErrorException grid_search(configs, ts, 2)
    end
    
    @testset "grid_search - empty parameter grid" begin
        configs = Dict(
            LinearModel => Dict()
        )
        result = grid_search(configs, ts, 2)
        
        @test result.best_params == Dict()
        @test result.best_model !== nothing
    end
    
    @testset "grid_search - results DataFrame structure" begin
        configs = Dict(
            LinearModel => Dict(:sliding_window => [2, 3])
        )
        result = grid_search(configs, ts, 2)
        df = result.results_df
        
        @test "model" in names(df)
        @test "mean_rmse" in names(df)
        @test "std_rmse" in names(df)
        @test "mean_mae" in names(df)
        @test "std_mae" in names(df)
        @test "params" in names(df)
        @test "fitted_params" in names(df)
        
        # Check that results are sorted by RMSE
        @test issorted(df.mean_rmse)
    end
    
    @testset "grid_search - best model selection" begin
        configs = Dict(
            LinearModel => Dict(:sliding_window => [2, 3, 4])
        )
        result = grid_search(configs, ts, 2)
        
        # Best model should have the lowest RMSE
        @test result.best_score == minimum(result.results_df.mean_rmse)
    end
    
    @testset "grid_search - fitted parameters formatting" begin
        configs = Dict(
            LinearModel => Dict(:sliding_window => [2])
        )
        result = grid_search(configs, ts, 2)
        
        # Check that fitted_params column exists and is formatted
        @test "fitted_params" in names(result.results_df)
        @test result.results_df.fitted_params[1] isa String
    end
    
    @testset "grid_search - cartesian product" begin
        # Test multiple parameters with multiple values
        configs = Dict(
            LinearModel => Dict(:sliding_window => [2, 3])
        )
        result = grid_search(configs, ts, 2)
        
        @test nrow(result.results_df) == 2
    end
    
    @testset "cross_validate - model with no parameters" begin
        model = LinearModel()
        result = cross_validate(model, ts, 2)
        
        @test result.mean_rmse >= 0
        @test length(result.cv_scores) > 0
    end
    
    @testset "grid_search - single model type" begin
        configs = Dict(
            LinearModel => Dict(:sliding_window => [2])
        )
        result = grid_search(configs, ts, 2)
        
        @test typeof(result.best_model) == LinearModel
        @test result.best_params[:sliding_window] == 2
    end
    
    @testset "cross_validate - model state reset" begin
        # Test that model state is properly reset between folds
        model = LinearModel()
        result1 = cross_validate(model, ts, 2)
        result2 = cross_validate(model, ts, 2)
        
        # Results should be consistent
        @test result1.mean_rmse ≈ result2.mean_rmse
    end
    
    @testset "grid_search - model fitting on full data" begin
        configs = Dict(
            LinearModel => Dict()
        )
        result = grid_search(configs, ts, 2)
        
        # Best model should be fitted
        @test TimeSeriesKit.is_fitted(result.best_model)
    end
    
    @testset "cross_validate - fold generation" begin
        # Test with minimal data to check fold boundaries
        minimal_ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        model = LinearModel(sliding_window=2)
        result = cross_validate(model, minimal_ts, 2)
        
        @test length(result.cv_scores) > 0
        @test result.mean_rmse >= 0
    end
    
    @testset "grid_search - parameter formatting" begin
        configs = Dict(
            LinearModel => Dict(:sliding_window => [2, 3])
        )
        result = grid_search(configs, ts, 2)
        
        # Check params column formatting
        @test all(x -> x isa String, result.results_df.params)
        @test any(x -> occursin("sliding_window", x), result.results_df.params)
    end
end
