using Test
using TimeSeriesKit
using Statistics

@testset "Linear Models" begin
    
    @testset "LinearModel - construction" begin
        model = LinearModel()
        @test model isa LinearModel
        @test model.sliding_window == 5
        @test !model.state.is_fitted
        
        # Custom sliding window
        model2 = LinearModel(sliding_window=10)
        @test model2.sliding_window == 10
    end
    
    @testset "LinearModel - construction error" begin
        @test_throws ArgumentError LinearModel(sliding_window=1)
        @test_throws ArgumentError LinearModel(sliding_window=0)
    end
    
    @testset "LinearModel - min_train_size" begin
        model = LinearModel(sliding_window=5)
        @test min_train_size(model) == 5
        
        model2 = LinearModel(sliding_window=10)
        @test min_train_size(model2) == 10
        
        # Should be at least 2
        model3 = LinearModel(sliding_window=2)
        @test min_train_size(model3) == 2
    end
    
    @testset "LinearModel - fit with simple trend" begin
        # Perfect linear trend: y = 2 + 3*x
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 8.0, 11.0, 14.0, 17.0])
        
        model = LinearModel()
        fit(model, ts)
        
        @test model.state.is_fitted
        @test haskey(model.state.parameters, :intercept)
        @test haskey(model.state.parameters, :slope)
        
        # Check fitted parameters (should be close to true values)
        @test model.state.parameters[:intercept] ≈ 2.0 atol=1e-10
        @test model.state.parameters[:slope] ≈ 3.0 atol=1e-10
        
        # Check fitted values
        @test length(model.state.fitted_values) == 5
        @test model.state.fitted_values ≈ [5.0, 8.0, 11.0, 14.0, 17.0] atol=1e-10
        
        # Check residuals (should be zero for perfect fit)
        @test all(abs.(model.state.residuals) .< 1e-10)
    end
    
    @testset "LinearModel - fit with noisy data" begin
        # Linear trend with noise: approximately y = 10 + 2*x
        ts = TimeSeries(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [12.1, 13.9, 16.2, 17.8, 20.1, 21.9, 23.8, 26.2, 27.9, 30.1]
        )
        
        model = LinearModel()
        fit(model, ts)
        
        @test model.state.is_fitted
        
        # Parameters should be approximately 10 and 2
        @test model.state.parameters[:intercept] ≈ 10.0 atol=0.5
        @test model.state.parameters[:slope] ≈ 2.0 atol=0.1
        
        # Residuals should exist
        @test length(model.state.residuals) == 10
        @test mean(abs.(model.state.residuals)) < 0.5  # Small average error
    end
    
    @testset "LinearModel - predict" begin
        # Fit to known data
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 8.0, 11.0, 14.0, 17.0])
        model = LinearModel()
        fit(model, ts)
        
        # Predict at new points
        predictions = predict(model, [6.0, 7.0, 8.0])
        
        @test predictions isa TimeSeries
        @test length(predictions) == 3
        @test predictions.timestamps == [6.0, 7.0, 8.0]
        
        # With intercept=2 and slope=3: y = 2 + 3*x
        @test predictions.values ≈ [20.0, 23.0, 26.0] atol=1e-10
    end
    
    @testset "LinearModel - predict before fit" begin
        model = LinearModel()
        @test_throws ErrorException predict(model, [1.0, 2.0, 3.0])
    end
    
    @testset "LinearModel - refit" begin
        # First fit
        ts1 = TimeSeries([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        model = LinearModel()
        fit(model, ts1)
        
        intercept1 = model.state.parameters[:intercept]
        slope1 = model.state.parameters[:slope]
        
        # Refit with different data
        ts2 = TimeSeries([1.0, 2.0, 3.0], [5.0, 7.0, 9.0])
        fit(model, ts2)
        
        intercept2 = model.state.parameters[:intercept]
        slope2 = model.state.parameters[:slope]
        
        # Parameters should be different
        @test intercept1 != intercept2
        @test slope1 == slope2  # Slope should be same (both have slope=2)
        @test intercept2 ≈ 3.0 atol=1e-10
    end
    
    @testset "LinearModel - validate_timeseries integration" begin
        # Empty time series should throw
        ts_empty = TimeSeries(Float64[])
        model = LinearModel()
        @test_throws ArgumentError fit(model, ts_empty)
    end
    
    @testset "LinearModel - create_matrix_X helper" begin
        x_values = [1.0, 2.0, 3.0, 4.0]
        X = TimeSeriesKit.Models.Linear.create_matrix_X(x_values)
        
        @test size(X) == (4, 2)
        @test X[:, 1] == [1.0, 1.0, 1.0, 1.0]  # Intercept column
        @test X[:, 2] == [1.0, 2.0, 3.0, 4.0]  # Time trend column
    end
    
    @testset "LinearModel - perfect horizontal line" begin
        # Constant values (slope should be 0)
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 10.0, 10.0, 10.0, 10.0])
        model = LinearModel()
        fit(model, ts)
        
        @test model.state.parameters[:intercept] ≈ 10.0 atol=1e-10
        @test model.state.parameters[:slope] ≈ 0.0 atol=1e-10
        
        # Predictions should all be 10
        pred = predict(model, [6.0, 7.0, 8.0])
        @test all(pred.values .≈ 10.0)
    end
    
    @testset "LinearModel - negative slope" begin
        # Decreasing trend: y = 20 - 2*x
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0], [18.0, 16.0, 14.0, 12.0])
        model = LinearModel()
        fit(model, ts)
        
        @test model.state.parameters[:intercept] ≈ 20.0 atol=1e-10
        @test model.state.parameters[:slope] ≈ -2.0 atol=1e-10
        
        # Predict future (should continue decreasing)
        pred = predict(model, [5.0, 6.0])
        @test pred.values ≈ [10.0, 8.0] atol=1e-10
    end
    
    @testset "LinearModel - two point minimum" begin
        # With only 2 points
        ts = TimeSeries([1.0, 2.0], [3.0, 5.0])
        model = LinearModel(sliding_window=2)
        fit(model, ts)
        
        @test model.state.is_fitted
        @test model.state.parameters[:intercept] ≈ 1.0 atol=1e-10
        @test model.state.parameters[:slope] ≈ 2.0 atol=1e-10
    end
    
    @testset "LinearModel - iterative_predict with sliding_window" begin
        # Test iterative prediction with sliding window constraint
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
                       [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
        model = LinearModel(sliding_window=3)
        
        pred = iterative_predict(model, ts, 2)
        
        @test pred isa TimeSeries
        # Should predict from position 4 onwards + 2 future = (8-3) + 2 = 7 predictions
        @test length(pred) == 7
        @test pred.name == "LinearModel (out-of-sample)"
        @test all(.!isnan.(pred.values))
    end
    
    @testset "LinearModel - iterative_predict insufficient data" begin
        # Test with time series smaller than min_train_size
        ts = TimeSeries([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        model = LinearModel(sliding_window=5)  # Requires at least 5 points
        
        @test_throws ArgumentError iterative_predict(model, ts, 2)
    end
    
    @testset "LinearModel - iterative_predict invalid horizon" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                       [2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        model = LinearModel(sliding_window=3)
        
        # Horizon must be at least 1
        @test_throws ArgumentError iterative_predict(model, ts, 0)
        @test_throws ArgumentError iterative_predict(model, ts, -1)
    end
    
    @testset "LinearModel - iterative_predict basic functionality" begin
        # Simple test without sliding window constraint
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], 
                       [3.0, 5.0, 7.0, 9.0, 11.0])
        model = LinearModel(sliding_window=2)
        
        pred = iterative_predict(model, ts, 3)
        
        @test pred isa TimeSeries
        @test length(pred) > 0
        # Check timestamps are extrapolated correctly
        @test pred.timestamps[end] > ts.timestamps[end]
    end
    
    @testset "LinearModel - fit stores variance parameters" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 4.0, 6.2, 7.9, 10.1])
        model = LinearModel()
        fit(model, ts)
        
        # Check that variance parameters are stored
        @test haskey(model.state.parameters, :residual_variance)
        @test haskey(model.state.parameters, :intercept_variance)
        @test haskey(model.state.parameters, :slope_variance)
        @test haskey(model.state.parameters, :covariance)
        
        # Variances should be non-negative
        @test model.state.parameters[:residual_variance] >= 0
        @test model.state.parameters[:intercept_variance] >= 0
        @test model.state.parameters[:slope_variance] >= 0
    end
    
    @testset "LinearModel - predict with uncertainty" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 4.0, 6.2, 7.9, 10.1])
        model = LinearModel()
        fit(model, ts)
        
        # Predict with uncertainty
        result = predict(model, [6.0, 7.0, 8.0], return_uncertainty=true)
        
        @test result isa PredictionResult
        @test result.predictions isa TimeSeries
        @test result.prediction_variance !== nothing
        @test result.prediction_std !== nothing
        
        @test length(result.predictions) == 3
        @test length(result.prediction_variance) == 3
        @test length(result.prediction_std) == 3
        
        # Standard deviation should be sqrt of variance
        @test result.prediction_std ≈ sqrt.(result.prediction_variance)
        
        # All variances should be non-negative
        @test all(result.prediction_variance .>= 0)
        @test all(result.prediction_std .>= 0)
    end
    
    @testset "LinearModel - predict without uncertainty (backward compatibility)" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 4.0, 6.2, 7.9, 10.1])
        model = LinearModel()
        fit(model, ts)
        
        # Predict without uncertainty (default behavior)
        result = predict(model, [6.0, 7.0, 8.0])
        
        @test result isa TimeSeries
        @test !(result isa PredictionResult)
        @test length(result) == 3
    end
    
    @testset "LinearModel - prediction variance grows with distance" begin
        # Use data with some noise to ensure non-zero variance
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.1, 7.8, 10.2])
        model = LinearModel()
        fit(model, ts)
        
        # Predict at different distances from center
        # Center of data is at x=3.0
        result = predict(model, [3.0, 10.0], return_uncertainty=true)
        
        center_var = result.prediction_variance[1]  # x=3.0 (center)
        far_var = result.prediction_variance[2]     # x=10.0 (far from center)
        
        # Variance should increase as we move away from center
        @test far_var >= center_var
        @test result.prediction_variance[1] >= 0
        @test result.prediction_variance[2] > 0  # Far prediction should have positive variance
    end
    
    @testset "LinearModel - iterative_predict with uncertainty" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 4.0, 6.2, 7.9, 10.1])
        model = LinearModel(sliding_window=2)
        
        result = iterative_predict(model, ts, 3, return_uncertainty=true)
        
        @test result isa PredictionResult
        @test result.predictions isa TimeSeries
        @test result.prediction_variance !== nothing
        @test result.prediction_std !== nothing
        
        # Should have predictions for in-sample and out-of-sample
        @test length(result.predictions) > 3  # At least the horizon
        @test length(result.prediction_variance) == length(result.predictions)
        @test length(result.prediction_std) == length(result.predictions)
        
        # All variances should be non-negative
        @test all(result.prediction_variance .>= 0)
    end
    
    @testset "LinearModel - iterative_predict without uncertainty" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 4.0, 6.2, 7.9, 10.1])
        model = LinearModel(sliding_window=2)
        
        result = iterative_predict(model, ts, 3)
        
        @test result isa TimeSeries
        @test !(result isa PredictionResult)
    end
    
    @testset "RidgeModel - predict with uncertainty not supported" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 4.0, 6.2, 7.9, 10.1])
        model = RidgeModel()
        fit(model, ts)
        
        # RidgeModel doesn't have variance parameters, so should return TimeSeries
        result = predict(model, [6.0, 7.0], return_uncertainty=true)
        
        @test result isa TimeSeries
        @test !(result isa PredictionResult)
    end
    
    @testset "PredictionResult - structure" begin
        ts = TimeSeries([1.0, 2.0], [2.0, 4.0])
        variance = [0.5, 1.0]
        
        result = PredictionResult(ts, variance)
        
        @test result.predictions === ts
        @test result.prediction_variance == variance
        @test result.prediction_std ≈ sqrt.(variance)
    end
    
    @testset "PredictionResult - with nothing variance" begin
        ts = TimeSeries([1.0, 2.0], [2.0, 4.0])
        
        result = PredictionResult(ts, nothing)
        
        @test result.predictions === ts
        @test result.prediction_variance === nothing
        @test result.prediction_std === nothing
    end
end

@testset "Ridge Models" begin
    
    @testset "RidgeModel - construction" begin
        model = RidgeModel()
        @test model isa RidgeModel
        @test model.λ == 1.0
        @test model.sliding_window == 5
        @test !model.state.is_fitted
        
        # Custom parameters
        model2 = RidgeModel(λ=0.5, sliding_window=10)
        @test model2.λ == 0.5
        @test model2.sliding_window == 10
    end
    
    @testset "RidgeModel - construction errors" begin
        @test_throws ArgumentError RidgeModel(λ=-1.0)
        @test_throws ArgumentError RidgeModel(sliding_window=1)
    end
    
    @testset "RidgeModel - min_train_size" begin
        model = RidgeModel(sliding_window=5)
        @test min_train_size(model) == 5
        
        model2 = RidgeModel(sliding_window=2)
        @test min_train_size(model2) == 2
    end
    
    @testset "RidgeModel - fit with simple trend" begin
        # Perfect linear trend: y = 2 + 3*x
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 8.0, 11.0, 14.0, 17.0])
        
        model = RidgeModel(λ=0.01)  # Small λ for near-OLS behavior
        fit(model, ts)
        
        @test model.state.is_fitted
        @test haskey(model.state.parameters, :intercept)
        @test haskey(model.state.parameters, :slope)
        @test haskey(model.state.parameters, :λ)
        
        # With small λ, should be close to OLS solution
        @test model.state.parameters[:intercept] ≈ 2.0 atol=0.1
        @test model.state.parameters[:slope] ≈ 3.0 atol=0.1
        @test model.state.parameters[:λ] == 0.01
    end
    
    @testset "RidgeModel - regularization effect" begin
        # Same data, different λ values
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 8.0, 11.0, 14.0, 17.0])
        
        # Small λ (close to OLS)
        model_small = RidgeModel(λ=0.001)
        fit(model_small, ts)
        slope_small = model_small.state.parameters[:slope]
        
        # Large λ (more regularization)
        model_large = RidgeModel(λ=100.0)
        fit(model_large, ts)
        slope_large = model_large.state.parameters[:slope]
        
        # Higher λ should shrink coefficients more
        @test abs(slope_large) < abs(slope_small)
    end
    
    @testset "RidgeModel - λ=0 approximates OLS" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 8.0, 11.0, 14.0, 17.0])
        
        # Ridge with λ=0
        ridge = RidgeModel(λ=0.0)
        fit(ridge, ts)
        
        # OLS
        linear = LinearModel()
        fit(linear, ts)
        
        # Should be very close
        @test ridge.state.parameters[:intercept] ≈ linear.state.parameters[:intercept] atol=1e-10
        @test ridge.state.parameters[:slope] ≈ linear.state.parameters[:slope] atol=1e-10
    end
    
    @testset "RidgeModel - predict" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 8.0, 11.0, 14.0, 17.0])
        model = RidgeModel(λ=0.01)
        fit(model, ts)
        
        # Predict at new points
        predictions = predict(model, [6.0, 7.0, 8.0])
        
        @test predictions isa TimeSeries
        @test length(predictions) == 3
        @test predictions.timestamps == [6.0, 7.0, 8.0]
        
        # Values should be close to linear extrapolation
        @test predictions.values[1] ≈ 20.0 atol=0.5
        @test predictions.values[2] ≈ 23.0 atol=0.5
        @test predictions.values[3] ≈ 26.0 atol=0.5
    end
    
    @testset "RidgeModel - predict before fit" begin
        model = RidgeModel()
        @test_throws ErrorException predict(model, [1.0, 2.0, 3.0])
    end
    
    @testset "RidgeModel - intercept not regularized" begin
        # Data with high intercept
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0], [100.0, 102.0, 104.0, 106.0])
        
        model = RidgeModel(λ=10.0)
        fit(model, ts)
        
        # Intercept should remain close to 98 (true value)
        # Slope should be shrunk toward 0
        @test model.state.parameters[:intercept] ≈ 101.0 atol=5.0
        @test abs(model.state.parameters[:slope]) < 2.0  # Shrunk from true value of 2
    end
    
    @testset "RidgeModel - noisy data regularization" begin
        # Noisy data where regularization should help
        ts = TimeSeries(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [10.5, 12.8, 14.2, 16.9, 18.1, 20.5, 22.3, 24.1]
        )
        
        model = RidgeModel(λ=1.0)
        fit(model, ts)
        
        @test model.state.is_fitted
        @test length(model.state.residuals) == 8
        @test length(model.state.fitted_values) == 8
    end
end
