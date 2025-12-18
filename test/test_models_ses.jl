using Test
using TimeSeriesKit
using TimeSeriesKit.Models
using Statistics

@testset "SES Model" begin
    
    @testset "SESModel - construction" begin
        # Default construction
        model = SESModel()
        @test model isa SESModel
        @test model.alpha == 0.8
        @test !is_fitted(model)
        @test min_train_size(model) == 2

        # Custom alpha
        model = SESModel(alpha=0.5)
        @test model.alpha == 0.5
    end
    
    @testset "SESModel - construction errors" begin
        # Alpha out of range
        @test_throws ArgumentError SESModel(alpha=0.0)
        @test_throws ArgumentError SESModel(alpha=1.0)
        @test_throws ArgumentError SESModel(alpha=-0.1)
        @test_throws ArgumentError SESModel(alpha=1.5)
    end
    
    @testset "SESModel - min_train_size" begin
        model = SESModel()
        @test min_train_size(model) == 2
    end
    
    @testset "ses_forecast helper function" begin
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        alpha = 0.5
        horizon = 3
        
        fitted, forecasts, level = TimeSeriesKit.Models.ETS.ses_forecast(values, alpha, horizon)
        
        # Check dimensions
        @test length(fitted) == length(values)
        @test length(forecasts) == horizon
        
        # Check first fitted value is first observation
        @test fitted[1] == values[1]
        
        # Check forecasts are constant (SES property)
        @test all(forecasts .== forecasts[1])
        @test forecasts[1] ≈ level
    end
    
    @testset "SESModel - fit" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0])
        model = SESModel(alpha=0.5)
        
        fit(model, ts)
        
        @test is_fitted(model)
        @test haskey(model.state.parameters, :alpha)
        @test haskey(model.state.parameters, :level)
        @test model.state.parameters[:alpha] == 0.5
        
        # Check fitted values
        @test length(model.state.fitted_values) == 5
        @test model.state.fitted_values[1] == 1.0  # First fitted value is first observation
        
        # Check residuals
        @test length(model.state.residuals) == 5
        @test model.state.residuals[1] == 0.0  # First residual is zero
    end
    
    @testset "SESModel - fit with different alpha values" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Low alpha (more smoothing)
        model_low = SESModel(alpha=0.1)
        fit(model_low, ts)
        
        # High alpha (less smoothing)
        model_high = SESModel(alpha=0.9)
        fit(model_high, ts)
        
        # High alpha should result in fitted values closer to actual values
        residuals_low = model_low.state.residuals
        residuals_high = model_high.state.residuals
        
        @test mean(abs.(residuals_high)) < mean(abs.(residuals_low))
    end
    
    @testset "SESModel - predict" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0])
        model = SESModel(alpha=0.5)
        fit(model, ts)
        
        # Predict
        x_values = [6.0, 7.0, 8.0]
        pred_ts = predict(model, x_values)
        
        @test pred_ts isa TimeSeries
        @test length(pred_ts) == 3
        @test pred_ts.timestamps == x_values
        
        # All predictions should be equal (constant forecast)
        @test all(pred_ts.values .== pred_ts.values[1])
        @test pred_ts.values[1] == model.state.parameters[:level]
    end
    
    @testset "SESModel - predict before fit" begin
        ts = TimeSeries([1.0, 2.0, 3.0])
        model = SESModel()
        
        @test_throws ErrorException predict(model, [4.0, 5.0])
    end
    
    @testset "SESModel - get_parameters" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0])
        model = SESModel(alpha=0.6)
        fit(model, ts)
        
        params = get_parameters(model)
        @test haskey(params, :alpha)
        @test haskey(params, :level)
        @test params[:alpha] == 0.6
    end
    
    @testset "SESModel - get_residuals" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0])
        model = SESModel(alpha=0.5)
        fit(model, ts)
        
        residuals = get_residuals(model)
        @test length(residuals) == 5
        @test residuals[1] == 0.0  # First residual is zero
        @test all(.!isnan.(residuals))
    end
    
    @testset "SESModel - trend detection" begin
        # Upward trend
        ts_up = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        model_up = SESModel(alpha=0.5)
        fit(model_up, ts_up)
        
        # Level should be below the last value (SES lags)
        @test model_up.state.parameters[:level] < 8.0
        
        # Constant trend
        ts_const = TimeSeries([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        model_const = SESModel(alpha=0.5)
        fit(model_const, ts_const)
        
        # Level should be approximately 5.0
        @test model_const.state.parameters[:level] ≈ 5.0 atol=0.1
    end
end
