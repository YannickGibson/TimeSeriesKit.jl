using TimeSeriesKit
using Aqua
using Test

@testset "TimeSeriesKit.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(TimeSeriesKit)
    end

    @testset "Core Types" begin
        # Test TimeSeries creation
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0])
        @test length(ts) == 5
        @test ts[1] == 1.0
        @test ts.values == [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test with timestamps
        ts_with_time = TimeSeries([1.0, 2.0, 3.0], [1, 2, 3])
        @test ts_with_time.timestamps == [1, 2, 3]
    end

    @testset "Core Utils" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        # Test validation
        @test validate_timeseries(ts) == true
        
        # Test train/test split
        train_ts, test_ts = split_train_test(ts, 0.8)
        @test length(train_ts) == 8
        @test length(test_ts) == 2
        
        # Test difference
        ts_diff = difference(ts, 1)
        @test length(ts_diff) == 9
        @test all(ts_diff.values .≈ 1.0)
    end

    @testset "Linear Model" begin
        # Create simple time series with linear trend
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        # Fit linear model
        model = LinearModel()
        fit(model, ts)
        
        @test is_fitted(model)
        
        # Make forecasts
        forecasts = forecast(model, ts, 3)
        @test length(forecasts) == 3
        @test forecasts[1] ≈ 11.0 atol=0.1
        
        # Test predict with specific x values
        predictions = predict(model, [11.0, 12.0, 13.0])
        @test length(predictions) == 3
    end

    @testset "AR Model" begin
        # Create simple AR(1) process
        ts = TimeSeries([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        
        # Fit AR(1) model
        model = ARModel(1)
        fit(model, ts)
        
        @test is_fitted(model)
        
        # Make forecasts
        forecasts = forecast(model, ts, 2)
        @test length(forecasts) == 2
    end

    @testset "SES Model" begin
        # Create time series
        ts = TimeSeries([10.0, 12.0, 11.0, 13.0, 12.5, 14.0, 13.5, 15.0])
        
        # Fit SES model with fixed alpha
        model = SESModel(0.3)
        fit(model, ts)
        
        @test is_fitted(model)
        @test model.state.parameters[:alpha] == 0.3
        
        # Make forecasts
        forecasts = forecast(model, ts, 3)
        @test length(forecasts) == 3
        # SES produces flat forecasts
        @test forecasts[1] ≈ forecasts[2] ≈ forecasts[3]
    end

    @testset "Evaluation Metrics" begin
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        predicted = [1.1, 2.1, 2.9, 4.2, 4.8]
        
        # Test MSE
        mse_val = mse(actual, predicted)
        @test mse_val ≈ 0.026 atol=0.01
        
        # Test MAE
        mae_val = mae(actual, predicted)
        @test mae_val ≈ 0.14 atol=0.01
        
        # Test RMSE
        rmse_val = rmse(actual, predicted)
        @test rmse_val ≈ sqrt(0.026) atol=0.04
    end

    @testset "Rolling Forecast" begin
        # Create simple time series
        ts = TimeSeries(collect(1.0:20.0))
        
        # Run rolling forecast with linear model
        model = LinearModel()
        result = rolling_forecast(model, ts, train_size=10, horizon=2, step=2)
        
        @test length(result.forecasts) > 0
        @test length(result.actuals) == length(result.forecasts)
        @test haskey(result.errors, :mse)
    end
end

