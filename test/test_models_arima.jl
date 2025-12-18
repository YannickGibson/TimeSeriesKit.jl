using Test
using TimeSeriesKit

@testset "ARIMA Model" begin
    
    @testset "ARIMAModel - construction" begin
        model = ARIMAModel(p=1, d=1, q=1)
        @test model isa ARIMAModel
        @test model.p == 1
        @test model.d == 1
        @test model.q == 1
        @test !model.state.is_fitted
    end
    
    @testset "ARIMAModel - construction errors" begin
        @test_throws ArgumentError ARIMAModel(p=-1, d=0, q=1)
        @test_throws ArgumentError ARIMAModel(p=0, d=-1, q=1)
        @test_throws ArgumentError ARIMAModel(p=1, d=0, q=-1)
        @test_throws ArgumentError ARIMAModel(p=0, d=0, q=0)
    end
    
    @testset "ARIMAModel - min_train_size" begin
        model1 = ARIMAModel(p=1, d=0, q=0)
        @test min_train_size(model1) >= 2
        
        model2 = ARIMAModel(p=2, d=1, q=1)
        @test min_train_size(model2) >= 4
        
        model3 = ARIMAModel(p=1, d=2, q=0)
        @test min_train_size(model3) >= 4
    end
    
    @testset "difference_series" begin
        values = [1.0, 3.0, 6.0, 10.0, 15.0]
        
        # d=1
        diff1 = TimeSeriesKit.Models.ARIMA.difference_series(values, 1)
        @test diff1 ≈ [2.0, 3.0, 4.0, 5.0]
        
        # d=2
        diff2 = TimeSeriesKit.Models.ARIMA.difference_series(values, 2)
        @test diff2 ≈ [1.0, 1.0, 1.0]
        
        # d=0
        diff0 = TimeSeriesKit.Models.ARIMA.difference_series(values, 0)
        @test diff0 == values
    end
    
    @testset "integrate_forecast" begin
        forecasts = [1.0, 1.0, 1.0]
        last_values = [10.0]
        
        # d=1
        integrated1 = TimeSeriesKit.Models.ARIMA.integrate_forecast(forecasts, last_values, 1)
        @test integrated1 ≈ [11.0, 12.0, 13.0]
        
        # d=0
        integrated0 = TimeSeriesKit.Models.ARIMA.integrate_forecast(forecasts, last_values, 0)
        @test integrated0 == forecasts
    end
    
    @testset "fit_arma - AR only (q=0)" begin
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        intercept, ar_coeffs, ma_coeffs, fitted, residuals = TimeSeriesKit.Models.ARIMA.fit_arma(values, 1, 0)
        
        @test length(ar_coeffs) == 1
        @test length(ma_coeffs) == 0
        @test length(fitted) == 6
        @test length(residuals) == 6
    end
    
    @testset "fit_arma - MA only (p=0)" begin
        values = [1.0, 2.0, 1.5, 2.5, 2.0, 3.0]
        intercept, ar_coeffs, ma_coeffs, fitted, residuals = TimeSeriesKit.Models.ARIMA.fit_arma(values, 0, 1)
        
        @test length(ar_coeffs) == 0
        @test length(ma_coeffs) == 1
        @test length(fitted) == 6
        @test length(residuals) == 6
    end
    
    @testset "fit_arma - full ARMA(1,1)" begin
        values = collect(1.0:10.0)
        intercept, ar_coeffs, ma_coeffs, fitted, residuals = TimeSeriesKit.Models.ARIMA.fit_arma(values, 1, 1)
        
        @test length(ar_coeffs) == 1
        @test length(ma_coeffs) == 1
        @test length(fitted) == 10
        @test length(residuals) == 10
    end
    
    @testset "fit_arma - too short series" begin
        values = [1.0, 2.0, 3.0]
        @test_throws ArgumentError TimeSeriesKit.Models.ARIMA.fit_arma(values, 5, 0)
        @test_throws ArgumentError TimeSeriesKit.Models.ARIMA.fit_arma(values, 3, 1)
    end
    
    @testset "ARIMAModel - fit ARIMA(1,0,0) = AR(1)" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        model = ARIMAModel(p=1, d=0, q=0)
        fit(model, ts)
        
        @test model.state.is_fitted
        @test model.p == 1
        @test model.d == 0
        @test model.q == 0
    end
    
    @testset "ARIMAModel - fit ARIMA(1,1,0)" begin
        ts = TimeSeries([1.0, 3.0, 6.0, 10.0, 15.0, 21.0])
        model = ARIMAModel(p=1, d=1, q=0)
        fit(model, ts)
        
        @test model.state.is_fitted
        @test haskey(model.state.parameters, :original_values)
        @test length(model.state.parameters[:original_values]) == 6
    end
    
    @testset "ARIMAModel - fit ARIMA(0,1,1)" begin
        ts = TimeSeries(collect(1.0:8.0))
        model = ARIMAModel(p=0, d=1, q=1)
        fit(model, ts)
        
        @test model.state.is_fitted
    end
    
    @testset "ARIMAModel - forecast" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        model = ARIMAModel(p=1, d=0, q=0)
        fit(model, ts)
        
        forecast_ts = forecast(model, ts, 3)
        
        @test forecast_ts isa TimeSeries
        @test length(forecast_ts) == 3
    end
    
    @testset "ARIMAModel - forecast before fit" begin
        ts = TimeSeries([1.0, 2.0, 3.0])
        model = ARIMAModel(p=1, d=0, q=0)
        
        @test_throws ErrorException forecast(model, ts, 2)
    end
end
