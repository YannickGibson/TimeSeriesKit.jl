using Test
using TimeSeriesKit
using Statistics

@testset "AR Model" begin
    
    @testset "ARModel - construction" begin
        model = ARModel(p=3)
        @test model isa ARModel
        @test model.p == 3
        @test !model.state.is_fitted
    end
    
    @testset "ARModel - construction error" begin
        @test_throws ArgumentError ARModel(p=0)
        @test_throws ArgumentError ARModel(p=-1)
    end
    
    @testset "ARModel - min_train_size" begin
        model1 = ARModel(p=1)
        @test min_train_size(model1) == 2
        
        model2 = ARModel(p=3)
        @test min_train_size(model2) == 4
        
        model3 = ARModel(p=5)
        @test min_train_size(model3) == 6
    end
    
    @testset "ARModel - create_ar_matrix" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0])
        X, y = TimeSeriesKit.Models.Autoregressive.create_ar_matrix(ts, 2)
        
        # Should have n-p rows
        @test size(X, 1) == 3
        @test size(X, 2) == 3  # intercept + 2 lags
        @test length(y) == 3
        
        # Check intercept column
        @test X[:, 1] == [1.0, 1.0, 1.0]
        
        # Check y values
        @test y == [3.0, 4.0, 5.0]
    end
    
    @testset "ARModel - create_ar_matrix too short" begin
        ts = TimeSeries([1.0, 2.0, 3.0])
        @test_throws ArgumentError TimeSeriesKit.Models.Autoregressive.create_ar_matrix(ts, 3)
        @test_throws ArgumentError TimeSeriesKit.Models.Autoregressive.create_ar_matrix(ts, 5)
    end
    
    @testset "ARModel - fit AR(1)" begin
        # Create AR(1) data: y_t = 0.8 * y_{t-1} + noise
        ts = TimeSeries([1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768])
        
        model = ARModel(p=1)
        fit(model, ts)
        
        @test model.state.is_fitted
        @test haskey(model.state.parameters, :intercept)
        @test haskey(model.state.parameters, :coefficients)
        @test length(model.state.parameters[:coefficients]) == 1
    end
    
    @testset "ARModel - fit AR(2)" begin
        ts = TimeSeries(1.0:10.0 |> collect)
        
        model = ARModel(p=2)
        fit(model, ts)
        
        @test model.state.is_fitted
        @test length(model.state.parameters[:coefficients]) == 2
        @test length(model.state.residuals) > 0
    end
    
    @testset "ARModel - predict" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        model = ARModel(p=2)
        fit(model, ts)
        
        # Predict at specific x values
        pred = predict(model, [7.0, 8.0])
        
        @test pred isa TimeSeries
        @test length(pred) == 2
        @test pred.timestamps == [7.0, 8.0]
    end
    
    @testset "ARModel - predict before fit" begin
        model = ARModel(p=2)
        @test_throws ErrorException predict(model, [1.0, 2.0])
    end
    
    @testset "ARModel - iterative_predict" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0])
        model = ARModel(p=2)
        fit(model, ts)
        
        # Iterative multi-step prediction
        pred = iterative_predict(model, ts, 3)
        
        @test pred isa TimeSeries
        @test length(pred) >= 5
    end
end
