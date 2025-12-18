using Test
using TimeSeriesKit

@testset "Abstract Model Functions" begin
    
    @testset "is_fitted - LinearModel" begin
        model = LinearModel()
        
        # Before fitting
        @test is_fitted(model) == false
        
        # After fitting
        ts = TimeSeries([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        fit(model, ts)
        @test is_fitted(model) == true
    end
    
    @testset "is_fitted - RidgeModel" begin
        model = RidgeModel()
        
        @test is_fitted(model) == false
        
        ts = TimeSeries([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        fit(model, ts)
        @test is_fitted(model) == true
    end
    
    @testset "get_parameters - before fit" begin
        model = LinearModel()
        @test_throws ErrorException get_parameters(model)
        
        model2 = RidgeModel()
        @test_throws ErrorException get_parameters(model2)
    end
    
    @testset "get_parameters - LinearModel after fit" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0], [5.0, 8.0, 11.0, 14.0])
        model = LinearModel()
        fit(model, ts)
        
        params = get_parameters(model)
        @test params isa Dict
        @test haskey(params, :intercept)
        @test haskey(params, :slope)
        @test params[:intercept] ≈ 2.0 atol=1e-10
        @test params[:slope] ≈ 3.0 atol=1e-10
    end
    
    @testset "get_parameters - RidgeModel after fit" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0], [5.0, 8.0, 11.0, 14.0])
        model = RidgeModel(λ=0.01)
        fit(model, ts)
        
        params = get_parameters(model)
        @test params isa Dict
        @test haskey(params, :intercept)
        @test haskey(params, :slope)
        @test haskey(params, :λ)
        @test params[:λ] == 0.01
    end
    
    @testset "get_residuals - before fit" begin
        model = LinearModel()
        @test_throws ErrorException get_residuals(model)
        
        model2 = RidgeModel()
        @test_throws ErrorException get_residuals(model2)
    end
    
    @testset "get_residuals - LinearModel after fit" begin
        # Perfect fit - residuals should be zero
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0], [5.0, 8.0, 11.0, 14.0])
        model = LinearModel()
        fit(model, ts)
        
        residuals = get_residuals(model)
        @test residuals isa Vector
        @test length(residuals) == 4
        @test all(abs.(residuals) .< 1e-10)
    end
    
    @testset "get_residuals - noisy data" begin
        # Noisy data - residuals should not be zero
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0], [5.1, 7.9, 11.2, 13.8])
        model = LinearModel()
        fit(model, ts)
        
        residuals = get_residuals(model)
        @test length(residuals) == 4
        @test !all(abs.(residuals) .< 1e-10)
        
        # Check that residuals sum to approximately zero
        @test abs(sum(residuals)) < 0.5
    end
    
    @testset "get_residuals - RidgeModel after fit" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0], [5.0, 8.0, 11.0, 14.0])
        model = RidgeModel(λ=0.01)
        fit(model, ts)
        
        residuals = get_residuals(model)
        @test residuals isa Vector
        @test length(residuals) == 4
    end
    
    @testset "min_train_size - LinearModel" begin
        model1 = LinearModel(sliding_window=5)
        @test min_train_size(model1) == 5
        
        model2 = LinearModel(sliding_window=10)
        @test min_train_size(model2) == 10
        
        model3 = LinearModel(sliding_window=2)
        @test min_train_size(model3) == 2
    end
    
    @testset "min_train_size - RidgeModel" begin
        model1 = RidgeModel(sliding_window=5)
        @test min_train_size(model1) == 5
        
        model2 = RidgeModel(sliding_window=7)
        @test min_train_size(model2) == 7
    end
    
    @testset "is_fitted persistence after refit" begin
        model = LinearModel()
        ts1 = TimeSeries([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        
        # First fit
        fit(model, ts1)
        @test is_fitted(model) == true
        
        # Refit
        ts2 = TimeSeries([1.0, 2.0, 3.0], [3.0, 5.0, 7.0])
        fit(model, ts2)
        @test is_fitted(model) == true
    end
    
    @testset "Parameters update after refit" begin
        model = LinearModel()
        
        # First fit
        ts1 = TimeSeries([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        fit(model, ts1)
        params1 = deepcopy(get_parameters(model))
        
        # Second fit with different data
        ts2 = TimeSeries([1.0, 2.0, 3.0], [7.0, 5.0, 3.0])
        fit(model, ts2)
        params2 = get_parameters(model)
        
        # Parameters should be different
        @test params1[:slope] != params2[:slope]
    end
    
    @testset "Residuals update after refit" begin
        model = LinearModel()
        
        # First fit - perfect fit
        ts1 = TimeSeries([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        fit(model, ts1)
        residuals1 = get_residuals(model)
        
        # Second fit - noisy data
        ts2 = TimeSeries([1.0, 2.0, 3.0], [2.1, 3.9, 6.2])
        fit(model, ts2)
        residuals2 = get_residuals(model)
        
        # Residuals should be different
        @test residuals1 != residuals2
        @test all(abs.(residuals1) .< 1e-10)
        @test !all(abs.(residuals2) .< 1e-10)
    end
    
    @testset "min_train_size - not implemented for base type" begin
        # Create a minimal model that doesn't implement min_train_size
        mutable struct DummyModel <: AbstractTimeSeriesModel
            state::ModelState
            DummyModel() = new(ModelState())
        end
        
        dummy = DummyModel()
        @test_throws ErrorException min_train_size(dummy)
    end
end
