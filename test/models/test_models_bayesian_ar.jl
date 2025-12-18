using Test
using TimeSeriesKit
using Statistics
using LinearAlgebra

@testset "Bayesian AR Models" begin
    
    @testset "BayesianARModel - construction" begin
        model = BayesianARModel(p=2)
        @test model isa BayesianARModel
        @test model.p == 2
        @test model.prior_variance == 1000.0
        @test length(model.prior_mean) == 3  # p + 1 for intercept
        @test all(model.prior_mean .== 0.0)  # Default: zero mean
        @test !model.state.is_fitted
        
        # Custom prior
        model2 = BayesianARModel(p=3, prior_mean=[0.0, 0.7, 0.2, 0.1], prior_variance=0.5)
        @test model2.p == 3
        @test model2.prior_mean == [0.0, 0.7, 0.2, 0.1]
        @test model2.prior_variance == 0.5
    end
    
    @testset "BayesianARModel - construction errors" begin
        @test_throws ArgumentError BayesianARModel(p=0)
        @test_throws ArgumentError BayesianARModel(p=-1)
        @test_throws ArgumentError BayesianARModel(p=2, prior_variance=-0.1)
        @test_throws ArgumentError BayesianARModel(p=2, prior_variance=0.0)
        @test_throws ArgumentError BayesianARModel(p=2, prior_mean=[0.0, 0.5])  # Wrong length
    end
    
    @testset "BayesianARModel - min_train_size" begin
        model = BayesianARModel(p=1)
        @test min_train_size(model) == 2  # p + 1
        
        model2 = BayesianARModel(p=3)
        @test min_train_size(model2) == 4  # p + 1
    end
    
    @testset "BayesianARModel - fit with AR(1) process" begin
        # Simulate AR(1): y_t = 0.5 + 0.7*y_{t-1} + ε_t
        n = 50
        values = zeros(n)
        values[1] = 1.0
        for i in 2:n
            values[i] = 0.5 + 0.7 * values[i-1] + 0.1 * randn()
        end
        ts = TimeSeries(collect(1.0:n), values)
        
        model = BayesianARModel(p=1, prior_variance=1000.0)
        fit(model, ts)
        
        @test model.state.is_fitted
        @test haskey(model.state.parameters, :intercept)
        @test haskey(model.state.parameters, :coefficients)
        @test haskey(model.state.parameters, :residual_variance)
        @test haskey(model.state.parameters, :posterior_covariance)
        @test haskey(model.state.parameters, :intercept_variance)
        @test haskey(model.state.parameters, :coefficient_variances)
        
        # Check coefficients are reasonable
        @test length(model.state.parameters[:coefficients]) == 1
        @test model.state.parameters[:intercept] ≈ 0.5 atol=0.5
        @test model.state.parameters[:coefficients][1] ≈ 0.7 atol=0.3
        
        # Check variances are positive
        @test model.state.parameters[:residual_variance] > 0
        @test model.state.parameters[:intercept_variance] > 0
        @test model.state.parameters[:coefficient_variances][1] > 0
    end
    
    @testset "BayesianARModel - fit with AR(2) process" begin
        # Simple AR(2) data
        n = 30
        values = zeros(n)
        values[1] = 1.0
        values[2] = 1.5
        for i in 3:n
            values[i] = 1.0 + 0.5 * values[i-1] + 0.3 * values[i-2] + 0.05 * randn()
        end
        ts = TimeSeries(collect(1.0:n), values)
        
        model = BayesianARModel(p=2)
        fit(model, ts)
        
        @test model.state.is_fitted
        @test length(model.state.parameters[:coefficients]) == 2
        @test length(model.state.parameters[:coefficient_variances]) == 2
        
        # Check covariance matrix dimensions
        Σ = model.state.parameters[:posterior_covariance]
        @test size(Σ) == (3, 3)  # intercept + 2 coefficients
        @test issymmetric(Σ)
    end
    
    @testset "BayesianARModel - predict without uncertainty" begin
        # Simple AR(1) data
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 2.5, 2.8, 3.0, 3.1])
        model = BayesianARModel(p=1)
        fit(model, ts)
        
        # Predict at future points
        predictions = predict(model, [6.0, 7.0, 8.0])
        
        @test predictions isa TimeSeries
        @test !(predictions isa PredictionResult)
        @test length(predictions) == 3
        @test predictions.timestamps == [6.0, 7.0, 8.0]
        @test all(.!isnan.(predictions.values))
    end
    
    @testset "BayesianARModel - predict with uncertainty" begin
        # AR(1) data with some variation
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 2.5, 2.9, 3.2, 3.6, 3.9])
        model = BayesianARModel(p=1, prior_variance=100.0)
        fit(model, ts)
        
        # Predict with uncertainty
        result = predict(model, [7.0, 8.0, 9.0], return_uncertainty=true)
        
        @test result isa PredictionResult
        @test result.predictions isa TimeSeries
        @test result.prediction_variance !== nothing
        @test result.prediction_std !== nothing
        
        @test length(result.predictions) == 3
        @test length(result.prediction_variance) == 3
        @test length(result.prediction_std) == 3
        
        # Check variance properties
        @test all(result.prediction_variance .> 0)
        @test all(result.prediction_std .> 0)
        @test result.prediction_std ≈ sqrt.(result.prediction_variance)
    end
    
    @testset "BayesianARModel - predict before fit" begin
        model = BayesianARModel(p=1)
        @test_throws ErrorException predict(model, [1.0, 2.0])
    end
    
    @testset "BayesianARModel - prior_variance effect" begin
        # Same data, different prior variances
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 2.5, 2.9, 3.2, 3.6, 3.9])
        
        # Weak prior (high variance - more like OLS)
        model_weak = BayesianARModel(p=1, prior_variance=10000.0)
        fit(model_weak, ts)
        
        # Strong prior (low variance - more regularization)
        model_strong = BayesianARModel(p=1, prior_variance=0.1)
        fit(model_strong, ts)
        
        # Strong prior should pull coefficients closer to zero (prior mean)
        # and give less influence to the data
        @test abs(model_strong.state.parameters[:coefficients][1]) <= 
              abs(model_weak.state.parameters[:coefficients][1])
    end
    
    @testset "BayesianARModel - uncertainty includes parameter and residual variance" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 2.5, 2.8, 3.0, 3.1])
        model = BayesianARModel(p=1)
        fit(model, ts)
        
        # Get prediction with uncertainty
        result = predict(model, [6.0], return_uncertainty=true)
        
        # Prediction variance should be at least as large as residual variance
        σ²_residual = model.state.parameters[:residual_variance]
        @test result.prediction_variance[1] >= σ²_residual
    end
    
    @testset "BayesianARModel - iterative_predict without uncertainty" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 2.5, 2.9, 3.2, 3.6, 3.9])
        model = BayesianARModel(p=1)
        
        result = iterative_predict(model, ts, 3)
        
        @test result isa TimeSeries
        @test !(result isa PredictionResult)
        @test length(result) > 3  # Should have in-sample + out-of-sample predictions
        @test all(.!isnan.(result.values))
    end
    
    @testset "BayesianARModel - iterative_predict with uncertainty" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 
                       [2.0, 2.5, 2.9, 3.2, 3.6, 3.9, 4.1])
        model = BayesianARModel(p=1, prior_variance=100.0)
        
        result = iterative_predict(model, ts, 3, return_uncertainty=true)
        
        @test result isa PredictionResult
        @test result.predictions isa TimeSeries
        @test result.prediction_variance !== nothing
        @test result.prediction_std !== nothing
        
        # Should have predictions for in-sample and out-of-sample
        @test length(result.predictions) > 3
        @test length(result.prediction_variance) == length(result.predictions)
        @test length(result.prediction_std) == length(result.predictions)
        
        # All variances should be positive
        @test all(result.prediction_variance .> 0)
        @test all(result.prediction_std .> 0)
    end
    
    @testset "BayesianARModel - iterative_predict insufficient data" begin
        ts = TimeSeries([1.0, 2.0], [2.0, 2.5])
        model = BayesianARModel(p=2)  # Requires at least 3 points
        
        @test_throws ArgumentError iterative_predict(model, ts, 2)
    end
    
    @testset "BayesianARModel - posterior covariance is symmetric" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 2.5, 2.8, 3.0, 3.1])
        model = BayesianARModel(p=2)
        fit(model, ts)
        
        Σ = model.state.parameters[:posterior_covariance]
        @test issymmetric(Σ)
        
        # Check it's positive definite (all eigenvalues positive)
        @test all(eigvals(Σ) .> 0)
    end
    
    @testset "BayesianARModel - comparison with OLS ARModel" begin
        # With weak prior, Bayesian AR should be close to OLS AR
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
                       [2.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1])
        
        # Bayesian with very weak prior (high variance = weak belief)
        bayes_model = BayesianARModel(p=1, prior_variance=1e6)
        fit(bayes_model, ts)
        
        # OLS AR
        ols_model = ARModel(p=1)
        fit(ols_model, ts)
        
        # Coefficients should be very close
        @test bayes_model.state.parameters[:intercept] ≈ 
              ols_model.state.parameters[:intercept] atol=0.01
        @test bayes_model.state.parameters[:coefficients][1] ≈ 
              ols_model.state.parameters[:coefficients][1] atol=0.01
    end
    
    @testset "BayesianARModel - validate_timeseries integration" begin
        # Empty time series should throw
        ts_empty = TimeSeries(Float64[])
        model = BayesianARModel(p=1)
        @test_throws ArgumentError fit(model, ts_empty)
    end
    
    @testset "BayesianARModel - higher order AR(3)" begin
        # Test with AR(3)
        n = 40
        values = zeros(n)
        values[1:3] = [1.0, 1.2, 1.4]
        for i in 4:n
            values[i] = 0.5 + 0.4*values[i-1] + 0.3*values[i-2] + 0.2*values[i-3] + 0.05*randn()
        end
        ts = TimeSeries(collect(1.0:n), values)
        
        model = BayesianARModel(p=3, prior_variance=100.0)
        fit(model, ts)
        
        @test model.state.is_fitted
        @test length(model.state.parameters[:coefficients]) == 3
        @test length(model.state.parameters[:coefficient_variances]) == 3
        
        # Predict with uncertainty
        result = predict(model, [41.0, 42.0], return_uncertainty=true)
        @test length(result.predictions) == 2
        @test all(result.prediction_variance .> 0)
    end
    
    @testset "PredictionResult - structure for BayesianARModel" begin
        ts = TimeSeries([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        model = BayesianARModel(p=1)
        fit(model, ts)
        
        result = predict(model, [4.0, 5.0], return_uncertainty=true)
        
        @test result.predictions isa TimeSeries
        @test result.prediction_variance isa Vector
        @test result.prediction_std ≈ sqrt.(result.prediction_variance)
    end
    
    @testset "BayesianARModel - informative prior" begin
        # Simulate AR(1): y_t = 0.5 + 0.7*y_{t-1} + ε_t with small sample
        n = 15
        values = zeros(n)
        values[1] = 1.0
        for i in 2:n
            values[i] = 0.5 + 0.7 * values[i-1] + 0.1 * randn()
        end
        ts = TimeSeries(collect(1.0:n), values)
        
        # Model with informative prior: expect AR(1) ≈ 0.7, intercept ≈ 0.5
        prior_mean = [0.5, 0.7]
        prior_variance = 0.1  # Strong prior
        model_informative = BayesianARModel(p=1, prior_mean=prior_mean, prior_variance=prior_variance)
        fit(model_informative, ts)
        
        # Model with weak prior
        model_weak = BayesianARModel(p=1, prior_variance=1000.0)
        fit(model_weak, ts)
        
        # Informative prior should pull estimates toward prior mean
        # In small samples, informative prior has more influence
        @test abs(model_informative.state.parameters[:intercept] - 0.5) < 
              abs(model_weak.state.parameters[:intercept] - 0.5)
        @test abs(model_informative.state.parameters[:coefficients][1] - 0.7) < 
              abs(model_weak.state.parameters[:coefficients][1] - 0.7)
    end
end
