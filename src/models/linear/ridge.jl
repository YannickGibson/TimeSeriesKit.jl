# Ridge Regression Model for Time Series

using ..TimeSeriesKit: AbstractTimeSeriesModel, ModelState, TimeSeries
using LinearAlgebra
using Statistics

"""
    RidgeModel

Ridge regression model for time series: y_t = a + b*t + ε_t with L2 regularization.

The regularization parameter λ controls the amount of shrinkage applied to coefficients.
Higher λ values lead to more regularization.
"""
mutable struct RidgeModel <: AbstractTimeSeriesModel
    λ::Float64  # Regularization parameter
    sliding_window::Int  # Window size for training
    state::ModelState
    
    function RidgeModel(; λ::Float64=1.0, sliding_window::Int=5)
        if λ < 0.0
            throw(ArgumentError("Lambda must be non-negative"))
        end
        if sliding_window < 2
            throw(ArgumentError("Sliding window must be at least 2"))
        end
        new(λ, sliding_window, ModelState())
    end
end

# Minimum training size implementation
TimeSeriesKit.Models.min_train_size(model::RidgeModel) = max(2, model.sliding_window)

# Export the model type
export RidgeModel
