# Linear Regression Model for Time Series

using ..TimeSeriesKit: AbstractTimeSeriesModel, ModelState, TimeSeries
using LinearAlgebra
using Statistics

"""
    LinearModel

Simple linear trend model: y_t = a + b*t + ε_t
"""
mutable struct LinearModel <: AbstractTimeSeriesModel
    sliding_window::Int  # For training -> predictions
    state::ModelState
    
    function LinearModel(; sliding_window::Int=5)
        if sliding_window < 2
            throw(ArgumentError("Sliding window must be at least 2"))
        end
        new(sliding_window, ModelState())
    end
end

"""
    create_matrix_X(x_values::Vector{<:Real})

Create design matrix for linear trend fitting.
"""
function create_matrix_X(x_values::Vector{<:Real})
    X = zeros(length(x_values), 2)
    X[:, 1] .= 1.0  # Intercept
    X[:, 2] = x_values  # Time trend
    return X
end

# Minimum training size implementations
TimeSeriesKit.Models.min_train_size(model::LinearModel) = max(2, model.sliding_window)

# Export the model type
export LinearModel
