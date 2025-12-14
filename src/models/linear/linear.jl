# Linear Regression Model for Time Series

using ..TimeSeriesKit: AbstractTimeSeriesModel, ModelState, TimeSeries
using LinearAlgebra
using Statistics

"""
    LinearModel

Simple linear trend model: y_t = a + b*t + ε_t
"""
mutable struct LinearModel <: AbstractTimeSeriesModel
    state::ModelState
    
    LinearModel() = new(ModelState())
end

"""
    create_matrix_X(n::Int)

Create design matrix for linear trend fitting.
"""
function create_matrix_X(n::Int)
    X = zeros(n, 2)
    X[:, 1] .= 1.0  # Intercept
    X[:, 2] = 1:n    # Time trend
    return X
end

# Export the model type
export LinearModel
