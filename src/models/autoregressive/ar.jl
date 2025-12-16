# Autoregressive (AR) Model

using ..TimeSeriesKit: AbstractTimeSeriesModel, ModelState, TimeSeries
using LinearAlgebra
using Statistics

"""
    ARModel

Autoregressive model of order p: y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} + ε_t
"""
mutable struct ARModel <: AbstractTimeSeriesModel
    p::Int  # Order of the AR model
    state::ModelState
    
    function ARModel(; p::Int)
        if p < 1
            throw(ArgumentError("AR order must be at least 1"))
        end
        new(p, ModelState())
    end
end

"""
    create_ar_matrix(ts::TimeSeries, p::Int)

Create design matrix for AR model fitting.
"""
function create_ar_matrix(ts::TimeSeries, p::Int)
    n = length(ts)
    if n <= p
        throw(ArgumentError("Time series too short for AR($p) model"))
    end
    
    # Create lagged design matrix
    X = zeros(n - p, p + 1)  # +1 for intercept
    y = zeros(n - p)
    
    for i in 1:(n - p)
        X[i, 1] = 1.0  # Intercept
        for j in 1:p
            X[i, j + 1] = ts.values[p + i - j]
        end
        y[i] = ts.values[p + i]
    end
    
    return X, y
end

# Minimum training size implementation
TimeSeriesKit.Models.min_train_size(model::ARModel) = model.p * 2  # Avoid X'X singularity

# Export the model type
export ARModel
