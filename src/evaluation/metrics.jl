# Evaluation metrics for time series forecasting

using Statistics

"""
    mse(actual::Vector{<:Real}, predicted::Vector{<:Real})

Calculate Mean Squared Error.
"""
function mse(actual::Vector{<:Real}, predicted::Vector{<:Real})
    if length(actual) != length(predicted)
        throw(ArgumentError("actual and predicted must have the same length"))
    end
    return mean((actual .- predicted).^2)
end

"""
    mae(actual::Vector{<:Real}, predicted::Vector{<:Real})

Calculate Mean Absolute Error.
"""
function mae(actual::Vector{<:Real}, predicted::Vector{<:Real})
    if length(actual) != length(predicted)
        throw(ArgumentError("actual and predicted must have the same length"))
    end
    return mean(abs.(actual .- predicted))
end

"""
    rmse(actual::Vector{<:Real}, predicted::Vector{<:Real})

Calculate Root Mean Squared Error.
"""
function rmse(actual::Vector{<:Real}, predicted::Vector{<:Real})
    return sqrt(mse(actual, predicted))
end

# Export all metrics
export mse, mae, rmse
