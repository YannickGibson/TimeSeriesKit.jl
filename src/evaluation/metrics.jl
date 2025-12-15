# Evaluation metrics for time series forecasting

using ..TimeSeriesKit: AbstractTimeSeriesModel, TimeSeries
using Statistics

"""
    _get_matching_values(ts1::TimeSeries, ts2::TimeSeries)

Private helper function to extract matching values from two TimeSeries.
Returns two vectors of values where timestamps match.
"""
function _get_matching_values(ts1::TimeSeries, ts2::TimeSeries)
    matching_indices_1 = Int[]
    matching_indices_2 = Int[]
    
    for (i, ts_1) in enumerate(ts1.timestamps)
        for (j, ts_2) in enumerate(ts2.timestamps)
            if ts_1 == ts_2
                push!(matching_indices_1, i)
                push!(matching_indices_2, j)
                break
            end
        end
    end
    
    if isempty(matching_indices_1)
        throw(ArgumentError("No matching timestamps found between the two TimeSeries"))
    end
    
    return ts1.values[matching_indices_1], ts2.values[matching_indices_2]
end

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
    mse(actual::TimeSeries, predicted::TimeSeries)

Calculate Mean Squared Error between two TimeSeries objects.
Only considers points where timestamps match on the x-axis.
"""
function mse(actual::TimeSeries, predicted::TimeSeries)
    actual_values, pred_values = _get_matching_values(actual, predicted)
    return mse(actual_values, pred_values)
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
    mae(actual::TimeSeries, predicted::TimeSeries)

Calculate Mean Absolute Error between two TimeSeries objects.
Only considers points where timestamps match on the x-axis.
"""
function mae(actual::TimeSeries, predicted::TimeSeries)
    actual_values, pred_values = _get_matching_values(actual, predicted)
    return mae(actual_values, pred_values)
end

"""
    rmse(actual::Vector{<:Real}, predicted::Vector{<:Real})

Calculate Root Mean Squared Error.
"""
function rmse(actual::Vector{<:Real}, predicted::Vector{<:Real})
    return sqrt(mse(actual, predicted))
end

"""
    rmse(actual::TimeSeries, predicted::TimeSeries)

Calculate Root Mean Squared Error between two TimeSeries objects.
Only considers points where timestamps match on the x-axis.
"""
function rmse(actual::TimeSeries, predicted::TimeSeries)
    return sqrt(mse(actual, predicted))
end

# Export all metrics
export mse, mae, rmse
