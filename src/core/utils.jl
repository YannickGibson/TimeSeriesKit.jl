# Utility functions for TimeSeriesKit

"""
    validate_timeseries(ts::TimeSeries)

Validate that a time series is properly formed.
"""
function validate_timeseries(ts::TimeSeries)
    if length(ts.values) == 0
        throw(ArgumentError("Time series cannot be empty"))
    end
    if any(isnan, ts.values) || any(isinf, ts.values)
        @warn "Time series contains NaN or Inf values"
    end
    return true
end

"""
    split_train_test(ts::TimeSeries, train_ratio::Float64=0.8)

Split time series into training and test sets.
"""
function split_train_test(ts::TimeSeries, train_ratio::Float64=0.8)
    if train_ratio <= 0.0 || train_ratio >= 1.0
        throw(ArgumentError("train_ratio must be between 0 and 1"))
    end
    
    n = length(ts)
    train_size = floor(Int, n * train_ratio)
    
    train_values = ts.values[1:train_size]
    test_values = ts.values[train_size+1:end]
    
    if ts.timestamps !== nothing
        train_ts = TimeSeries(train_values, ts.timestamps[1:train_size])
        test_ts = TimeSeries(test_values, ts.timestamps[train_size+1:end])
    else
        train_ts = TimeSeries(train_values)
        test_ts = TimeSeries(test_values)
    end
    
    return train_ts, test_ts
end

"""
    difference(ts::TimeSeries, order::Int=1)

Compute differences of a time series.
"""
function difference(ts::TimeSeries, order::Int=1)
    if order < 1
        throw(ArgumentError("Order must be at least 1"))
    end
    
    values = copy(ts.values)
    for _ in 1:order
        values = diff(values)
    end
    
    return TimeSeries(values)
end

"""
    normalize(ts::TimeSeries; method::Symbol=:zscore)

Normalize time series using z-score or min-max scaling.
"""
function normalize(ts::TimeSeries; method::Symbol=:zscore)
    if method == :zscore
        μ = mean(ts.values)
        σ = std(ts.values)
        normalized = (ts.values .- μ) ./ σ
    elseif method == :minmax
        min_val = minimum(ts.values)
        max_val = maximum(ts.values)
        normalized = (ts.values .- min_val) ./ (max_val - min_val)
    else
        throw(ArgumentError("Unknown normalization method: $method"))
    end
    
    return TimeSeries(normalized, ts.timestamps)
end
