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

"""
    differentiate(ts::TimeSeries; order::Int=1)

Compute the discrete difference of a time series.

# Arguments
- `ts::TimeSeries`: The time series to differentiate
- `order::Int=1`: The order of differentiation (how many times to apply the difference)

# Returns
- A new `TimeSeries` with differenced values: y_t = x_t - x_{t-1}

# Example
```julia
ts = TimeSeries([1.0, 3.0, 6.0, 10.0])
diff_ts = differentiate(ts)  # Returns [2.0, 3.0, 4.0]
```
"""
function differentiate(ts::TimeSeries; order::Int=1)
    if order < 1
        throw(ArgumentError("Order must be at least 1"))
    end
    
    values = ts.values
    timestamps = ts.timestamps
    
    for _ in 1:order
        if length(values) < 2
            throw(ArgumentError("Time series too short to differentiate"))
        end
        values = diff(values)
        timestamps = timestamps[2:end]
    end
    
    if ts.name == ""
        new_name = "Differentiated"
    else
        if order == 1
            new_name = "$(ts.name) (Differentiated)"
        else
            new_name = "$(ts.name) (Differentiated $(order) times)"
        end
    end
    return TimeSeries(timestamps, values; name=new_name)
end

"""
     integrate(ts::TimeSeries; order::Int=1)

Compute the cumulative sum (integration) of a time series.

# Arguments
- `ts::TimeSeries`: The time series to integrate
- `order::Int=1`: The order of integration (how many times to apply cumulative sum)

# Returns
- A new `TimeSeries` with integrated values: y_t = sum(x_1 to x_t)

# Example
```julia
ts = TimeSeries([1.0, 2.0, 3.0, 4.0])
int_ts = integrate(ts)  # Returns [1.0, 3.0, 6.0, 10.0]
int_ts2 = integrate(ts, order=2)  # Integrate twice
```
"""
function integrate(ts::TimeSeries; order::Int=1)
    values = ts.values
    for i in 1:order
        values = cumsum(values)
    end
    
    if ts.name == ""
        new_name = "Integrated"
    else
        if order == 1
            new_name = "$(ts.name) (Integrated)"
        else
            new_name = "$(ts.name) (Integrated $(order) times)"
        end
    end 

    return TimeSeries(ts.timestamps, values; name=new_name)
end
