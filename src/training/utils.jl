# Utility functions for training

using ..TimeSeriesKit: TimeSeries

"""
    extrapolate_timestamps(ts::TimeSeries, horizon::Int)

Generate future timestamps by extrapolating the step difference between the last two timestamps.

# Arguments
- `ts::TimeSeries`: The input time series
- `horizon::Int`: Number of future timestamps to generate

# Returns
- `Vector`: A vector of extrapolated timestamps
"""
function extrapolate_timestamps(ts::TimeSeries, horizon::Int)
    step = ts.timestamps[end] - ts.timestamps[end-1]
    return [ts.timestamps[end] + step * h for h in 1:horizon]
end

export extrapolate_timestamps
