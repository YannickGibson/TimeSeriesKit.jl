# Core type definitions for TimeSeriesKit

"""
    TimeSeries

A time series data structure containing values and timestamps.

# Constructors
- `TimeSeries(values::Vector{T})`: Creates a TimeSeries with values only, timestamps are automatically 1:n
- `TimeSeries(x_values::Vector, y_values::Vector{T})`: Creates a TimeSeries with x values (timestamps) and y values
"""
struct TimeSeries{T<:Real}
    timestamps::Vector{<:Any}
    values::Vector{T}
    
    # Constructor with only y values - automatically creates x values from 1 to n
    function TimeSeries(values::Vector{T}) where T<:Real
        timestamps = collect(1:length(values))
        new{T}(timestamps, values)
    end
    
    # Constructor with x values first, then y values
    function TimeSeries(x_values::Vector, y_values::Vector{T}) where T<:Real
        if length(x_values) != length(y_values)
            throw(ArgumentError("Length of x_values and y_values must match"))
        end
        new{T}(x_values, y_values)
    end
end

Base.length(ts::TimeSeries) = length(ts.values)
Base.getindex(ts::TimeSeries, i) = ts.values[i]
Base.lastindex(ts::TimeSeries) = lastindex(ts.values)

"""
    AbstractTimeSeriesModel

Abstract base type for all time series models.
"""
abstract type AbstractTimeSeriesModel end

"""
    ModelState

Stores the fitted state of a time series model.
"""
mutable struct ModelState
    parameters::Dict{Symbol, Any}
    fitted_values::Union{Vector{<:Real}, Nothing}
    residuals::Union{Vector{<:Real}, Nothing}
    is_fitted::Bool
    
    ModelState() = new(Dict{Symbol, Any}(), nothing, nothing, false)
end
