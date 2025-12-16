# Core type definitions for TimeSeriesKit

using CSV
using DataFrames

"""
    TimeSeries

A time series data structure containing values and timestamps.

# Constructors
- `TimeSeries(values::Vector{T}; name::String="")`: Creates a TimeSeries with values only, timestamps are automatically 1:n
- `TimeSeries(x_values::Vector, y_values::Vector{T}; name::String="")`: Creates a TimeSeries with x values (timestamps) and y values
- `TimeSeries(csv_path::String, country::String; name::String="")`: Loads SDMX-CSV 1.0 format emissions data from a CSV file for the specified country
"""
struct TimeSeries{T<:Real}
    timestamps::Vector{<:Any}
    values::Vector{T}
    name::String
    
    # Constructor with only y values - automatically creates x values from 1 to n
    function TimeSeries(values::Vector{T}; name::String="") where T<:Real
        timestamps = collect(1:length(values))
        new{T}(timestamps, values, name)
    end
    
    # Constructor with x values first, then y values
    function TimeSeries(x_values::Vector, y_values::Vector{T}; name::String="") where T<:Real
        if length(x_values) != length(y_values)
            throw(ArgumentError("Length of x_values and y_values must match"))
        end
        new{T}(x_values, y_values, name)
    end
    
    # Constructor from CSV file path - loads SDMX-CSV 1.0 format emissions data
    function TimeSeries(csv_path::String, country::String; name::String="")
        # Load SDMX-CSV 1.0 format
        df = CSV.read(csv_path, DataFrame)
        
        # Filter by country
        country_data = df[df.geo .== country, :]
        
        if nrow(country_data) == 0
            throw(ArgumentError("No data found for country: $country"))
        end
        
        # Sort by year to ensure time series order
        sort!(country_data, :TIME_PERIOD)
        
        # Extract years and emissions values
        years = country_data.TIME_PERIOD
        data = Float32.(country_data.OBS_VALUE)
        
        # Use provided name or default to country name
        ts_name = isempty(name) ? country : name
        
        new{Float32}(years, data, ts_name)
    end
end

Base.length(ts::TimeSeries) = length(ts.values)
Base.getindex(ts::TimeSeries, i) = ts.values[i]
Base.lastindex(ts::TimeSeries) = lastindex(ts.values)

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
    
    new_name = ts.name == "" ? "Differenced" : "$(ts.name) (d=$order)"
    return TimeSeries(timestamps, values; name=new_name)
end

"""
    integrate(ts::TimeSeries)

Compute the cumulative sum (integration) of a time series.

# Arguments
- `ts::TimeSeries`: The time series to integrate

# Returns
- A new `TimeSeries` with integrated values: y_t = sum(x_1 to x_t)

# Example
```julia
ts = TimeSeries([1.0, 2.0, 3.0, 4.0])
int_ts = integrate(ts)  # Returns [1.0, 3.0, 6.0, 10.0]
```
"""
function integrate(ts::TimeSeries)
    values = cumsum(ts.values)
    new_name = ts.name == "" ? "Integrated" : "$(ts.name) (integrated)"
    return TimeSeries(ts.timestamps, values; name=new_name)
end

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
