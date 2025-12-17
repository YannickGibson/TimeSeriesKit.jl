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
    
    # Constructor with UnitRange for x values
    function TimeSeries(x_range::UnitRange, y_values::Vector{T}; name::String="") where T<:Real
        x_values = collect(x_range)
        if length(x_values) != length(y_values)
            throw(ArgumentError("Length of x_range and y_values must match"))
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

"""
    WhiteNoise(length::Int; mean::Real=0.0, variance::Real=1.0, name::String="White Noise")

Generate a white noise time series with specified length, mean, and variance.

White noise is a sequence of uncorrelated random variables: ε_t ~ N(mean, variance)

# Arguments
- `length::Int`: Number of time steps
- `mean::Real=0.0`: Mean of the white noise
- `variance::Real=1.0`: Variance of the white noise
- `name::String="White Noise"`: Name for the time series

# Returns
- `TimeSeries`: A time series containing white noise

# Example
```julia
wn = WhiteNoise(100, mean=0.0, variance=5.0)
```
"""
function WhiteNoise(length::Int; mean::Real=0.0, variance::Real=1.0, name::String="White Noise")
    if length < 1
        throw(ArgumentError("Length must be at least 1"))
    end
    if variance <= 0
        throw(ArgumentError("Variance must be positive"))
    end
    
    # Generate white noise with specified mean and variance
    std_dev = sqrt(variance)
    noise = randn(length) .* std_dev .+ mean
    
    return TimeSeries(noise; name=name)
end

"""
    ARProcess(length::Int; phi::Real=0.5, constant::Real=0.0, noise_variance::Real=1.0, name::String="AR(1) Process")

Generate an AR(1) autoregressive time series.

The AR(1) process is generated as: y_t = constant - phi * y_{t-1} + ε_t, where ε_t ~ N(0, noise_variance)

# Arguments
- `length::Int`: Number of time steps
- `phi::Real=0.5`: AR(1) coefficient (typically -1 < phi < 1 for stationarity)
- `constant::Real=0.0`: Constant term in the AR process
- `noise_variance::Real=1.0`: Variance of the white noise innovations
- `name::String="AR(1) Process"`: Name for the time series

# Returns
- `TimeSeries`: A time series containing the AR(1) process

# Example
```julia
ar = ARProcess(5000, phi=0.85, constant=10.0, noise_variance=1.0)
```
"""
function ARProcess(length::Int; phi::Real=0.5, constant::Real=0.0, noise_variance::Real=1.0, name::String="AR(1) Process")
    if length < 1
        throw(ArgumentError("Length must be at least 1"))
    end
    if noise_variance <= 0
        throw(ArgumentError("Noise variance must be positive"))
    end
    
    # Generate AR(1) process
    data = zeros(length)
    std_dev = sqrt(noise_variance)
    data[1] = constant  # Initialize first value
    
    for t in 2:length
        data[t] = constant - phi * data[t-1] + randn() * std_dev
    end
    
    return TimeSeries(data; name=name)
end

"""
    RandomWalk(length::Int; mean::Real=0.0, variance::Real=1.0, name::String="Random Walk")

Generate a random walk time series with specified length, mean, and variance.

The random walk is generated as: y_t = y_{t-1} + ε_t, where ε_t ~ N(mean, variance)

# Arguments
- `length::Int`: Number of time steps
- `mean::Real=0.0`: Mean of the random increments
- `variance::Real=1.0`: Variance of the random increments
- `name::String="Random Walk"`: Name for the time series

# Returns
- `TimeSeries`: A time series containing the random walk

# Example
```julia
rw = RandomWalk(100, mean=0.5, variance=2.0)
```
"""
function RandomWalk(length::Int; mean::Real=0.0, variance::Real=1.0, name::String="Random Walk")
    if length < 1
        throw(ArgumentError("Length must be at least 1"))
    end
    if variance <= 0
        throw(ArgumentError("Variance must be positive"))
    end
    
    # Generate random increments with specified mean and variance
    std_dev = sqrt(variance)
    increments = randn(length) .* std_dev .+ mean
    
    # Compute cumulative sum to get random walk
    walk = cumsum(increments)
    
    return TimeSeries(walk; name=name)
end

Base.length(ts::TimeSeries) = length(ts.values)
Base.lastindex(ts::TimeSeries) = lastindex(ts.values)

# Indexing support - returns scalar for single index
Base.getindex(ts::TimeSeries, i::Int) = ts.values[i]

# Indexing support - returns new TimeSeries for range/vector indices
function Base.getindex(ts::TimeSeries{T}, indices) where T
    new_timestamps = ts.timestamps[indices]
    new_values = ts.values[indices]
    new_name = ts.name == "" ? "" : "$(ts.name) [subset]"
    return TimeSeries(new_timestamps, new_values; name=new_name)
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
