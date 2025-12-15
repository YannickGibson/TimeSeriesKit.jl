# Simple Exponential Smoothing (SES) Model

using ..TimeSeriesKit: AbstractTimeSeriesModel, ModelState, TimeSeries
using Statistics

"""
    SESModel

Simple Exponential Smoothing model: ŷ_{t+1} = αy_t + (1-α)ŷ_t

The smoothing parameter α ∈ (0, 1) controls the weight given to recent observations.
"""
mutable struct SESModel <: AbstractTimeSeriesModel
    alpha::Union{Float64, Nothing}  # Smoothing parameter (if Nothing, will be optimized)
    state::ModelState
    
    function SESModel(; alpha::Union{Float64, Nothing}=nothing)
        if alpha !== nothing && (alpha <= 0.0 || alpha >= 1.0)
            throw(ArgumentError("Alpha must be between 0 and 1"))
        end
        new(alpha, ModelState())
    end
end

"""
    ses_forecast(values::Vector{<:Real}, alpha::Float64, horizon::Int)

Generate forecasts using simple exponential smoothing.
"""
function ses_forecast(values::Vector{<:Real}, alpha::Float64, horizon::Int)
    n = length(values)
    
    # Initialize with first value
    level = values[1]
    fitted = zeros(n)
    fitted[1] = level
    
    # Compute fitted values
    for t in 2:n
        level = alpha * values[t-1] + (1 - alpha) * level
        fitted[t] = level
    end
    
    # Forecast future values (constant forecast in SES)
    final_level = alpha * values[n] + (1 - alpha) * level
    forecasts = fill(final_level, horizon)
    
    return fitted, forecasts, final_level
end

# Export the model type
export SESModel
