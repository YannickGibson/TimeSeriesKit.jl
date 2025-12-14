# Model prediction functions (predict at specific x values)

using ..TimeSeriesKit: AbstractTimeSeriesModel, TimeSeries, ARModel, LinearModel, SESModel
using Statistics

"""
    predict(model::AbstractTimeSeriesModel, x_values::Vector{<:Real})

Generate predictions at specific x values from a fitted model.
"""
function predict end

"""
    predict(model::LinearModel, x_values::Vector{<:Real})

Generate predictions at specific x values using a fitted linear model.

Returns a TimeSeries with the predictions and x values as timestamps.
"""
function predict(model::LinearModel, x_values::Vector{<:Real})
    if !model.state.is_fitted
        throw(ErrorException("Model must be fitted before prediction"))
    end
    
    # Get model parameters
    intercept = model.state.parameters[:intercept]
    slope = model.state.parameters[:slope]
    
    # Generate predictions: y = intercept + slope * x
    predictions = intercept .+ slope .* x_values
    
    # Return as TimeSeries with x_values as timestamps
    return TimeSeries(x_values, predictions)
end

"""
    predict(model::SESModel, x_values::Vector{<:Real})

Generate predictions at specific x values using a fitted SES model.
For SES, all predictions are the same (the final level).

Returns a TimeSeries with the predictions and x values as timestamps.
"""
function predict(model::SESModel, x_values::Vector{<:Real})
    if !model.state.is_fitted
        throw(ErrorException("Model must be fitted before prediction"))
    end
    
    # Get model parameters - SES level is constant for all predictions
    level = model.state.parameters[:level]
    
    # Generate predictions: all equal to the level
    predictions = fill(level, length(x_values))
    
    # Return as TimeSeries with x_values as timestamps
    return TimeSeries(x_values, predictions)
end

# Export the predict function
export predict
