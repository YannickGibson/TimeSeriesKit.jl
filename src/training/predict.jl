# Model prediction functions (predict at specific x values)

using ..TimeSeriesKit: AbstractTimeSeriesModel, TimeSeries, ARModel, MAModel, LinearModel, RidgeModel, SESModel
using Statistics

"""
    predict(model::AbstractTimeSeriesModel, x_values::Vector{<:Real})

Generate predictions at specific x values from a fitted model.
"""
function predict end

"""
    predict(model::Union{LinearModel, RidgeModel}, x_values::Vector{<:Real})

Generate predictions at specific x values using a fitted linear or ridge regression model.

Returns a TimeSeries with the predictions and x values as timestamps.
"""
function predict(model::Union{LinearModel, RidgeModel}, x_values::Vector{<:Real})
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

"""
    predict(model::ARModel, x_values::Vector{<:Real})

Generate predictions at specific x values using a fitted AR model.
Note: For AR models, this generates constant forecasts based on the last observed values.
For multi-step ahead forecasts, use iterative_predict instead.

Returns a TimeSeries with the predictions and x values as timestamps.
"""
function predict(model::ARModel, x_values::Vector{<:Real})
    if !model.state.is_fitted
        throw(ErrorException("Model must be fitted before prediction"))
    end
    
    # Get model parameters
    intercept = model.state.parameters[:intercept]
    coefficients = model.state.parameters[:coefficients]
    
    # For simple predict, use the last p fitted values to make a one-step forecast
    # and repeat it for all requested x_values
    fitted_values = model.state.fitted_values
    p = length(coefficients)
    if length(fitted_values) < p
        throw(ErrorException("Not enough fitted values for AR($p) prediction"))
    end
    
    # Take the last p fitted values
    y_values = fitted_values[end-p+1:end]
    
    # Make one-step forecast: y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p}
    forecast = intercept + sum(coefficients .* y_values)
    
    # Repeat this forecast for all x_values
    predictions = fill(forecast, length(x_values))
    
    # Return as TimeSeries with x_values as timestamps
    return TimeSeries(x_values, predictions)
end

# MA Model implementation
"""
    predict(model::MAModel, x_values::Vector{<:Real})

Generate predictions at specific x values using a fitted MA model.
Note: For MA models, predictions converge to the mean quickly.
For multi-step ahead forecasts, use forecast instead.

Returns a TimeSeries with the predictions and x values as timestamps.
"""
function predict(model::MAModel, x_values::Vector{<:Real})
    if !model.state.is_fitted
        throw(ErrorException("Model must be fitted before prediction"))
    end
    
    # Get model parameters
    μ = model.state.parameters[:mean]
    
    # For simple predict, MA model predictions are just the mean
    # since we don't have recent errors for new x_values
    predictions = fill(μ, length(x_values))
    
    # Return as TimeSeries with x_values as timestamps
    return TimeSeries(x_values, predictions)
end

# Export the predict function
export predict
