# Model prediction functions (predict at specific x values)

using ..TimeSeriesKit: AbstractTimeSeriesModel, TimeSeries, ARModel, ARIMAModel, LinearModel, RidgeModel, SESModel
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
    
    # Take the last p fitted values
    y_values = fitted_values[end-p+1:end]
    
    # Make one-step forecast: y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p}
    forecast = intercept + sum(coefficients .* y_values)
    
    # Repeat this forecast for all x_values
    predictions = fill(forecast, length(x_values))
    
    # Return as TimeSeries with x_values as timestamps
    return TimeSeries(x_values, predictions)
end

# ARIMA Model implementation
"""
    predict(model::ARIMAModel, x_values::Vector{<:Real})

Generate predictions at specific x values using a fitted ARIMA model.
Note: This generates simple one-step forecasts. For proper multi-step forecasts, use forecast.

Returns a TimeSeries with the predictions and x values as timestamps.
"""
function predict(model::ARIMAModel, x_values::Vector{<:Real})
    if !model.state.is_fitted
        throw(ErrorException("Model must be fitted before prediction"))
    end
    
    # Get parameters
    intercept = model.state.parameters[:intercept]
    ar_coeffs = model.state.parameters[:ar_coefficients]
    ma_coeffs = model.state.parameters[:ma_coefficients]
    d = model.state.parameters[:d]
    
    # For simple predict, make one-step forecast on differenced scale
    if model.p > 0
        fitted_diff = model.state.fitted_values
        p = model.p
        
        # Filter out NaN values
        valid_fitted = fitted_diff[.!isnan.(fitted_diff)]
        
        y_values = valid_fitted[end-p+1:end]
        forecast_diff = intercept + sum(ar_coeffs .* y_values)
    else
        # MA model: predict mean
        forecast_diff = intercept
    end
    
    # Integrate back if needed
    if d > 0
        original_values = model.state.parameters[:original_values]
        last_values = original_values[end-d+1:end]
        forecast_original = TimeSeriesKit.Models.ARIMA.integrate_forecast([forecast_diff], last_values, d)[1]
    else
        forecast_original = forecast_diff
    end
    
    # Repeat for all x_values
    predictions = fill(forecast_original, length(x_values))
    
    return TimeSeries(x_values, predictions)
end

# Export the predict function
export predict
