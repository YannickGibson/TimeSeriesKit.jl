# Model forecasting functions (horizon-based predictions)

using ..TimeSeriesKit: AbstractTimeSeriesModel, TimeSeries, ARModel, LinearModel, SESModel
using Statistics

"""
    forecast(model::AbstractTimeSeriesModel, ts::TimeSeries, horizon::Int)

Generate forecasts from a fitted model for a given horizon.

Returns a TimeSeries with the forecast values.
"""
function forecast end

"""
    forecast(model::ARModel, ts::TimeSeries, horizon::Int)

Generate forecasts using a fitted AR model.

Returns a TimeSeries with the forecast values.
"""
function forecast(model::ARModel, ts::TimeSeries, horizon::Int)
    if !model.state.is_fitted
        throw(ErrorException("Model must be fitted before forecasting"))
    end
    
    if horizon < 1
        throw(ArgumentError("Horizon must be at least 1"))
    end
    
    # Get model parameters
    intercept = model.state.parameters[:intercept]
    coeffs = model.state.parameters[:coefficients]
    p = model.p
    
    # Initialize with last p values from the time series
    history = collect(ts.values[end-p+1:end])
    forecasts = zeros(horizon)
    
    # Generate forecasts iteratively
    for h in 1:horizon
        # Compute forecast: y_t = c + φ₁y_{t-1} + ... + φₚy_{t-p}
        forecast_val = intercept
        for i in 1:p
            forecast_val += coeffs[i] * history[end - i + 1]
        end
        
        forecasts[h] = forecast_val
        
        # Update history for next forecast
        push!(history, forecast_val)
    end
    
    # Generate x values and return TimeSeries
    x_values = extrapolate_timestamps(ts, horizon)
    return TimeSeries(x_values, forecasts)
end

"""
    forecast(model::LinearModel, ts::TimeSeries, horizon::Int)

Generate forecasts using a fitted linear trend model.

Returns a TimeSeries with the forecast values.
"""
function forecast(model::LinearModel, ts::TimeSeries, horizon::Int)
    if !model.state.is_fitted
        throw(ErrorException("Model must be fitted before forecasting"))
    end
    
    if horizon < 1
        throw(ArgumentError("Horizon must be at least 1"))
    end
    
    # Get model parameters
    intercept = model.state.parameters[:intercept]
    slope = model.state.parameters[:slope]
    
    # Generate x values for forecasts
    x_values = extrapolate_timestamps(ts, horizon)
    
    # Generate forecasts by extrapolating the linear trend
    forecasts = zeros(horizon)
    for h in 1:horizon
        forecasts[h] = intercept + slope * x_values[h]
    end
    
    return TimeSeries(x_values, forecasts)
end

"""
    forecast(model::SESModel, ts::TimeSeries, horizon::Int)

Generate forecasts using a fitted Simple Exponential Smoothing model.

Returns a TimeSeries with the forecast values.
"""
function forecast(model::SESModel, ts::TimeSeries, horizon::Int)
    if !model.state.is_fitted
        throw(ErrorException("Model must be fitted before forecasting"))
    end
    
    if horizon < 1
        throw(ArgumentError("Horizon must be at least 1"))
    end
    
    # SES produces flat forecasts equal to the final level
    level = model.state.parameters[:level]
    forecasts = fill(level, horizon)
    
    # Generate x values and return TimeSeries
    x_values = extrapolate_timestamps(ts, horizon)
    return TimeSeries(x_values, forecasts)
end

# Export the forecast function
export forecast
