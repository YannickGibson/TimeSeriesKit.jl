# Model forecasting functions (horizon-based predictions)

using ..TimeSeriesKit: AbstractTimeSeriesModel, TimeSeries, ARModel, ARIMAModel, LinearModel, SESModel
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

"""
    forecast(model::ARIMAModel, ts::TimeSeries, horizon::Int)

Generate forecasts using a fitted ARIMA model.

The forecasting process:
1. Generate forecasts on the differenced scale using ARMA
2. Integrate forecasts back to the original scale

Returns a TimeSeries with the forecast values.
"""
function forecast(model::ARIMAModel, ts::TimeSeries, horizon::Int)
    if !model.state.is_fitted
        throw(ErrorException("Model must be fitted before forecasting"))
    end
    
    if horizon < 1
        throw(ArgumentError("Horizon must be at least 1"))
    end
    
    # Get model parameters
    intercept = model.state.parameters[:intercept]
    ar_coeffs = model.state.parameters[:ar_coefficients]
    ma_coeffs = model.state.parameters[:ma_coefficients]
    d = model.state.parameters[:d]
    p = model.p
    q = model.q
    
    # Get differenced values and residuals for MA component
    if d > 0
        diff_values = model.state.parameters[:differenced_values]
    else
        diff_values = Float64.(ts.values)
    end
    
    # Initialize history for AR and errors for MA
    history = collect(diff_values[end-max(p,q)+1:end])
    
    # Get recent residuals for MA component
    residuals = model.state.residuals
    valid_residuals = residuals[.!isnan.(residuals)]
    errors = length(valid_residuals) >= q ? collect(valid_residuals[end-q+1:end]) : zeros(q)
    
    forecasts_diff = zeros(horizon)
    
    # Generate forecasts on differenced scale
    for h in 1:horizon
        forecast_val = intercept
        
        # AR component
        if p > 0
            for i in 1:p
                if length(history) >= i
                    forecast_val += ar_coeffs[i] * history[end - i + 1]
                end
            end
        end
        
        # MA component (errors decay to zero for future)
        if q > 0
            for i in 1:min(q, length(errors))
                if h <= i  # Only use errors that are "known"
                    forecast_val += ma_coeffs[i] * errors[end - i + 1]
                end
            end
        end
        
        forecasts_diff[h] = forecast_val
        
        # Update history for next forecast
        push!(history, forecast_val)
        
        # Future errors are assumed to be zero
        if length(errors) >= q
            popfirst!(errors)
        end
        push!(errors, 0.0)
    end
    
    # Integrate forecasts back to original scale
    if d > 0
        original_values = model.state.parameters[:original_values]
        last_values = original_values[end-d+1:end]
        forecasts = TimeSeriesKit.Models.ARIMA.integrate_forecast(forecasts_diff, last_values, d)
    else
        forecasts = forecasts_diff
    end
    
    # Generate x values and return TimeSeries
    x_values = extrapolate_timestamps(ts, horizon)
    return TimeSeries(x_values, forecasts)
end

# Export the forecast function
export forecast
