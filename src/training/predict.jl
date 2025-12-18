# Model prediction functions (predict at specific x values)

using ..TimeSeriesKit: AbstractTimeSeriesModel, TimeSeries, ARModel, ARIMAModel, LinearModel, RidgeModel, SESModel, BayesianARModel, PredictionResult
using Statistics

"""
    predict(model::AbstractTimeSeriesModel, x_values::Vector{<:Real})

Generate predictions at specific x values from a fitted model.
"""
function predict end

"""
    predict(model::Union{LinearModel, RidgeModel}, x_values::Vector{<:Real}; return_uncertainty::Bool=false)

Generate predictions at specific x values using a fitted linear or ridge regression model.

Returns a TimeSeries with the predictions and x values as timestamps, or a PredictionResult
if return_uncertainty=true (only supported for LinearModel).

# Arguments
- `return_uncertainty::Bool=false`: If true, returns PredictionResult with variance estimates (LinearModel only)
"""
function predict(model::Union{LinearModel, RidgeModel}, x_values::Vector{<:Real}; return_uncertainty::Bool=false)
    if !model.state.is_fitted
        throw(ErrorException("Model must be fitted before prediction"))
    end
    
    # Get model parameters
    intercept = model.state.parameters[:intercept]
    slope = model.state.parameters[:slope]
    
    # Generate predictions: y = intercept + slope * x
    predictions = intercept .+ slope .* x_values
    
    # Calculate prediction variance if requested and available
    if return_uncertainty && model isa LinearModel && haskey(model.state.parameters, :residual_variance)
        σ² = model.state.parameters[:residual_variance]
        
        # Get variance-covariance matrix components
        # For prediction variance: Var(ŷ₀) = σ²(x₀ᵀ(X'X)⁻¹x₀)
        # where x₀ = [1, x_value]
        intercept_var = model.state.parameters[:intercept_variance]
        slope_var = model.state.parameters[:slope_variance]
        covar = model.state.parameters[:covariance]
        
        # Calculate prediction variance for each x value
        pred_variance = similar(x_values, Float64)
        for (i, x) in enumerate(x_values)
            # x₀ᵀ Var(β) x₀ = [1, x] * [[var_int, cov], [cov, var_slope]] * [1, x]'
            # = var_int + 2*x*cov + x²*var_slope
            pred_variance[i] = intercept_var + 2 * x * covar + x^2 * slope_var
        end
        
        ts = TimeSeries(x_values, predictions)
        return PredictionResult(ts, pred_variance)
    end
    
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

"""
    predict(model::BayesianARModel, x_values::Vector{<:Real}; return_uncertainty::Bool=false)

Generate predictions at specific x values using a fitted Bayesian AR model.

# Arguments
- `model::BayesianARModel`: Fitted Bayesian AR model
- `x_values::Vector{<:Real}`: Timestamps at which to generate predictions
- `return_uncertainty::Bool`: If true, returns PredictionResult with uncertainty estimates

# Returns
- If `return_uncertainty=false`: TimeSeries with predictions
- If `return_uncertainty=true`: PredictionResult with predictions and uncertainty

Note: For simple predict, this generates one-step forecasts based on the last observed values.
For multi-step ahead forecasts with proper uncertainty propagation, use iterative_predict.
"""
function predict(model::BayesianARModel, x_values::Vector{<:Real}; return_uncertainty::Bool=false)
    if !model.state.is_fitted
        throw(ErrorException("Model must be fitted before prediction"))
    end
    
    # Get model parameters
    intercept = model.state.parameters[:intercept]
    coefficients = model.state.parameters[:coefficients]
    σ²_post = model.state.parameters[:residual_variance]
    Σ_post = model.state.parameters[:posterior_covariance]
    
    # For simple predict, use the last p fitted values to make a one-step forecast
    fitted_values = model.state.fitted_values
    p = length(coefficients)
    
    # Take the last p fitted values
    y_values = fitted_values[end-p+1:end]
    
    # Make one-step forecast: y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p}
    forecast = intercept + sum(coefficients .* y_values)
    
    # Repeat this forecast for all x_values
    predictions = fill(forecast, length(x_values))
    
    if return_uncertainty
        # Multi-step ahead prediction with uncertainty propagation
        # For each step, we use previous predictions as inputs, and uncertainty grows
        
        pred_variances = zeros(length(x_values))
        predictions_vec = zeros(length(x_values))
        
        # Keep track of the lagged values (starting with actual observations)
        current_lags = copy(y_values)
        
        for i in 1:length(x_values)
            # Create feature vector: [1, y_{t-1}, y_{t-2}, ..., y_{t-p}]
            x_pred = vcat([1.0], reverse(current_lags))
            
            # Make prediction
            y_pred = intercept + sum(coefficients .* reverse(current_lags))
            predictions_vec[i] = y_pred + rand() * model.state.parameters[:residual_variance]
            
            # Calculate prediction variance for this step
            # Var(y_new) = σ² + x'Σ_post x
            param_uncertainty = dot(x_pred, Σ_post * x_pred)
            param_uncertainty = max(0.0, param_uncertainty)
            pred_variances[i] = σ²_post + param_uncertainty
            
            # Update lags: shift and add new prediction
            # For next step, use the prediction we just made
            current_lags = vcat(y_pred, current_lags[1:end-1])
        end
        ts = TimeSeries(x_values, predictions_vec)
        return PredictionResult(ts, pred_variances)
    else
        return TimeSeries(x_values, predictions)
    end
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
