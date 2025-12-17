# Model fitting functions

using ..TimeSeriesKit: AbstractTimeSeriesModel, TimeSeries, ARModel, ARIMAModel, LinearModel, SESModel
using ..TimeSeriesKit: validate_timeseries
using LinearAlgebra
using Statistics

"""
    fit(model::AbstractTimeSeriesModel, ts::TimeSeries)

Fit a time series model to data. If the model was fitted previously, this function will refit it completely.
"""
function fit end

"""
    fit(model::ARModel, ts::TimeSeries)

Fit an AR model to time series data using least squares.
"""
function fit(model::ARModel, ts::TimeSeries)
    validate_timeseries(ts)
    
    # Import the helper function
    X, y = TimeSeriesKit.Models.Autoregressive.create_ar_matrix(ts, model.p)
    
    if rank(X' * X) == max(size(X)...)
        # Ordinary least squares: β = (X'X)^(-1)X'y
        β = (X' * X) \ (X' * y)
    else
        β = pinv(X' * X) * (X' * y)
    end
    # Store parameters
    model.state.parameters[:intercept] = β[1]
    model.state.parameters[:coefficients] = β[2:end]
    
    # Compute fitted values and residuals
    fitted = X * β
    residuals = y .- fitted
    
    model.state.fitted_values = ts.values
    model.state.residuals = residuals
    model.state.is_fitted = true
    
    return model
end

"""
    fit(model::LinearModel, ts::TimeSeries)

Fit a linear trend model to time series data.
"""
function fit(model::LinearModel, ts::TimeSeries)
    validate_timeseries(ts)
    
    X = TimeSeriesKit.Models.Linear.create_matrix_X(ts.timestamps)
    y = ts.values
    
    # Ordinary least squares
    w_ols = (X' * X) \ (X' * y)  # Inverse of left side, then multiplying
    
    # Store parameters
    model.state.parameters[:intercept] = w_ols[1]
    model.state.parameters[:slope] = w_ols[2]
    
    # Compute fitted values and residuals
    fitted = X * w_ols
    residuals = y .- fitted
    
    model.state.fitted_values = fitted
    model.state.residuals = residuals
    model.state.is_fitted = true
    
    return model
end

"""
    fit(model::RidgeModel, ts::TimeSeries)

Fit a ridge regression model to time series data.
"""
function fit(model::RidgeModel, ts::TimeSeries)
    validate_timeseries(ts)
    
    X = TimeSeriesKit.Models.Linear.create_matrix_X(ts.timestamps)
    y = ts.values
    
    # Ridge regression: w = (X'X + λI)^(-1) X'y
    p = size(X, 2)
    I_ridge = Matrix{Float64}(I, p, p)
    # Don't regularize the intercept term
    I_ridge[1, 1] = 0.0
    
    w_ridge = (X' * X + model.λ * I_ridge) \ (X' * y)
    
    # Store parameters
    model.state.parameters[:intercept] = w_ridge[1]
    model.state.parameters[:slope] = w_ridge[2]
    model.state.parameters[:λ] = model.λ
    
    # Compute fitted values and residuals
    fitted = X * w_ridge
    residuals = y .- fitted
    
    model.state.fitted_values = fitted
    model.state.residuals = residuals
    model.state.is_fitted = true
    
    return model
end

"""
    fit(model::SESModel, ts::TimeSeries)

Fit a Simple Exponential Smoothing model to time series data.
"""
function fit(model::SESModel, ts::TimeSeries)
    validate_timeseries(ts)
    
    values = ts.values
    
    # Fit the model
    fitted, _, level = TimeSeriesKit.Models.ETS.ses_forecast(values, model.alpha, 1)
    residuals = values .- fitted
    
    # Store parameters and results
    model.state.parameters[:alpha] = model.alpha
    model.state.parameters[:level] = level
    model.state.fitted_values = fitted
    model.state.residuals = residuals
    model.state.is_fitted = true
    
    return model
end

# ARIMA Model implementation
function fit(model::ARIMAModel, ts::TimeSeries)
    validate_timeseries(ts)
    
    values = Float64.(ts.values)
    n = length(values)
    p, d, q = model.p, model.d, model.q
    
    if n < max(p, q) * 3 + d
        throw(ArgumentError("Time series too short for ARIMA($p,$d,$q) model"))
    end
    
    # Store original values for integration later
    model.state.parameters[:original_values] = values
    
    # Apply differencing
    if d > 0
        diff_values = TimeSeriesKit.Models.ARIMA.difference_series(values, d)
        model.state.parameters[:differenced_values] = diff_values
    else
        diff_values = values
    end
    
    # Fit ARMA model to differenced data
    intercept, ar_coeffs, ma_coeffs, fitted_diff, residuals_diff = 
        TimeSeriesKit.Models.ARIMA.fit_arma(diff_values, p, q)
    
    # Store parameters
    model.state.parameters[:intercept] = intercept
    model.state.parameters[:ar_coefficients] = ar_coeffs
    model.state.parameters[:ma_coefficients] = ma_coeffs
    model.state.parameters[:d] = d
    
    # Store fitted values and residuals (on differenced scale)
    model.state.fitted_values = fitted_diff
    model.state.residuals = residuals_diff
    model.state.is_fitted = true
    
    return model
end

# Export the fit function
export fit
