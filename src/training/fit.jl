# Model fitting functions

using ..TimeSeriesKit: AbstractTimeSeriesModel, TimeSeries, ARModel, LinearModel, SESModel
using ..TimeSeriesKit: validate_timeseries
using LinearAlgebra
using Statistics

"""
    fit(model::AbstractTimeSeriesModel, ts::TimeSeries)

Fit a time series model to data.
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
    
    # Ordinary least squares: β = (X'X)^(-1)X'y
    β = (X' * X) \ (X' * y)
    
    # Store parameters
    model.state.parameters[:intercept] = β[1]
    model.state.parameters[:coefficients] = β[2:end]
    
    # Compute fitted values and residuals
    fitted = X * β
    residuals = y .- fitted
    
    model.state.fitted_values = fitted
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
    
    n = length(ts)
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
    
    n = length(ts)
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
    
    # Optimize alpha if not provided
    if model.alpha === nothing
        alpha = TimeSeriesKit.Models.ETS.optimize_alpha(values)
        model.alpha = alpha
    end
    
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

# Export the fit function
export fit
