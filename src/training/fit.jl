# Model fitting functions

using ..TimeSeriesKit: AbstractTimeSeriesModel, TimeSeries, ARModel, ARIMAModel, LinearModel, SESModel, BayesianARModel
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
    fit(model::BayesianARModel, ts::TimeSeries)

Fit a Bayesian AR model to time series data using Bayesian linear regression.

Uses conjugate Normal-Inverse-Gamma prior:
- Prior: β ~ N(0, (1/λ)I), where λ is prior_precision
- Posterior: β|y,σ² ~ N(μ_post, Σ_post)
- Posterior variance: σ²_post ~ Inverse-Gamma(a_post, b_post)

The posterior mean is: μ_post = (X'X + λI)^(-1) X'y
The posterior covariance is: Σ_post = σ²_post (X'X + λI)^(-1)
"""
function fit(model::BayesianARModel, ts::TimeSeries)
    validate_timeseries(ts)
    
    # Create AR design matrix
    X, y = TimeSeriesKit.Models.Autoregressive.create_ar_matrix(ts, model.p)
    n = length(y)
    k = size(X, 2)  # Number of parameters (p + 1 for intercept)
    
    # Prior precision matrix
    λ = model.prior_precision
    Λ = λ * I(k)
    
    # Posterior precision matrix: X'X + Λ
    XtX = X' * X
    precision_post = XtX + Λ
    
    # Posterior mean: (X'X + Λ)^(-1) X'y
    β_post = precision_post \ (X' * y)
    
    # Compute fitted values and residuals
    fitted = X * β_post
    residuals = y .- fitted
    
    # Posterior parameters for σ²
    # Using weak prior: a_0 = b_0 = 0.001
    a_0 = 0.001
    b_0 = 0.001
    a_post = a_0 + n / 2
    b_post = b_0 + 0.5 * sum(residuals .^ 2)
    
    # Posterior mean of σ²
    # Check if a_post > 1 for valid mean
    if a_post <= 1
        σ²_post = b_post / max(a_post, 0.5)  # Fallback for numerical stability
    else
        σ²_post = b_post / (a_post - 1)  # Mean of Inverse-Gamma(a, b) = b/(a-1) for a > 1
    end
    
    # Posterior covariance matrix of β
    # Use more numerically stable computation
    Σ_post = σ²_post * inv(Matrix(precision_post))
    
    # Ensure positive definiteness (numerical stability)
    Σ_post = (Σ_post + Σ_post') / 2  # Make symmetric
    
    # Store parameters
    model.state.parameters[:intercept] = β_post[1]
    model.state.parameters[:coefficients] = β_post[2:end]
    model.state.parameters[:residual_variance] = σ²_post
    model.state.parameters[:posterior_covariance] = Σ_post
    model.state.parameters[:posterior_precision] = precision_post
    
    # Store individual parameter variances for easy access
    model.state.parameters[:intercept_variance] = Σ_post[1, 1]
    model.state.parameters[:coefficient_variances] = [Σ_post[i, i] for i in 2:k]
    
    # Store posterior hyperparameters for prediction
    model.state.parameters[:a_post] = a_post
    model.state.parameters[:b_post] = b_post
    
    # Store fitted values and residuals
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
    n = length(y)
    
    # Ordinary least squares
    w_ols = (X' * X) \ (X' * y)  # Inverse of left side, then multiplying
    
    # Compute fitted values and residuals
    fitted = X * w_ols
    residuals = y .- fitted
    
    # Calculate residual variance (σ²)
    # Degrees of freedom: n - number of parameters (2 for intercept and slope)
    # If df would be 0 or negative, use n-1 as fallback
    # β̂ is linear in y:
    # β̂ = (X'X)^(-1) * X' * y = (X'X)^(-1) * X' * (Xβ + ε) = β + (X'X)^(-1) * X' * ε
    # Since Cov(ε) = σ^2 * I:
    # Cov(β̂) = (X'X)^(-1) * X' * Cov(ε) * X * (X'X)^(-1) = σ^2 * (X'X)^(-1)
    df = n - size(X, 2)
    if df <= 0
        df = n - 1  # Fallback for small sample sizes
    end
    σ² = sum(residuals .^ 2) / df
    
    # Calculate variance-covariance matrix: Var(β) = σ²(X'X)⁻¹
    XtX_inv = inv(X' * X)
    var_covar = σ² * XtX_inv
    
    # Store parameters
    model.state.parameters[:intercept] = w_ols[1]
    model.state.parameters[:slope] = w_ols[2]
    model.state.parameters[:intercept_variance] = var_covar[1, 1]
    model.state.parameters[:slope_variance] = var_covar[2, 2]
    model.state.parameters[:covariance] = var_covar[1, 2]
    model.state.parameters[:residual_variance] = σ²
    
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
