# ARIMA (Autoregressive Integrated Moving Average) Model

using ..TimeSeriesKit: AbstractTimeSeriesModel, ModelState, TimeSeries
using LinearAlgebra
using Statistics

"""
    ARIMAModel

ARIMA(p,d,q) model: Combines AR(p), differencing (d), and MA(q) components.

The model is applied to differenced data:
- First, the series is differenced d times to achieve stationarity
- Then an ARMA(p,q) model is fitted to the differenced series
- Forecasts are integrated back to the original scale

Model structure: (1-φ₁L-...-φₚLᵖ)(1-L)ᵈyₜ = c + (1+θ₁L+...+θ_qLᵍ)εₜ

where L is the lag operator.
"""
mutable struct ARIMAModel <: AbstractTimeSeriesModel
    p::Int  # AR order
    d::Int  # Differencing order
    q::Int  # MA order
    state::ModelState
    
    function ARIMAModel(; p::Int, d::Int, q::Int)
        if p < 0 || d < 0 || q < 0
            throw(ArgumentError("ARIMA orders must be non-negative"))
        end
        if p == 0 && q == 0
            throw(ArgumentError("At least one of p or q must be positive"))
        end
        new(p, d, q, ModelState())
    end
end

"""
    difference_series(values::Vector{Float64}, d::Int)

Apply differencing d times to a series.
"""
function difference_series(values::Vector{Float64}, d::Int)
    result = values
    for _ in 1:d
        result = diff(result)
    end
    return result
end

"""
    integrate_forecast(forecasts::Vector{Float64}, last_values::Vector{Float64}, d::Int)

Integrate differenced forecasts back to original scale.
"""
function integrate_forecast(forecasts::Vector{Float64}, last_values::Vector{Float64}, d::Int)
    if d == 0
        return forecasts
    end
    
    result = copy(forecasts)
    
    # For each differencing level, integrate
    for _ in 1:d
        # Need the last value from the previous level
        last_val = last_values[end]
        integrated = zeros(length(result))
        integrated[1] = last_val + result[1]
        for i in 2:length(result)
            integrated[i] = integrated[i-1] + result[i]
        end
        result = integrated
        last_values = vcat(last_values, integrated)
    end
    
    return result
end

"""
    fit_arma(values::Vector{Float64}, p::Int, q::Int)

Fit ARMA(p,q) model to stationary data.
Returns intercept, AR coefficients, MA coefficients, fitted values, and residuals.
"""
function fit_arma(values::Vector{Float64}, p::Int, q::Int)
    n = length(values)
    
    # If only AR component (ARMA(p,0) = AR(p))
    if q == 0
        # Create design matrix for AR
        if n <= p
            throw(ArgumentError("Time series too short for AR($p) model"))
        end
        
        X = zeros(n - p, p + 1)
        y = zeros(n - p)
        
        for i in 1:(n - p)
            X[i, 1] = 1.0
            for j in 1:p
                X[i, j + 1] = values[p + i - j]
            end
            y[i] = values[p + i]
        end
        
        # OLS estimation
        β = (X' * X) \ (X' * y)
        intercept = β[1]
        ar_coeffs = β[2:end]
        ma_coeffs = Float64[]
        
        # Compute fitted values and residuals
        fitted = X * β
        residuals = y .- fitted
        
        # Pad fitted values
        full_fitted = vcat(fill(NaN, p), fitted)
        full_residuals = vcat(fill(NaN, p), residuals)
        
        return intercept, ar_coeffs, ma_coeffs, full_fitted, full_residuals
    end
    
    # If only MA component (ARMA(0,q) = MA(q))
    if p == 0
        μ = mean(values)
        centered = values .- μ
        
        # Estimate MA coefficients using method of moments
        c0 = sum(centered.^2) / n
        acf = zeros(q)
        for lag in 1:q
            ck = sum(centered[1:n-lag] .* centered[lag+1:n]) / n
            acf[lag] = ck / c0
        end
        
        # Simplified MA parameter estimation
        ma_coeffs = if q == 1
            rho1 = acf[1]
            if abs(rho1) < 0.5
                [(-1 + sqrt(1 - 4*rho1^2)) / (2*rho1)]
            else
                [rho1]
            end
        else
            acf
        end
        
        # Compute fitted values
        fitted = fill(μ, n)
        errors = zeros(n)
        
        for i in 1:min(q, n)
            errors[i] = centered[i]
        end
        
        for t in (q+1):n
            ma_component = sum(ma_coeffs[j] * errors[t-j] for j in 1:q)
            fitted[t] = μ + ma_component
            errors[t] = values[t] - fitted[t]
        end
        
        residuals = values .- fitted
        
        return μ, Float64[], ma_coeffs, fitted, residuals
    end
    
    # Full ARMA(p,q) - simplified approach: fit AR first, then estimate MA on residuals
    # This is a simplification; proper ARMA fitting requires MLE
    
    # Fit AR part
    if n <= p
        throw(ArgumentError("Time series too short for ARMA($p,$q) model"))
    end
    
    X = zeros(n - p, p + 1)
    y = zeros(n - p)
    
    for i in 1:(n - p)
        X[i, 1] = 1.0
        for j in 1:p
            X[i, j + 1] = values[p + i - j]
        end
        y[i] = values[p + i]
    end
    
    β = (X' * X) \ (X' * y)
    intercept = β[1]
    ar_coeffs = β[2:end]
    
    # Get residuals from AR fit
    ar_fitted = X * β
    ar_residuals = y .- ar_fitted
    
    # Estimate MA coefficients on residuals
    μ_res = mean(ar_residuals)
    centered_res = ar_residuals .- μ_res
    n_res = length(centered_res)
    
    c0 = sum(centered_res.^2) / n_res
    acf = zeros(q)
    for lag in 1:min(q, n_res-1)
        ck = sum(centered_res[1:n_res-lag] .* centered_res[lag+1:n_res]) / n_res
        acf[lag] = ck / c0
    end
    
    ma_coeffs = acf[1:q]
    
    # Compute full fitted values
    full_fitted = vcat(fill(NaN, p), ar_fitted)
    full_residuals = vcat(fill(NaN, p), ar_residuals)
    
    return intercept, ar_coeffs, ma_coeffs, full_fitted, full_residuals
end

# Minimum training size implementation
TimeSeriesKit.Models.min_train_size(model::ARIMAModel) = max(model.p, model.q) * 3 + model.d

# Export the model type
export ARIMAModel
