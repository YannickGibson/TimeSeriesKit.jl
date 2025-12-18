# Autoregressive (AR) Model

using ..TimeSeriesKit: AbstractTimeSeriesModel, ModelState, TimeSeries
using LinearAlgebra
using Statistics

"""
    ARModel

Autoregressive model of order p: y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} + ε_t
"""
mutable struct ARModel <: AbstractTimeSeriesModel
    p::Int  # Order of the AR model
    state::ModelState
    
    function ARModel(; p::Int)
        if p < 1
            throw(ArgumentError("AR order must be at least 1"))
        end
        new(p, ModelState())
    end
end

"""
    create_ar_matrix(ts::TimeSeries, p::Int)

Create design matrix for AR model fitting.
"""
function create_ar_matrix(ts::TimeSeries, p::Int)
    n = length(ts)
    if n <= p
        throw(ArgumentError("Time series too short for AR($p) model"))
    end
    
    # Create lagged design matrix
    X = zeros(n - p, p + 1)  # +1 for intercept
    y = zeros(n - p)
    
    for i in 1:(n - p)
        X[i, 1] = 1.0  # Intercept
        for j in 1:p
            X[i, j + 1] = ts.values[p + i - j]
        end
        y[i] = ts.values[p + i]
    end
    
    return X, y
end

# Minimum training size implementation
TimeSeriesKit.Models.min_train_size(model::ARModel) = model.p + 1 # Needs at least one Y data point

"""
    BayesianARModel

Bayesian Autoregressive model of order p with uncertainty quantification.
Uses Bayesian linear regression with conjugate Normal-Inverse-Gamma prior.

The model estimates: y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} + ε_t

With Bayesian estimation, we also get:
- Posterior distribution of parameters
- Parameter uncertainties (variance-covariance matrix)
- Prediction uncertainties

# Arguments
- `p::Int`: Order of the AR model (must be at least 1)
- `prior_mean::Vector{Float64}`: Prior mean for parameters [intercept, φ₁, ..., φₚ] (default: zeros)
- `prior_variance::Float64`: Prior variance for parameters (default: 1000.0, weak prior)

# Prior specification
Uses conjugate Normal-Inverse-Gamma prior:
- β ~ N(μ₀, σ₀²I) where μ₀ is prior_mean and σ₀² is prior_variance
- σ² ~ Inverse-Gamma(a, b) with a=b=0.001 (weak prior)

# Examples
```julia
# Non-informative prior (default)
model = BayesianARModel(p=2)

# Informative prior: expect AR(1)≈0.7, AR(2)≈0.2, intercept≈0
model = BayesianARModel(p=2, prior_mean=[0.0, 0.7, 0.2], prior_variance=0.1)
```
"""
mutable struct BayesianARModel <: AbstractTimeSeriesModel
    p::Int  # Order of the AR model
    prior_mean::Vector{Float64}  # Prior mean for parameters
    prior_variance::Float64  # Prior variance for parameters
    state::ModelState
    
    function BayesianARModel(; p::Int, prior_mean::Union{Vector{Float64}, Nothing}=nothing, prior_variance::Float64=1000.0)
        if p < 1
            throw(ArgumentError("AR order must be at least 1"))
        end
        if prior_variance <= 0
            throw(ArgumentError("Prior variance must be positive"))
        end
        
        # Default prior mean: zero vector
        if prior_mean === nothing
            prior_mean = zeros(p + 1)  # +1 for intercept
        elseif length(prior_mean) != p + 1
            throw(ArgumentError("Prior mean must have length p+1 (got $(length(prior_mean)), expected $(p+1))"))
        end
        
        new(p, prior_mean, prior_variance, ModelState())
    end
end

# Minimum training size implementation
TimeSeriesKit.Models.min_train_size(model::BayesianARModel) = model.p + 1

# Export the model types
export ARModel, BayesianARModel
