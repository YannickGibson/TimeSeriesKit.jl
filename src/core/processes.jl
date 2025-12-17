# Time series process generators (outer constructors)

"""
    WhiteNoise(length::Int; mean::Real=0.0, variance::Real=1.0, name::String="White Noise")

Generate a white noise time series with specified length, mean, and variance.

White noise is a sequence of uncorrelated random variables: ε_t ~ N(mean, variance)

# Arguments
- `length::Int`: Number of time steps
- `mean::Real=0.0`: Mean of the white noise
- `variance::Real=1.0`: Variance of the white noise
- `name::String="White Noise"`: Name for the time series

# Returns
- `TimeSeries`: A time series containing white noise

# Example
```julia
wn = WhiteNoise(100, mean=0.0, variance=5.0)
```
"""
function WhiteNoise(length::Int; mean::Real=0.0, variance::Real=1.0, name::String="White Noise")
    if length < 1
        throw(ArgumentError("Length must be at least 1"))
    end
    if variance <= 0
        throw(ArgumentError("Variance must be positive"))
    end
    
    # Generate white noise with specified mean and variance
    std_dev = sqrt(variance)
    noise = randn(length) .* std_dev .+ mean
    
    return TimeSeries(noise; name=name)
end

"""
    ARProcess(length::Int; phi::Union{Real,Vector{<:Real}}=0.5, constant::Real=0.0, noise_variance::Real=1.0, name::String="")

Generate an AR(p) autoregressive time series.

For AR(1): y_t = constant + phi * y_{t-1} + ε_t
For AR(p): y_t = constant + phi[1] * y_{t-1} + phi[2] * y_{t-2} + ... + phi[p] * y_{t-p} + ε_t

where ε_t ~ N(0, noise_variance)

# Arguments
- `length::Int`: Number of time steps
- `phi::Union{Real,Vector{<:Real}}=0.5`: AR coefficient(s). Can be a single value for AR(1) or a vector for AR(p)
- `constant::Real=0.0`: Constant term in the AR process
- `noise_variance::Real=1.0`: Variance of the white noise innovations
- `name::String=""`: Name for the time series (auto-generated if empty)

# Returns
- `TimeSeries`: A time series containing the AR(p) process

# Examples
```julia
# AR(1) process
ar1 = ARProcess(5000, phi=0.85, constant=10.0)

# AR(2) process
ar2 = ARProcess(5000, phi=[0.5, 0.3], constant=0.0)

# AR(3) process
ar3 = ARProcess(5000, phi=[0.6, -0.2, 0.1])
```
"""
function ARProcess(length::Int; phi::Union{Real,Vector{<:Real}}=0.5, constant::Real=0.0, noise_variance::Real=1.0, name::String="")
    # AR process is ARMA with theta=0 (no MA component)
    if name == ""
        p = phi isa Real ? 1 : Base.length(phi)
        name = "AR($p) Process"
    end
    return ARMAProcess(length; phi=phi, theta=0.0, constant=constant, noise_variance=noise_variance, name=name)
end

"""
    MAProcess(length::Int; theta::Union{Real,Vector{<:Real}}=0.5, mean::Real=0.0, noise_variance::Real=1.0, name::String="")

Generate an MA(q) moving average time series.

For MA(1): y_t = mean + ε_t + theta * ε_{t-1}
For MA(q): y_t = mean + ε_t + theta[1] * ε_{t-1} + theta[2] * ε_{t-2} + ... + theta[q] * ε_{t-q}

where ε_t ~ N(0, noise_variance)

# Arguments
- `length::Int`: Number of time steps
- `theta::Union{Real,Vector{<:Real}}=0.5`: MA coefficient(s). Can be a single value for MA(1) or a vector for MA(q)
- `mean::Real=0.0`: Mean of the process
- `noise_variance::Real=1.0`: Variance of the white noise innovations
- `name::String=""`: Name for the time series (auto-generated if empty)

# Returns
- `TimeSeries`: A time series containing the MA(q) process

# Examples
```julia
# MA(1) process
ma1 = MAProcess(5000, theta=0.8, mean=0.0)

# MA(2) process
ma2 = MAProcess(5000, theta=[0.9, -0.4], mean=0.0)

# MA(3) process
ma3 = MAProcess(5000, theta=[0.6, -0.3, 0.2])
```
"""
function MAProcess(length::Int; theta::Union{Real,Vector{<:Real}}=0.5, mean::Real=0.0, noise_variance::Real=1.0, name::String="")
    # MA process is ARMA with phi=0 (no AR component)
    if name == ""
        q = theta isa Real ? 1 : Base.length(theta)
        name = "MA($q) Process"
    end
    return ARMAProcess(length; phi=0.0, theta=theta, constant=mean, noise_variance=noise_variance, name=name)
end

"""
    ARMAProcess(length::Int; phi::Union{Real,Vector{<:Real}}=0.5, theta::Union{Real,Vector{<:Real}}=0.5, constant::Real=0.0, noise_variance::Real=1.0, name::String="")

Generate an ARMA(p,q) autoregressive moving average time series.

For ARMA(1,1): y_t = constant + phi * y_{t-1} + ε_t + theta * ε_{t-1}
For ARMA(p,q): y_t = constant + phi[1] * y_{t-1} + ... + phi[p] * y_{t-p} + ε_t + theta[1] * ε_{t-1} + ... + theta[q] * ε_{t-q}

where ε_t ~ N(0, noise_variance)

# Arguments
- `length::Int`: Number of time steps
- `phi::Union{Real,Vector{<:Real}}=0.5`: AR coefficient(s). Can be a single value for AR(1) or a vector for AR(p)
- `theta::Union{Real,Vector{<:Real}}=0.5`: MA coefficient(s). Can be a single value for MA(1) or a vector for MA(q)
- `constant::Real=0.0`: Constant term in the ARMA process
- `noise_variance::Real=1.0`: Variance of the white noise innovations
- `name::String=""`: Name for the time series (auto-generated if empty)

# Returns
- `TimeSeries`: A time series containing the ARMA(p,q) process

# Examples
```julia
# ARMA(1,1) process
arma11 = ARMAProcess(5000, phi=0.7, theta=0.5, constant=0.0)

# ARMA(2,1) process
arma21 = ARMAProcess(5000, phi=[0.5, 0.3], theta=0.8)

# ARMA(1,2) process
arma12 = ARMAProcess(5000, phi=0.6, theta=[0.4, -0.2])

# ARMA(2,2) process
arma22 = ARMAProcess(5000, phi=[0.7, -0.2], theta=[0.5, 0.3])
```
"""
function ARMAProcess(length::Int; phi::Union{Real,Vector{<:Real}}=0.5, theta::Union{Real,Vector{<:Real}}=0.5, constant::Real=0.0, noise_variance::Real=1.0, name::String="")
    if length < 1
        throw(ArgumentError("Length must be at least 1"))
    end
    if noise_variance <= 0
        throw(ArgumentError("Noise variance must be positive"))
    end
    
    # Convert scalar parameters to vectors
    phi_vec = phi isa Real ? [phi] : phi
    theta_vec = theta isa Real ? [theta] : theta
    p = Base.length(phi_vec)
    q = Base.length(theta_vec)
    
    # Auto-generate name if not provided
    if isempty(name)
        name = "ARMA($p,$q) Process"
    end
    
    # Generate white noise innovations
    std_dev = sqrt(noise_variance)
    errors = randn(length) .* std_dev
    
    # Generate ARMA(p,q) process
    data = zeros(length)
    
    for t in 1:length
        # Start with constant and current error
        data[t] = constant + errors[t]
        
        # Add AR terms
        for i in 1:min(p, t-1)
            data[t] += phi_vec[i] * data[t-i]
        end
        
        # Add MA terms
        for j in 1:min(q, t-1)
            data[t] += theta_vec[j] * errors[t-j]
        end
    end
    
    return TimeSeries(data; name=name)
end

"""
    RandomWalk(length::Int; mean::Real=0.0, variance::Real=1.0, name::String="Random Walk")

Generate a random walk time series with specified length, mean, and variance.

The random walk is generated as: y_t = y_{t-1} + ε_t, where ε_t ~ N(mean, variance)

# Arguments
- `length::Int`: Number of time steps
- `mean::Real=0.0`: Mean of the random increments
- `variance::Real=1.0`: Variance of the random increments
- `name::String="Random Walk"`: Name for the time series

# Returns
- `TimeSeries`: A time series containing the random walk

# Example
```julia
rw = RandomWalk(100, mean=0.5, variance=2.0)
```
"""
function RandomWalk(length::Int; mean::Real=0.0, variance::Real=1.0, name::String="Random Walk")
    if length < 1
        throw(ArgumentError("Length must be at least 1"))
    end
    if variance <= 0
        throw(ArgumentError("Variance must be positive"))
    end
    
    # Generate random increments with specified mean and variance
    std_dev = sqrt(variance)
    increments = randn(length) .* std_dev .+ mean
    
    # Compute cumulative sum to get random walk
    walk = cumsum(increments)
    
    return TimeSeries(walk; name=name)
end
