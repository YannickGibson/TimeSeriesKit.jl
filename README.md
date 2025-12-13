# TimeSeriesKit.jl

[![Build Status](https://github.com/YannickGibson/TimeSeriesKit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/YannickGibson/TimeSeriesKit.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/YannickGibson/TimeSeriesKit.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/YannickGibson/TimeSeriesKit.jl)

**TimeSeriesKit.jl** is a Julia package for time series analysis and manipulation. It provides tools for managing time series datasets, methods for forecasting, validation techniques, and visualization capabilities.

# Overview
## Features
- ETS (Error, Trend, Seasonality) models such as Single Exponential Smoothing (SES) and Double Exponential Smoothing (DES)
- AR/MA, ARMA
- Cross-validation techniques for time series
- Dataset loading
- Uncertainty estimation
- Visualization tools
- Kalman filter implementation



### Julia and implementation features
- Multiple dispatch
- Broadcasting
- Code coverage

# Installation

For development Julia 1.11.7 was used.

**Clone repository**
```bash
git clone https://github.com/username/ProjectName.git TimeSeriesKit
```
**Open Julia**
```bash
julia
```

**Add the package to an environment**
```bash
]
add TimeSeriesKit
⌫
```


**Use the package**
```julia

using TimeSeriesKit
```

# Usage

Check the folder `examples/` for usage illustration.