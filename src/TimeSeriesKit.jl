module TimeSeriesKit

# Core functionality
include("core/types.jl")
include("core/processes.jl")
include("core/utils.jl")

# Export core types
export TimeSeries, AbstractTimeSeriesModel, ModelState
export RandomWalk, WhiteNoise, ARProcess, MAProcess, ARMAProcess

# Export core utilities
export validate_timeseries, split_train_test, difference, normalize
export differentiate, integrate

# Models submodule structure
module Models
    using ..TimeSeriesKit
    
    # Abstract models
    include("models/abstract.jl")
    export is_fitted, get_parameters, get_residuals, min_train_size
    
    # Autoregressive models
    module Autoregressive
        using ..TimeSeriesKit
        include("models/autoregressive/ar.jl")
    end
    using .Autoregressive
    export ARModel
    
    # ARIMA models
    module ARIMA
        using ..TimeSeriesKit
        include("models/arima/arima.jl")
    end
    using .ARIMA
    export ARIMAModel
    
    # Linear models
    module Linear
        using ..TimeSeriesKit
        include("models/linear/linear.jl")
    end
    using .Linear
    export LinearModel, RidgeModel
    
    # ETS models
    module ETS
        using ..TimeSeriesKit
        include("models/ets/ses.jl")
    end
    using .ETS
    export SESModel
end

# Export model types at package level
using .Models
export ARModel, ARIMAModel, LinearModel, RidgeModel, SESModel
export is_fitted, get_parameters, get_residuals, min_train_size

# Training module
module Training
    using ..TimeSeriesKit
    using ..Models
    using LinearAlgebra
    using Statistics
    
    include("training/utils.jl")
    include("training/fit.jl")
    include("training/predict.jl")
    include("training/forecast.jl")
    include("training/iterative.jl")
end
using .Training
export fit, predict, forecast
export extrapolate_timestamps, iterative_predict

# Evaluation module
module Evaluation
    using ..TimeSeriesKit
    using ..Models
    using ..Training
    using Statistics
    
    include("evaluation/metrics.jl")
    include("evaluation/backtest.jl")
end
using .Evaluation
export mse, mae, rmse
export cross_validate, grid_search

# Extensions
function plot_timeseries end
function plot_residuals end
function plot_acf_pacf end

# Shorthand aliases
const plot_ts = plot_timeseries

export plot_timeseries, plot_ts, plot_residuals, plot_ac, plot_acf_pacf

end
