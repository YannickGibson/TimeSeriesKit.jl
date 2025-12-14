# Backtesting utilities for time series models

using ..TimeSeriesKit: AbstractTimeSeriesModel, TimeSeries
using ..TimeSeriesKit: fit, forecast
using Statistics

"""
    RollingForecastResult

Stores results from a rolling forecast backtest.
"""
struct RollingForecastResult
    forecasts::Vector{Vector{Float64}}
    actuals::Vector{Vector{Float64}}
    errors::Dict{Symbol, Vector{Float64}}
    
    function RollingForecastResult(forecasts, actuals, errors)
        new(forecasts, actuals, errors)
    end
end

"""
    rolling_forecast(model::AbstractTimeSeriesModel, ts::TimeSeries; 
                    train_size::Int, horizon::Int=1, step::Int=1)

Perform rolling forecast backtesting.

# Arguments
- `model`: The model to evaluate
- `ts`: The time series data
- `train_size`: Initial training window size
- `horizon`: Forecast horizon (default: 1)
- `step`: Number of steps to roll forward (default: 1)

# Returns
- `RollingForecastResult`: Object containing forecasts, actuals, and errors
"""
function rolling_forecast(model_template::AbstractTimeSeriesModel, ts::TimeSeries; 
                         train_size::Int, horizon::Int=1, step::Int=1)
    n = length(ts)
    
    if train_size >= n
        throw(ArgumentError("train_size must be less than time series length"))
    end
    
    if horizon < 1 || step < 1
        throw(ArgumentError("horizon and step must be at least 1"))
    end
    
    forecasts = Vector{Vector{Float64}}()
    actuals = Vector{Vector{Float64}}()
    
    # Rolling window forecasting
    current_pos = train_size
    while current_pos + horizon <= n
        # Create training data
        train_values = ts.values[1:current_pos]
        train_ts = TimeSeries(train_values)
        
        # Create a fresh model instance (copy parameters from template)
        model = deepcopy(model_template)
        
        # Fit model on training data
        fit(model, train_ts)
        
        # Make forecast (returns TimeSeries)
        forecast_ts = forecast(model, train_ts, horizon)
        
        # Get actual values
        actual = ts.values[current_pos+1:current_pos+horizon]
        
        push!(forecasts, forecast_ts.values)
        push!(actuals, actual)
        
        # Move forward
        current_pos += step
    end
    
    # Calculate errors for each horizon step
    errors = Dict{Symbol, Vector{Float64}}()
    
    if length(forecasts) > 0
        # Calculate metrics across all forecasts
        all_forecasts = reduce(vcat, forecasts)
        all_actuals = reduce(vcat, actuals)
        
        # Import metrics from the parent module
        mse_val = TimeSeriesKit.Evaluation.mse(all_actuals, all_forecasts)
        mae_val = TimeSeriesKit.Evaluation.mae(all_actuals, all_forecasts)
        rmse_val = TimeSeriesKit.Evaluation.rmse(all_actuals, all_forecasts)
        
        errors[:mse] = [mse_val]
        errors[:mae] = [mae_val]
        errors[:rmse] = [rmse_val]
    end
    
    return RollingForecastResult(forecasts, actuals, errors)
end

"""
    print_backtest_summary(result::RollingForecastResult)

Print a summary of backtest results.
"""
function print_backtest_summary(result::RollingForecastResult)
    println("Rolling Forecast Backtest Results")
    println("=" ^ 40)
    println("Number of forecasts: ", length(result.forecasts))
    
    if length(result.forecasts) > 0
        println("\nError Metrics:")
        for (metric, values) in result.errors
            println("  ", uppercase(String(metric)), ": ", round(values[1], digits=4))
        end
    end
end

# Export backtest functions and types
export rolling_forecast, RollingForecastResult, print_backtest_summary
