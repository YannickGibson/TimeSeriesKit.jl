# Backtesting utilities for time series models

using ..TimeSeriesKit: AbstractTimeSeriesModel, TimeSeries
using ..TimeSeriesKit: fit, iterative_predict
using Statistics
using DataFrames

"""
    cross_validate(model::AbstractTimeSeriesModel, ts::TimeSeries, n_splits::Int=5; verbose::Bool=false)

Perform time series cross-validation for a model.

# Arguments
- `model::AbstractTimeSeriesModel`: The model to validate
- `ts::TimeSeries`: The time series data
- `n_splits::Int=5`: Number of splits for cross-validation
- `verbose::Bool=false`: Whether to print detailed information

# Returns
A named tuple containing:
- `mean_rmse`: Mean RMSE across all folds
- `std_rmse`: Standard deviation of RMSE
- `mean_mae`: Mean MAE across all folds
- `std_mae`: Standard deviation of MAE
- `cv_scores`: Vector of RMSE scores for each fold

# Example
```julia
model = LinearModel()
ts = TimeSeries("data.csv", "Country")
results = cross_validate(model, ts, 5, verbose=true)
println("Mean RMSE: \$(results.mean_rmse)")
```
"""
function cross_validate(model::AbstractTimeSeriesModel, ts::TimeSeries, n_splits::Int=5; verbose::Bool=false)
    n = length(ts.values)
    
    if n_splits < 2
        throw(ArgumentError("n_splits must be at least 2"))
    end
    
    if n < n_splits + 1
        throw(ArgumentError("Time series too short for $(n_splits) splits"))
    end
    
    rmse_scores = Float64[]
    mae_scores = Float64[]
    
    # Calculate split sizes - start with minimum training size
    min_train_size = div(n, n_splits + 1)
    
    for i in 1:n_splits
        # Growing training window approach
        train_end = min_train_size + i * div(n - min_train_size, n_splits)
        test_end = min(train_end + div(n - min_train_size, n_splits), n)
        
        if train_end >= test_end
            continue
        end
        
        # Split the data
        train_values = ts.values[1:train_end]
        train_timestamps = ts.timestamps[1:train_end]
        test_values = ts.values[train_end+1:test_end]
        test_timestamps = ts.timestamps[train_end+1:test_end]
        
        train_ts = TimeSeries(train_timestamps, train_values; name=ts.name)
        test_ts = TimeSeries(test_timestamps, test_values; name=ts.name)
        
        if verbose
            println("Split $(i): Train len: $(length(train_values)), Test len: $(length(test_values))")
        end
        
        try
            # Create a fresh model instance for each fold
            # Use deepcopy to preserve model parameters
            model_copy = deepcopy(model)
            # Reset the state but keep parameters
            model_copy.state = ModelState()
            
            # Fit the model
            fit(model_copy, train_ts)
            
            # Forecast
            forecast = predict(model_copy, test_ts.timestamps)
            
            # Calculate metrics
            rmse_val = TimeSeriesKit.Evaluation.rmse(test_ts, forecast)
            mae_val = TimeSeriesKit.Evaluation.mae(test_ts, forecast)
            
            push!(rmse_scores, rmse_val)
            push!(mae_scores, mae_val)
            
            if verbose
                println("  RMSE: $(round(rmse_val, digits=4)), MAE: $(round(mae_val, digits=4))")
            end
            
        catch e
            if verbose
                println("  Fold $(i) failed: $(e)")
            end
            continue
        end
    end
    
    if isempty(rmse_scores)
        throw(ErrorException("All cross-validation folds failed"))
    end
    
    return (
        mean_rmse = mean(rmse_scores),
        std_rmse = std(rmse_scores),
        mean_mae = mean(mae_scores),
        std_mae = std(mae_scores),
        cv_scores = rmse_scores
    )
end

"""
    grid_search(model_configs::Dict, ts::TimeSeries, n_splits::Int=5; verbose::Bool=false)

Perform grid search over multiple models and their parameter ranges.

# Arguments
- `model_configs::Dict`: Dictionary with model types as keys and parameter grids as values
  Format: Dict(ModelType => Dict(:param1 => [values...], :param2 => [values...]))
- `ts::TimeSeries`: The time series data
- `n_splits::Int=5`: Number of splits for cross-validation
- `verbose::Bool=false`: Whether to print detailed information

# Returns
A named tuple containing:
- `best_model`: The best performing model instance with optimal parameters
- `best_score`: The RMSE score of the best model
- `best_params`: Dictionary of the best parameters
- `results_df`: DataFrame with all model configurations and their scores

# Example
```julia
configs = Dict(
    LinearModel => Dict(),  # No parameters to tune
    SESModel => Dict(:alpha => [0.1, 0.3, 0.5, 0.7, 0.9])
)
ts = TimeSeries("data.csv", "Country")
result = grid_search(configs, ts, 5, verbose=true)
println("Best model: \$(typeof(result.best_model))")
println("Best score: \$(result.best_score)")
println(result.results_df)
```
"""
function grid_search(model_configs::Dict, ts::TimeSeries, n_splits::Int=5; verbose::Bool=false)
    results = []
    best_score = Inf
    best_model = nothing
    best_params = Dict{Symbol, Any}()
    
    # Cartesian product helper function
    function cartesian_product(arrays)
        if isempty(arrays)
            return [[]]
        end
        result = []
        for value in arrays[1]
            for rest in cartesian_product(arrays[2:end])
                push!(result, [value, rest...])
            end
        end
        return result
    end
    
    for (ModelType, param_grid) in model_configs
        if verbose
            println("\n" * "="^60)
            println("Testing model: $(ModelType)")
            println("="^60)
        end
        
        # Generate all parameter combinations (empty grid yields one empty combination)
        param_names = collect(keys(param_grid))
        param_values = [param_grid[k] for k in param_names]
        combinations = cartesian_product(param_values)
        
        for combo in combinations
            params = Dict(zip(param_names, combo))
            
            if verbose
                params_str = join(["$k=$v" for (k, v) in params], ", ")
                println("\nTesting: $(params_str)")
            end
            
            try
                # Create model with parameters
                model = ModelType(; params...)
                
                # Cross-validate
                cv_result = cross_validate(model, ts, n_splits, verbose=false)
                score = cv_result.mean_rmse
                
                push!(results, (
                    model_name = string(ModelType),
                    params = copy(params),
                    mean_rmse = cv_result.mean_rmse,
                    std_rmse = cv_result.std_rmse,
                    mean_mae = cv_result.mean_mae,
                    std_mae = cv_result.std_mae
                ))
                
                if score < best_score
                    best_score = score
                    best_model = model
                    best_params = copy(params)
                end
                
                if verbose
                    println("✓ Score: $(round(score, digits=4))")
                end
                
            catch e
                if verbose
                    println("✗ Failed: $(e)")
                end
            end
        end
    end
    
    if best_model === nothing
        throw(ErrorException("Grid search failed: no valid model configurations found"))
    end
    
    # Helper function to format parameters nicely
    function format_params(params::Dict)
        if isempty(params)
            return "default"
        end
        return join(["$k=$v" for (k, v) in sort(collect(params), by=x->string(x[1]))], ", ")
    end
    
    # Create DataFrame from results
    results_df = DataFrame(
        model = [r.model_name for r in results],
        mean_rmse = [r.mean_rmse for r in results],
        std_rmse = [r.std_rmse for r in results],
        mean_mae = [r.mean_mae for r in results],
        std_mae = [r.std_mae for r in results],
        params = [format_params(r.params) for r in results],
    )
    
    # Sort by RMSE (best first)
    sort!(results_df, :mean_rmse)
    
    if verbose
        println("\n" * "="^60)
        println("Grid Search Complete!")
        println("="^60)
        println("Best Model: $(typeof(best_model))")
        println("Best Parameters: $(best_params)")
        println("Best RMSE: $(round(best_score, digits=4))")
    end
    
    return (
        best_model = best_model,
        best_score = best_score,
        best_params = best_params,
        results_df = results_df
    )
end

# Export backtest functions
export cross_validate, grid_search
