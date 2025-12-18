# Iterative prediction functions

using ..TimeSeriesKit: AbstractTimeSeriesModel, TimeSeries, PredictionResult, min_train_size, LinearModel, RidgeModel, BayesianARModel
using ..Training: predict, fit

"""
    iterative_predict(model::AbstractTimeSeriesModel, ts::TimeSeries, horizon::Int; return_uncertainty::Bool=false, use_predictions::Bool=false)

Iteratively train and predict for a given horizon using expanding window.

Starting with the first 2 data points:
1. Train a model on the first 2 points
2. Predict the 3rd point
3. Add the actual 3rd point to training data
4. Train on the first 3 points, predict the 4th
5. Continue until we've used all historical data
6. Then continue predicting into the future for the remaining horizon

This creates an expanding window forecast where the model is retrained
at each step with one additional data point.

# Arguments
- `model::AbstractTimeSeriesModel`: A model instance (will be refitted at each step)
- `ts::TimeSeries`: The historical time series data (must have at least 2 points)
- `horizon::Int`: Number of steps to predict ahead beyond the historical data
- `return_uncertainty::Bool=false`: If true, returns PredictionResult with uncertainty estimates (LinearModel only)
- `use_predictions::Bool=false`: If true, future predictions include previous predictions in training data

# Returns
- `TimeSeries`: A TimeSeries containing the iterative predictions with extrapolated timestamps
- `PredictionResult`: If return_uncertainty=true, contains predictions with variance estimates
"""
function iterative_predict(model::AbstractTimeSeriesModel, ts::TimeSeries, horizon::Int; return_uncertainty::Bool=false, use_predictions::Bool=false)
    n = length(ts)
    
    min_size = min_train_size(model)
    if n < min_size
        throw(ArgumentError("Time series must have at least $(min_size) points for this model"))
    end
    
    if horizon < 1
        throw(ArgumentError("Horizon must be at least 1"))
    end
    
    # Total predictions: from point (min_size+1) to end of ts, plus horizon into future
    total_predictions = (n - min_size) + horizon
    predictions = zeros(total_predictions)
    pred_variances = return_uncertainty ? zeros(total_predictions) : nothing
    
    # Generate all x values we'll predict
    step = ts.timestamps[end] - ts.timestamps[end-1]
    future_x = [ts.timestamps[end] + step * h for h in 1:horizon]
    all_pred_x = [ts.timestamps[min_size+1:end]; future_x]
    
    pred_idx = 1
    
    # Phase 1: Predict within historical data (point min_size+1 to n)
    for i in (min_size+1):n
        # Train on points to i-1 
        knowledge_start = 1
        if hasproperty(model, :sliding_window)
            knowledge_start = max(1, i - model.sliding_window)
        end

        train_x = ts.timestamps[knowledge_start:i-1]
        train_y = ts.values[knowledge_start:i-1]
        train_ts = TimeSeries(train_x, train_y)
        
        # Fit model
        fit(model, train_ts)
        
        # Predict point i (using actual x value)
        # Check if model supports uncertainty (LinearModel, RidgeModel, or BayesianARModel)
        if return_uncertainty && (model isa LinearModel || model isa RidgeModel || model isa BayesianARModel)
            pred_result = predict(model, [ts.timestamps[i]], return_uncertainty=true)
        else
            pred_result = predict(model, [ts.timestamps[i]])
        end
        
        if return_uncertainty && pred_result isa PredictionResult
            predictions[pred_idx] = pred_result.predictions.values[1]
            pred_variances[pred_idx] = pred_result.prediction_variance[1]
        else
            pred_ts = pred_result isa PredictionResult ? pred_result.predictions : pred_result
            predictions[pred_idx] = pred_ts.values[1]
        end
        pred_idx += 1
    end
    
    # Phase 2: Predict into the future (horizon steps beyond historical data)
    # Optionally include predictions in training data
    # Note: use_predictions should only be used with AR-type models, not LinearModel
    for h in 1:horizon
        # Determine training data range - for future predictions, use the most recent window
        knowledge_start = 1
        if hasproperty(model, :sliding_window)
            knowledge_start = max(1, n - model.sliding_window + 1)
        end

        # Build training data based on use_predictions flag
        # For LinearModel and RidgeModel, predictions lie exactly on the trend line
        # so adding them creates a singular matrix. Only use for AR-type models.
        if use_predictions && h > 1 && !(model isa LinearModel) && !(model isa RidgeModel)
            # Include previous predictions in training data
            # Number of predictions to include
            num_past_preds = min(h - 1, n + h - 1 - knowledge_start)
            
            if knowledge_start <= n
                # Include some historical data and some predictions
                train_x = vcat(ts.timestamps[knowledge_start:n], all_pred_x[n - min_size + 1 : n - min_size + h - 1])
                train_y = vcat(ts.values[knowledge_start:n], predictions[n - min_size + 1 : n - min_size + h - 1])
            else
                # Only use predictions (when sliding window is smaller than available predictions)
                pred_start_idx = knowledge_start - n + (n - min_size)
                train_x = all_pred_x[pred_start_idx : n - min_size + h - 1]
                train_y = predictions[pred_start_idx : n - min_size + h - 1]
            end
        else
            # Use only historical data (default behavior or for Linear/Ridge models)
            train_x = ts.timestamps[knowledge_start:n]
            train_y = ts.values[knowledge_start:n]
        end
        
        train_ts = TimeSeries(train_x, train_y)
        # Fit model
        fit(model, train_ts)
        
        # Predict future point
        next_x = future_x[h]
        
        # Check if model supports uncertainty (LinearModel, RidgeModel, or BayesianARModel)
        if return_uncertainty && (model isa LinearModel || model isa RidgeModel || model isa BayesianARModel)
            pred_result = predict(model, [next_x], return_uncertainty=true)
        else
            pred_result = predict(model, [next_x])
        end
        
        if return_uncertainty && pred_result isa PredictionResult
            predictions[pred_idx] = pred_result.predictions.values[1]
            pred_variances[pred_idx] = pred_result.prediction_variance[1]
        else
            pred_ts = pred_result isa PredictionResult ? pred_result.predictions : pred_result
            predictions[pred_idx] = pred_ts.values[1]
        end
        pred_idx += 1
    end
    
    # dynamically get the class name
    cls_name = nameof(typeof(model))
    new_name = "$(cls_name) (out-of-sample)"
    
    if return_uncertainty && pred_variances !== nothing
        result_ts = TimeSeries(all_pred_x, predictions; name = new_name)
        return PredictionResult(result_ts, pred_variances)
    else
        return TimeSeries(all_pred_x, predictions; name = new_name)
    end
end

export iterative_predict
