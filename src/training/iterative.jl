# Iterative prediction functions

using ..TimeSeriesKit: AbstractTimeSeriesModel, TimeSeries, min_train_size
using ..Training: predict, fit

"""
    iterative_predict(model::AbstractTimeSeriesModel, ts::TimeSeries, horizon::Int)

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

# Returns
- `TimeSeries`: A TimeSeries containing the iterative predictions with extrapolated timestamps
"""
function iterative_predict(model::AbstractTimeSeriesModel, ts::TimeSeries, horizon::Int)
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
        pred_ts = predict(model, [ts.timestamps[i]])
        predictions[pred_idx] = pred_ts.values[1]
        pred_idx += 1
    end
    
    # Phase 2: Predict into the future (horizon steps beyond historical data)
    for h in 1:horizon
        # Train on all historical data (points 1 to n)
        knowledge_start = 1
        if hasproperty(model, :sliding_window)
            knowledge_start = max(1, n - model.sliding_window + 1)
        end

        train_x = ts.timestamps[knowledge_start:n]
        train_y = ts.values[knowledge_start:n]
        train_ts = TimeSeries(train_x, train_y)
        # Fit model
        fit(model, train_ts)
        
        # Predict future point
        next_x = future_x[h]
        pred_ts = predict(model, [next_x])
        predictions[pred_idx] = pred_ts.values[1]
        pred_idx += 1
    end
    # dynamically get the class name
    cls_name = nameof(typeof(model))
    new_name = "$(cls_name) (out-of-sample)"
    return TimeSeries(all_pred_x, predictions; name = new_name)
end

export iterative_predict
