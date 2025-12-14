# Iterative prediction functions

using ..TimeSeriesKit: TimeSeries, LinearModel, SESModel
using ..Models: fit
using ..Training: predict

"""
    iterative_predict(model::Union{LinearModel, SESModel}, ts::TimeSeries, horizon::Int)

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
- `model::Union{LinearModel, SESModel}`: A model instance (will be refitted at each step)
- `ts::TimeSeries`: The historical time series data (must have at least 2 points)
- `horizon::Int`: Number of steps to predict ahead beyond the historical data

# Returns
- `TimeSeries`: A TimeSeries containing the iterative predictions with extrapolated timestamps
"""
function iterative_predict(model::Union{LinearModel, SESModel}, ts::TimeSeries, horizon::Int)
    n = length(ts)
    
    if n < 2
        throw(ArgumentError("Time series must have at least 2 points"))
    end
    
    if horizon < 1
        throw(ArgumentError("Horizon must be at least 1"))
    end
    
    # Total predictions: from point 3 to end of ts, plus horizon into future
    total_predictions = (n - 2) + horizon
    predictions = zeros(total_predictions)
    
    # Generate all x values we'll predict
    step = ts.timestamps[end] - ts.timestamps[end-1]
    future_x = [ts.timestamps[end] + step * h for h in 1:horizon]
    all_pred_x = [ts.timestamps[3:end]; future_x]
    
    pred_idx = 1
    
    # Phase 1: Predict within historical data (point 3 to n)
    for i in 3:n
        # Train on points 1 to i-1
        train_x = ts.timestamps[1:i-1]
        train_y = ts.values[1:i-1]
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
        train_ts = TimeSeries(ts.timestamps, ts.values)
        
        # Fit model
        fit(model, train_ts)
        
        # Predict future point
        next_x = future_x[h]
        pred_ts = predict(model, [next_x])
        predictions[pred_idx] = pred_ts.values[1]
        pred_idx += 1
    end
    
    return TimeSeries(all_pred_x, predictions)
end

export iterative_predict
