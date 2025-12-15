# Abstract model definitions

using ..TimeSeriesKit: AbstractTimeSeriesModel, ModelState, TimeSeries

"""
    is_fitted(model)

Check if a model has been fitted to data.
"""
function is_fitted(model::AbstractTimeSeriesModel)
    return hasfield(typeof(model), :state) && model.state.is_fitted
end

"""
    get_parameters(model)

Get the parameters of a fitted model.
"""
function get_parameters(model::AbstractTimeSeriesModel)
    if !is_fitted(model)
        throw(ErrorException("Model has not been fitted yet"))
    end
    return model.state.parameters
end

"""
    get_residuals(model)

Get the residuals from a fitted model.
"""
function get_residuals(model::AbstractTimeSeriesModel)
    if !is_fitted(model)
        throw(ErrorException("Model has not been fitted yet"))
    end
    return model.state.residuals
end

"""
    min_train_size(model)

Get the minimum number of data points required to train this model.
Each model type must implement this method.
"""
function min_train_size(model::AbstractTimeSeriesModel)
    error("min_train_size not implemented for $(typeof(model))")
end

export is_fitted, get_parameters, get_residuals, min_train_size
