module PlotsExt

using Plots
using TimeSeriesKit
using StatsBase

"""
    plot_timeseries(ts::TimeSeries; kwargs...)

Plot a single time series.
"""
function TimeSeriesKit.plot_timeseries(ts::TimeSeriesKit.TimeSeries; 
                                       title::String="Time Series", 
                                       xlabel::String="X", 
                                       ylabel::String="Value",
                                       color=:blue,
                                       kwargs...)
    x_values = ts.timestamps
    
    return plot(x_values, ts.values;
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                label=ts.name == "" ? "Data" : ts.name,
                color=color,
                linewidth=2,
                legend=:best,
                markershape=:circle,
                kwargs...)
end


"""
    plot_timeseries(ts::TimeSeries, out_of_sample::TimeSeries; kwargs...)

Plot historical data with out-of-sample predictions/forecasts.
"""
function TimeSeriesKit.plot_timeseries(ts::TimeSeriesKit.TimeSeries,
                                       out_of_sample::TimeSeriesKit.TimeSeries; 
                                       title::String="Time Series with Predictions", 
                                       show_history::Union{Int, Nothing}=nothing, 
                                       kwargs...)
    n_history = show_history === nothing ? length(ts) : min(show_history, length(ts))
    
    # Get historical data
    hist_x = ts.timestamps[end-n_history+1:end]
    hist_y = ts.values[end-n_history+1:end]
    
    # Plot historical data
    p = plot(hist_x, hist_y;
             label=ts.name == "" ? "Data" : ts.name,
             linewidth=2,
             color=:blue,
             title=title,
             xlabel="X",
             ylabel="Value",
             legend=:best,
             markershape=:circle,
             kwargs...)
    
    # Add out-of-sample predictions/forecasts
    scatter!(p, out_of_sample.timestamps, out_of_sample.values;
             label=out_of_sample.name == "" ? "Out-of-sample" : out_of_sample.name,
             markersize=6,
             color=:red,
             markershape=:cross)
    
    return p
end

"""
    plot_timeseries(ts::TimeSeries, in_sample::TimeSeries, out_of_sample::TimeSeries; kwargs...)

Plot historical data with in-sample predictions and out-of-sample predictions/forecasts.
"""
function TimeSeriesKit.plot_timeseries(ts::TimeSeriesKit.TimeSeries,
                                       in_sample::TimeSeriesKit.TimeSeries,
                                       out_of_sample::TimeSeriesKit.TimeSeries; 
                                       title::String="Time Series with Predictions", 
                                       show_history::Union{Int, Nothing}=nothing, 
                                       kwargs...)
    n_history = show_history === nothing ? length(ts) : min(show_history, length(ts))
    
    # Get historical data
    hist_x = ts.timestamps[end-n_history+1:end]
    hist_y = ts.values[end-n_history+1:end]
    
    # Plot historical data
    p = plot(hist_x, hist_y;
             label="Historical",
             linewidth=2,
             color=:blue,
             title=title,
             xlabel="X",
             ylabel="Value",
             legend=:best,
             markershape=:circle,
             kwargs...)
    
    # Add in-sample predictions (scatter for fitted values)
    scatter!(p, in_sample.timestamps, in_sample.values;
             label="In-sample",
             markersize=4,
             color=:green,
             markershape=:diamond,
             alpha=0.6)
    
    # Add out-of-sample predictions/forecasts
    scatter!(p, out_of_sample.timestamps, out_of_sample.values;
             label="Out-of-sample",
             markersize=6,
             color=:red,
             markershape=:cross)
    
    return p
end


"""
    plot_residuals(model::AbstractTimeSeriesModel; title::String="Residuals", kwargs...)

Plot residuals from a fitted model using Plots.jl.
"""
function TimeSeriesKit.plot_residuals(model::TimeSeriesKit.AbstractTimeSeriesModel; 
                                      title::String="Residuals", 
                                      kwargs...)
    if !model.state.is_fitted
        throw(ErrorException("Model must be fitted before plotting residuals"))
    end
    
    residuals = model.state.residuals
    
    p = plot(1:length(residuals), residuals;
             label="Residuals",
             linewidth=1.5,
             color=:blue,
             title=title,
             xlabel="Time",
             ylabel="Residual",
             legend=:best,
             kwargs...)
    
    # Add zero line
    hline!(p, [0];
           color=:red,
           linestyle=:dash,
           linewidth=1.5,
           label="Zero")
    
    # Add scatter plot overlay
    scatter!(p, 1:length(residuals), residuals;
             color=:blue,
             alpha=0.5,
             markersize=3,
             label="")
    
    return p
end

"""
    plot_acf_pacf(ts::TimeSeries; lags::Int=20)

Plot time series with ACF and PACF in a custom layout: time series on top, ACF and PACF side by side below.
"""

function TimeSeriesKit.plot_acf_pacf(ts::TimeSeriesKit.TimeSeries; lags::Int=20)
    acf_values = autocor(ts.values, 0:lags)
    println("Pacf input vals: $(ts.values) and lags: $(0:lags)")
    pacf_values = pacf(ts.values, 0:lags)
    
    # Create plots with adjusted margins
    bar1 = bar(0:lags, acf_values, title="ACF", xlabel="Lag", ylabel="Autocorrelation", 
               ylim=(-1.2, 1.2), left_margin=5Plots.mm, right_margin=2Plots.mm, bottom_margin=5Plots.mm)
    bar2 = bar(0:lags, pacf_values, title="PACF", xlabel="Lag", ylabel="Partial Autocorrelation", 
               ylim=(-1.2, 1.2), left_margin=2Plots.mm, right_margin=5Plots.mm, bottom_margin=5Plots.mm)
    ts_plot = plot(ts.timestamps, ts.values, title=ts.name == "" ? "Time Series" : ts.name, 
                   xlabel="Time", ylabel="Value", 
                   legend=false, left_margin=5Plots.mm, right_margin=5Plots.mm, bottom_margin=5Plots.mm)
    
    # Create layout
    layout = @layout [a{0.3h}; [b c]]
    return plot(ts_plot, bar1, bar2, layout=layout, size=(1000, 400))
end

# Shorthand alias
const plot_ts = plot_timeseries

end
