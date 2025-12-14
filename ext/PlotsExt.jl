module PlotsExt

using Plots
using TimeSeriesKit

"""
    plot_timeseries(ts::TimeSeries; kwargs...)

Plot a single time series.
"""
function TimeSeriesKit.plot_timeseries(ts::TimeSeriesKit.TimeSeries; 
                                       title::String="Time Series", 
                                       xlabel::String="X", 
                                       ylabel::String="Value",
                                       label::String="Data",
                                       color=:blue,
                                       kwargs...)
    x_values = ts.timestamps
    
    return Plots.plot(x_values, ts.values;
                     title=title,
                     xlabel=xlabel,
                     ylabel=ylabel,
                     label=label,
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
                                       title::String="Time Series with Forecast", 
                                       show_history::Int=50, 
                                       kwargs...)
    n_history = min(show_history, length(ts))
    
    # Get historical data
    hist_x = ts.timestamps[end-n_history+1:end]
    hist_y = ts.values[end-n_history+1:end]
    
    # Plot historical data
    p = Plots.plot(hist_x, hist_y;
                   label="Historical",
                   linewidth=2,
                   color=:blue,
                   title=title,
                   xlabel="X",
                   ylabel="Value",
                   legend=:best,
                   markershape=:circle,
                   kwargs...)
    
    # Add out-of-sample predictions/forecasts
    Plots.scatter!(p, out_of_sample.timestamps, out_of_sample.values;
               label="Out-of-sample",
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
                                       show_history::Int=50, 
                                       kwargs...)
    n_history = min(show_history, length(ts))
    
    # Get historical data
    hist_x = ts.timestamps[end-n_history+1:end]
    hist_y = ts.values[end-n_history+1:end]
    
    # Plot historical data
    p = Plots.plot(hist_x, hist_y;
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
    Plots.scatter!(p, in_sample.timestamps, in_sample.values;
                  label="In-sample",
                  markersize=4,
                  color=:green,
                  markershape=:diamond,
                  alpha=0.6)
    
    # Add out-of-sample predictions/forecasts
    Plots.scatter!(p, out_of_sample.timestamps, out_of_sample.values;
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
    
    p = Plots.plot(1:length(residuals), residuals;
                   label="Residuals",
                   linewidth=1.5,
                   color=:blue,
                   title=title,
                   xlabel="Time",
                   ylabel="Residual",
                   legend=:best,
                   kwargs...)
    
    # Add zero line
    Plots.hline!(p, [0];
                color=:red,
                linestyle=:dash,
                linewidth=1.5,
                label="Zero")
    
    # Add scatter plot overlay
    Plots.scatter!(p, 1:length(residuals), residuals;
                  color=:blue,
                  alpha=0.5,
                  markersize=3,
                  label="")
    
    return p
end

"""
    plot_ac(ts::TimeSeries; max_lag::Int=20, kwargs...)

Plot both ACF and PACF side by side using Plots.jl.
"""
function TimeSeriesKit.plot_ac(ts::TimeSeriesKit.TimeSeries; 
                               max_lag::Int=20, 
                               kwargs...)
    values = ts.values
    n = length(values)
    
    if max_lag >= n
        max_lag = n - 1
    end
    
    # Calculate ACF
    mean_val = sum(values) / n
    c0 = sum((values .- mean_val).^2) / n
    
    acf = zeros(max_lag + 1)
    acf[1] = 1.0  # ACF at lag 0 is always 1
    
    for lag in 1:max_lag
        ck = sum((values[1:n-lag] .- mean_val) .* (values[lag+1:n] .- mean_val)) / n
        acf[lag + 1] = ck / c0
    end
    
    # Calculate PACF using Durbin-Levinson algorithm
    pacf = zeros(max_lag + 1)
    pacf[1] = 1.0  # PACF at lag 0 is always 1
    
    if max_lag > 0
        pacf[2] = acf[2]  # PACF at lag 1 equals ACF at lag 1
        
        # Durbin-Levinson recursion
        phi = zeros(max_lag, max_lag)
        phi[1, 1] = acf[2]
        
        for k in 2:max_lag
            # Calculate phi[k, k]
            numerator = acf[k + 1]
            for j in 1:k-1
                numerator -= phi[k-1, j] * acf[k - j + 1]
            end
            
            denominator = 1.0
            for j in 1:k-1
                denominator -= phi[k-1, j] * acf[j + 1]
            end
            
            phi[k, k] = numerator / denominator
            pacf[k + 1] = phi[k, k]
            
            # Update phi values
            for j in 1:k-1
                phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]
            end
        end
    end
    
    # Confidence interval (95%)
    # Under the null hypothesis of white noise (no correlation), ACF/PACF coefficients
    # are approximately normally distributed: ACF(k) ~ N(0, 1/n)
    # For 95% confidence: ±1.96 * SE = ±1.96 / √n
    # Values outside these bounds suggest significant autocorrelation at that lag
    ci = 1.96 / sqrt(n)
    
    lags = 0:max_lag
    
    # Create ACF plot
    p1 = Plots.bar(lags, acf;
                   title="ACF",
                   xlabel="Lag",
                   ylabel="ACF",
                   label="",
                   color=:steelblue,
                   legend=:topright,
                   ylims=(-1, 1),
                   kwargs...)
    
    Plots.hline!(p1, [ci, -ci];
                color=:red,
                linestyle=:dash,
                linewidth=1.5,
                label="95% CI")
    
    Plots.hline!(p1, [0];
                color=:black,
                linewidth=1,
                label="")
    
    # Create PACF plot
    p2 = Plots.bar(lags, pacf;
                   title="PACF",
                   xlabel="Lag",
                   ylabel="PACF",
                   label="",
                   color=:coral,
                   legend=:topright,
                   ylims=(-1, 1),
                   kwargs...)
    
    Plots.hline!(p2, [ci, -ci];
                color=:red,
                linestyle=:dash,
                linewidth=1.5,
                label="95% CI")
    
    Plots.hline!(p2, [0];
                color=:black,
                linewidth=1,
                label="")
    
    # Combine plots side by side
    return Plots.plot(p1, p2; layout=(1, 2), size=(900, 400))
end

end
