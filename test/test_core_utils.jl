using Test
using TimeSeriesKit
using Statistics

@testset "Core Utilities" begin
    
    @testset "validate_timeseries" begin
        # Valid time series
        ts = TimeSeries([1.0, 2.0, 3.0])
        @test validate_timeseries(ts) == true
        
        # Empty time series should throw
        ts_empty = TimeSeries(Float64[])
        @test_throws ArgumentError validate_timeseries(ts_empty)
        
        # NaN values should warn but return true
        ts_nan = TimeSeries([1.0, NaN, 3.0])
        @test_logs (:warn, r"NaN or Inf") validate_timeseries(ts_nan)
        
        # Inf values should warn but return true
        ts_inf = TimeSeries([1.0, Inf, 3.0])
        @test_logs (:warn, r"NaN or Inf") validate_timeseries(ts_inf)
    end
    
    @testset "differentiate - order 2" begin
        ts = TimeSeries([1.0, 3.0, 6.0, 10.0, 15.0])
        
        diff_ts = differentiate(ts; order=2)
        @test length(diff_ts) == 3
        @test diff_ts.values ≈ [1.0, 1.0, 1.0]
        @test occursin("2 times", diff_ts.name)
    end
    
    @testset "differentiate - with timestamps" begin
        ts = TimeSeries([10, 20, 30, 40], [1.0, 3.0, 6.0, 10.0])
        
        diff_ts = differentiate(ts)
        @test diff_ts.timestamps == [20, 30, 40]
        @test diff_ts.values ≈ [2.0, 3.0, 4.0]
        
        # Order 2
        diff_ts2 = differentiate(ts; order=2)
        @test diff_ts2.timestamps == [30, 40]
        @test diff_ts2.values ≈ [1.0, 1.0]
    end
    
    @testset "differentiate - with name" begin
        ts = TimeSeries([1.0, 2.0, 3.0], name="My Series")
        
        diff_ts = differentiate(ts)
        @test diff_ts.name == "My Series (Differentiated)"
        
        diff_ts2 = differentiate(ts; order=2)
        @test diff_ts2.name == "My Series (Differentiated 2 times)"
    end
    
    @testset "differentiate - error cases" begin
        ts = TimeSeries([1.0, 2.0])
        
        # Order < 1 should throw
        @test_throws ArgumentError differentiate(ts; order=0)
        @test_throws ArgumentError differentiate(ts; order=-1)
        
        # Too short series
        ts_short = TimeSeries([1.0])
        @test_throws ArgumentError differentiate(ts_short)
    end
    
    @testset "integrate - order 1" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0])
        
        int_ts = integrate(ts)
        @test length(int_ts) == 4
        @test int_ts.values ≈ [1.0, 3.0, 6.0, 10.0]
        @test occursin("Integrated", int_ts.name)
    end
    
    @testset "integrate - order 2" begin
        ts = TimeSeries([1.0, 1.0, 1.0, 1.0])
        
        int_ts = integrate(ts; order=2)
        @test int_ts.values ≈ [1.0, 3.0, 6.0, 10.0]
        @test occursin("2 times", int_ts.name)
    end
    
    @testset "integrate - preserves timestamps" begin
        ts = TimeSeries([10, 20, 30, 40], [1.0, 2.0, 3.0, 4.0])
        
        int_ts = integrate(ts)
        @test int_ts.timestamps == ts.timestamps
        @test int_ts.values ≈ [1.0, 3.0, 6.0, 10.0]
    end
    
    @testset "integrate - with name" begin
        ts = TimeSeries([1.0, 2.0, 3.0], name="My Series")
        
        int_ts = integrate(ts)
        @test int_ts.name == "My Series (Integrated)"
        
        int_ts2 = integrate(ts; order=2)
        @test int_ts2.name == "My Series (Integrated 2 times)"
    end
    
    @testset "differentiate/integrate roundtrip" begin
        ts = TimeSeries([1.0, 3.0, 6.0, 10.0])
        
        # Differentiate then integrate should recover original (with length adjustment)
        diff_ts = differentiate(ts)
        int_ts = integrate(diff_ts)
        @test int_ts.values ≈ ts.values[2:end] .- ts.values[1]
    end
end
