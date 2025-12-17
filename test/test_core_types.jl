using TimeSeriesKit
using Test

@testset "Core Types" begin
    @testset "TimeSeries - Basic Construction" begin
        # Test construction with values only
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0])
        @test length(ts) == 5
        @test ts.values == [1.0, 2.0, 3.0, 4.0, 5.0]
        @test ts.timestamps == [1, 2, 3, 4, 5]
        @test ts.name == ""
        
        # Test with custom name
        ts_named = TimeSeries([1.0, 2.0, 3.0]; name="test series")
        @test ts_named.name == "test series"
    end
    
    @testset "TimeSeries - Construction with Timestamps" begin
        # Test with Vector timestamps
        ts = TimeSeries([10, 20, 30], [1.0, 2.0, 3.0])
        @test ts.timestamps == [10, 20, 30]
        @test ts.values == [1.0, 2.0, 3.0]
        
        # Test with UnitRange timestamps
        ts_range = TimeSeries(1:5, [10.0, 20.0, 30.0, 40.0, 50.0])
        @test ts_range.timestamps == [1, 2, 3, 4, 5]
        @test ts_range.values == [10.0, 20.0, 30.0, 40.0, 50.0]
        
        # Test with custom name
        ts_named = TimeSeries([1, 2, 3], [1.0, 2.0, 3.0]; name="custom")
        @test ts_named.name == "custom"
    end
    
    @testset "TimeSeries - Error Handling" begin
        # Mismatched lengths for Vector constructor
        @test_throws ArgumentError TimeSeries([1, 2], [1.0, 2.0, 3.0])
        
        # Mismatched lengths for UnitRange constructor
        @test_throws ArgumentError TimeSeries(1:3, [1.0, 2.0])
    end
    
    @testset "TimeSeries - Indexing" begin
        ts = TimeSeries([10.0, 20.0, 30.0, 40.0, 50.0])
        
        # Single index - returns scalar
        @test ts[1] == 10.0
        @test ts[3] == 30.0
        @test ts[end] == 50.0
        
        # Range index - returns new TimeSeries
        subset = ts[2:4]
        @test subset isa TimeSeries
        @test subset.values == [20.0, 30.0, 40.0]
        @test subset.timestamps == [2, 3, 4]
        @test subset.name == "[Subset]"
        
        # Vector index - returns new TimeSeries
        subset2 = ts[[1, 3, 5]]
        @test subset2.values == [10.0, 30.0, 50.0]
        @test subset2.timestamps == [1, 3, 5]
        
        # Named series subset
        ts_named = TimeSeries([1.0, 2.0, 3.0]; name="original")
        subset_named = ts_named[1:2]
        @test subset_named.name == "original [Subset]"
    end
    
    @testset "TimeSeries - Base Methods" begin
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test length
        @test length(ts) == 5
        @test Base.length(ts) == 5
        
        # Test lastindex
        @test lastindex(ts) == 5
        @test ts[end] == 5.0
    end
    
    @testset "TimeSeries - Type Parameters" begin
        # Test with different numeric types
        ts_float64 = TimeSeries([1.0, 2.0, 3.0])
        @test eltype(ts_float64.values) == Float64
        
        ts_float32 = TimeSeries(Float32[1.0, 2.0, 3.0])
        @test eltype(ts_float32.values) == Float32
        
        ts_int = TimeSeries([1, 2, 3])
        @test eltype(ts_int.values) == Int
    end
    
    @testset "TimeSeries - Timestamps Types" begin
        # Test with different timestamp types
        ts_int = TimeSeries([1, 2, 3], [10.0, 20.0, 30.0])
        @test ts_int.timestamps isa Vector
        
        ts_float = TimeSeries([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        @test ts_float.timestamps isa Vector
        
        ts_string = TimeSeries(["2020", "2021", "2022"], [10.0, 20.0, 30.0])
        @test ts_string.timestamps isa Vector
    end
    
    @testset "ModelState" begin
        state = ModelState()
        
        # Test initial state
        @test state.parameters isa Dict{Symbol, Any}
        @test isempty(state.parameters)
        @test state.fitted_values === nothing
        @test state.residuals === nothing
        @test state.is_fitted == false
        
        # Test state modification
        state.parameters[:test] = 1.0
        @test state.parameters[:test] == 1.0
        
        state.is_fitted = true
        @test state.is_fitted == true
        
        state.fitted_values = [1.0, 2.0, 3.0]
        @test state.fitted_values == [1.0, 2.0, 3.0]
        
        state.residuals = [0.1, -0.1, 0.05]
        @test state.residuals == [0.1, -0.1, 0.05]
    end
    
    @testset "AbstractTimeSeriesModel" begin
        # Test that it's defined and is abstract
        @test AbstractTimeSeriesModel isa Type
        @test isabstracttype(AbstractTimeSeriesModel)
    end
    
    @testset "TimeSeries - CSV Loading" begin
        csv_path = joinpath(@__DIR__, "..", "examples", "data", "emissions.csv")
        
        # Test loading from CSV with country filter
        ts = TimeSeries(csv_path, "Czechia")
        @test ts isa TimeSeries
        @test eltype(ts.values) == Float32
        @test length(ts) > 0
        @test ts.name == "Czechia"
        @test ts.timestamps isa Vector
        
        # Test with custom name
        ts_named = TimeSeries(csv_path, "Czechia"; name="Czech Emissions")
        @test ts_named.name == "Czech Emissions"
        
        # Test data is sorted by time
        @test issorted(ts.timestamps)
        
        # Test error for non-existent country
        @test_throws ArgumentError TimeSeries(csv_path, "NOTEXIST")
    end
end
