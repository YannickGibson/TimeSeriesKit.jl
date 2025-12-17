using Test
using TimeSeriesKit

@testset "Evaluation Metrics" begin
    
    @testset "mse - Vector" begin
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        predicted = [1.1, 2.2, 2.9, 4.3, 4.8]
        
        mse_value = mse(actual, predicted)
        expected = mean(([1.0, 2.0, 3.0, 4.0, 5.0] .- [1.1, 2.2, 2.9, 4.3, 4.8]).^2)
        @test mse_value ≈ expected
        @test mse_value ≈ 0.038  atol=0.001
    end
    
    @testset "mse - Vector perfect prediction" begin
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [1.0, 2.0, 3.0, 4.0]
        
        @test mse(actual, predicted) == 0.0
    end
    
    @testset "mse - Vector length mismatch" begin
        actual = [1.0, 2.0, 3.0]
        predicted = [1.0, 2.0]
        
        @test_throws ArgumentError mse(actual, predicted)
    end
    
    @testset "mse - TimeSeries with matching timestamps" begin
        actual_ts = TimeSeries([1, 2, 3, 4, 5], [1.0, 2.0, 3.0, 4.0, 5.0])
        predicted_ts = TimeSeries([1, 2, 3, 4, 5], [1.1, 2.2, 2.9, 4.3, 4.8])
        
        mse_value = mse(actual_ts, predicted_ts)
        @test mse_value ≈ 0.038  atol=0.001
    end
    
    @testset "mse - TimeSeries with partial overlap" begin
        actual_ts = TimeSeries([1, 2, 3, 4, 5], [1.0, 2.0, 3.0, 4.0, 5.0])
        predicted_ts = TimeSeries([3, 4, 5, 6], [3.1, 4.2, 5.3, 6.0])
        
        # Only timestamps [3, 4, 5] match
        mse_value = mse(actual_ts, predicted_ts)
        expected_actual = [3.0, 4.0, 5.0]
        expected_pred = [3.1, 4.2, 5.3]
        expected_mse = mean((expected_actual .- expected_pred).^2)
        @test mse_value ≈ expected_mse
    end
    
    @testset "mse - TimeSeries no overlap" begin
        actual_ts = TimeSeries([1, 2, 3], [1.0, 2.0, 3.0])
        predicted_ts = TimeSeries([4, 5, 6], [4.0, 5.0, 6.0])
        
        @test_throws ArgumentError mse(actual_ts, predicted_ts)
    end
    
    @testset "mae - Vector" begin
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        predicted = [1.1, 2.2, 2.9, 4.3, 4.8]
        
        mae_value = mae(actual, predicted)
        expected = mean(abs.([1.0, 2.0, 3.0, 4.0, 5.0] .- [1.1, 2.2, 2.9, 4.3, 4.8]))
        @test mae_value ≈ expected
        @test mae_value ≈ 0.18  atol=0.01
    end
    
    @testset "mae - Vector perfect prediction" begin
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [1.0, 2.0, 3.0, 4.0]
        
        @test mae(actual, predicted) == 0.0
    end
    
    @testset "mae - Vector length mismatch" begin
        actual = [1.0, 2.0, 3.0]
        predicted = [1.0, 2.0]
        
        @test_throws ArgumentError mae(actual, predicted)
    end
    
    @testset "mae - TimeSeries with matching timestamps" begin
        actual_ts = TimeSeries([1, 2, 3, 4, 5], [1.0, 2.0, 3.0, 4.0, 5.0])
        predicted_ts = TimeSeries([1, 2, 3, 4, 5], [1.1, 2.2, 2.9, 4.3, 4.8])
        
        mae_value = mae(actual_ts, predicted_ts)
        @test mae_value ≈ 0.18  atol=0.01
    end
    
    @testset "mae - TimeSeries with partial overlap" begin
        actual_ts = TimeSeries([1, 2, 3, 4, 5], [1.0, 2.0, 3.0, 4.0, 5.0])
        predicted_ts = TimeSeries([3, 4, 5, 6], [3.1, 4.2, 5.3, 6.0])
        
        # Only timestamps [3, 4, 5] match
        mae_value = mae(actual_ts, predicted_ts)
        expected_actual = [3.0, 4.0, 5.0]
        expected_pred = [3.1, 4.2, 5.3]
        expected_mae = mean(abs.(expected_actual .- expected_pred))
        @test mae_value ≈ expected_mae
    end
    
    @testset "mae - TimeSeries no overlap" begin
        actual_ts = TimeSeries([1, 2, 3], [1.0, 2.0, 3.0])
        predicted_ts = TimeSeries([4, 5, 6], [4.0, 5.0, 6.0])
        
        @test_throws ArgumentError mae(actual_ts, predicted_ts)
    end
    
    @testset "rmse - Vector" begin
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        predicted = [1.1, 2.2, 2.9, 4.3, 4.8]
        
        rmse_value = rmse(actual, predicted)
        expected = sqrt(mse(actual, predicted))
        @test rmse_value ≈ expected
        @test rmse_value ≈ 0.195  atol=0.001
    end
    
    @testset "rmse - Vector perfect prediction" begin
        actual = [1.0, 2.0, 3.0, 4.0]
        predicted = [1.0, 2.0, 3.0, 4.0]
        
        @test rmse(actual, predicted) == 0.0
    end
    
    @testset "rmse - TimeSeries with matching timestamps" begin
        actual_ts = TimeSeries([1, 2, 3, 4, 5], [1.0, 2.0, 3.0, 4.0, 5.0])
        predicted_ts = TimeSeries([1, 2, 3, 4, 5], [1.1, 2.2, 2.9, 4.3, 4.8])
        
        rmse_value = rmse(actual_ts, predicted_ts)
        @test rmse_value ≈ 0.195  atol=0.001
    end
    
    @testset "rmse - TimeSeries with partial overlap" begin
        actual_ts = TimeSeries([1, 2, 3, 4, 5], [1.0, 2.0, 3.0, 4.0, 5.0])
        predicted_ts = TimeSeries([3, 4, 5, 6], [3.1, 4.2, 5.3, 6.0])
        
        # Only timestamps [3, 4, 5] match
        rmse_value = rmse(actual_ts, predicted_ts)
        expected_mse = mse(actual_ts, predicted_ts)
        expected_rmse = sqrt(expected_mse)
        @test rmse_value ≈ expected_rmse
    end
    
    @testset "rmse - relationship to mse" begin
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        predicted = [1.5, 2.5, 3.5, 4.5, 5.5]
        
        mse_val = mse(actual, predicted)
        rmse_val = rmse(actual, predicted)
        
        @test rmse_val^2 ≈ mse_val
        @test rmse_val ≈ sqrt(mse_val)
    end
    
    @testset "Metrics comparison" begin
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        predicted = [2.0, 3.0, 4.0, 5.0, 6.0]  # Constant offset of 1
        
        mae_val = mae(actual, predicted)
        mse_val = mse(actual, predicted)
        rmse_val = rmse(actual, predicted)
        
        # For constant error, MAE should equal the constant offset
        @test mae_val ≈ 1.0
        
        # MSE should be error squared
        @test mse_val ≈ 1.0
        
        # RMSE should equal MAE when errors are constant
        @test rmse_val ≈ 1.0
        @test rmse_val ≈ mae_val
    end
    
    @testset "Metrics with unordered timestamps" begin
        # Test that metrics work even when timestamps are not in order
        actual_ts = TimeSeries([5, 2, 8, 1], [50.0, 20.0, 80.0, 10.0])
        predicted_ts = TimeSeries([1, 5, 2, 8], [11.0, 52.0, 21.0, 79.0])
        
        # Should match all timestamps correctly
        mae_val = mae(actual_ts, predicted_ts)
        @test mae_val ≈ mean([1.0, 2.0, 1.0, 1.0])  # Errors: |10-11|, |50-52|, |20-21|, |80-79|
    end
end
