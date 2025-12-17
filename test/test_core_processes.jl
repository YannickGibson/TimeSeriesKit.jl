using Test
using TimeSeriesKit
using Statistics
using Random

@testset "Core Processes" begin
    
    # Set seed for reproducibility
    Random.seed!(42)
    
    @testset "WhiteNoise - basic generation" begin
        wn = WhiteNoise(1000)
        @test wn isa TimeSeries
        @test length(wn) == 1000
        @test wn.name == "White Noise"
        
        # Check statistical properties (with tolerance)
        @test abs(mean(wn.values)) < 0.1  # Should be close to 0
        @test abs(std(wn.values) - 1.0) < 0.1  # Should be close to 1
    end
    
    @testset "WhiteNoise - custom parameters" begin
        wn = WhiteNoise(5000, mean=5.0, variance=4.0, name="Custom WN")
        @test length(wn) == 5000
        @test wn.name == "Custom WN"
        
        # Check statistical properties
        @test abs(mean(wn.values) - 5.0) < 0.1  # Should be close to 5
        @test abs(var(wn.values) - 4.0) < 0.2  # Should be close to 4
    end
    
    @testset "WhiteNoise - error cases" begin
        @test_throws ArgumentError WhiteNoise(0)
        @test_throws ArgumentError WhiteNoise(-1)
        @test_throws ArgumentError WhiteNoise(100, variance=-1.0)
        @test_throws ArgumentError WhiteNoise(100, variance=0.0)
    end
    
    @testset "ARProcess - AR(1)" begin
        Random.seed!(42)
        ar1 = ARProcess(1000, phi=0.8, constant=10.0)
        
        @test ar1 isa TimeSeries
        @test length(ar1) == 1000
        @test ar1.name == "AR(1) Process"
        
        # AR(1) mean should be approximately constant/(1-phi) = 10/(1-0.8) = 50
        @test abs(mean(ar1.values) - 50.0) < 5.0
    end
    
    @testset "ARProcess - AR(2)" begin
        Random.seed!(42)
        ar2 = ARProcess(1000, phi=[0.5, 0.3], constant=0.0, name="Custom AR2")
        
        @test length(ar2) == 1000
        @test ar2.name == "Custom AR2"
        
        # With constant=0 and stable coefficients, mean should be near 0
        @test abs(mean(ar2.values)) < 2.0
    end
    
    @testset "ARProcess - AR(3)" begin
        Random.seed!(42)
        ar3 = ARProcess(500, phi=[0.6, -0.2, 0.1])
        
        @test length(ar3) == 500
        @test ar3.name == "AR(3) Process"
    end
    
    @testset "ARProcess - scalar phi" begin
        ar = ARProcess(100, phi=0.5)
        @test ar.name == "AR(1) Process"
    end
    
    @testset "MAProcess - MA(1)" begin
        Random.seed!(42)
        ma1 = MAProcess(1000, theta=0.8, mean=5.0)
        
        @test ma1 isa TimeSeries
        @test length(ma1) == 1000
        @test ma1.name == "MA(1) Process"
        
        # MA mean should be approximately equal to the specified mean
        @test abs(mean(ma1.values) - 5.0) < 1.0
    end
    
    @testset "MAProcess - MA(2)" begin
        Random.seed!(42)
        ma2 = MAProcess(1000, theta=[0.9, -0.4], mean=0.0, name="Custom MA2")
        
        @test length(ma2) == 1000
        @test ma2.name == "Custom MA2"
        @test abs(mean(ma2.values)) < 1.0
    end
    
    @testset "MAProcess - MA(3)" begin
        Random.seed!(42)
        ma3 = MAProcess(500, theta=[0.6, -0.3, 0.2])
        
        @test length(ma3) == 500
        @test ma3.name == "MA(3) Process"
    end
    
    @testset "MAProcess - scalar theta" begin
        ma = MAProcess(100, theta=0.5)
        @test ma.name == "MA(1) Process"
    end
    
    @testset "ARMAProcess - ARMA(1,1)" begin
        Random.seed!(42)
        arma11 = ARMAProcess(1000, phi=0.7, theta=0.5, constant=0.0)
        
        @test arma11 isa TimeSeries
        @test length(arma11) == 1000
        @test arma11.name == "ARMA(1,1) Process"
    end
    
    @testset "ARMAProcess - ARMA(2,1)" begin
        Random.seed!(42)
        arma21 = ARMAProcess(1000, phi=[0.5, 0.3], theta=0.8, name="Custom ARMA")
        
        @test length(arma21) == 1000
        @test arma21.name == "Custom ARMA"
    end
    
    @testset "ARMAProcess - ARMA(1,2)" begin
        Random.seed!(42)
        arma12 = ARMAProcess(500, phi=0.6, theta=[0.4, -0.2])
        
        @test length(arma12) == 500
        @test arma12.name == "ARMA(1,2) Process"
    end
    
    @testset "ARMAProcess - ARMA(2,2)" begin
        Random.seed!(42)
        arma22 = ARMAProcess(500, phi=[0.7, -0.2], theta=[0.5, 0.3])
        
        @test length(arma22) == 500
        @test arma22.name == "ARMA(2,2) Process"
    end
    
    @testset "ARMAProcess - error cases" begin
        @test_throws ArgumentError ARMAProcess(0, phi=0.5, theta=0.5)
        @test_throws ArgumentError ARMAProcess(-1, phi=0.5, theta=0.5)
        @test_throws ArgumentError ARMAProcess(100, phi=0.5, theta=0.5, noise_variance=-1.0)
        @test_throws ArgumentError ARMAProcess(100, phi=0.5, theta=0.5, noise_variance=0.0)
    end
    
    @testset "ARMAProcess - reduces to AR when theta=0" begin
        Random.seed!(42)
        ar_direct = ARProcess(1000, phi=0.8, constant=5.0)
        
        Random.seed!(42)
        ar_via_arma = ARMAProcess(1000, phi=0.8, theta=0.0, constant=5.0)
        
        @test ar_direct.values ≈ ar_via_arma.values
    end
    
    @testset "ARMAProcess - reduces to MA when phi=0" begin
        Random.seed!(42)
        ma_direct = MAProcess(1000, theta=0.8, mean=5.0)
        
        Random.seed!(42)
        ma_via_arma = ARMAProcess(1000, phi=0.0, theta=0.8, constant=5.0)
        
        @test ma_direct.values ≈ ma_via_arma.values
    end
    
    @testset "RandomWalk - basic generation" begin
        Random.seed!(42)
        rw = RandomWalk(1000)
        
        @test rw isa TimeSeries
        @test length(rw) == 1000
        @test rw.name == "Random Walk"
        
        # Random walk should generally drift away from 0
        @test abs(rw.values[end]) > 1.0
    end
    
    @testset "RandomWalk - custom parameters" begin
        Random.seed!(42)
        rw = RandomWalk(5000, mean=0.5, variance=2.0, name="Custom RW")
        
        @test length(rw) == 5000
        @test rw.name == "Custom RW"
        
        # With positive mean, should trend upward
        @test rw.values[end] > rw.values[1]
    end
    
    @testset "RandomWalk - zero mean drifts" begin
        Random.seed!(42)
        rw = RandomWalk(10000, mean=0.0, variance=1.0)
        
        # Check that it's actually a cumulative sum of increments
        # by verifying differences are independent
        diffs = diff(rw.values)
        @test length(diffs) == 9999
        @test abs(mean(diffs)) < 0.1  # Mean of increments should be near 0
    end
    
    @testset "RandomWalk - error cases" begin
        @test_throws ArgumentError RandomWalk(0)
        @test_throws ArgumentError RandomWalk(-1)
        @test_throws ArgumentError RandomWalk(100, variance=-1.0)
        @test_throws ArgumentError RandomWalk(100, variance=0.0)
    end
    
    @testset "Process consistency - reproducibility" begin
        # Test that same seed produces same results
        Random.seed!(123)
        wn1 = WhiteNoise(100)
        
        Random.seed!(123)
        wn2 = WhiteNoise(100)
        
        @test wn1.values == wn2.values
    end
end
