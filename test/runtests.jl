using TimeSeriesKit
using Aqua
using Test

@testset "TimeSeriesKit.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(TimeSeriesKit)
    end

    @testset "Operation XY" begin
        @test TimeSeriesKit.operation_xy(2, 3) == 5
        @test TimeSeriesKit.operation_xy(17, 4) == 21
    end
end
