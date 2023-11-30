using Pkg
metal_dir = dirname(@__DIR__)
Pkg.activate(; temp=true)
Pkg.add(["Metal", "BenchmarkTools", "LinearAlgebra"])
Pkg.develop(path=metal_dir)

println("Package Status:")
Pkg.status()
println()

ENV["METAL_CAPTURE_ENABLED"] = 1

using Metal, BenchmarkTools, LinearAlgebra

function matrix_multiplication_kernel(A, B, C)
    idx = thread_position_in_grid_1d()
    N = 10
    for _ in 1:N
        C[idx] += sin(A[idx]) * exp(B[idx])
    end
    return
end


m, n, p = 1_000, 1_000, 1_000

A = Metal.rand(Float32, (m, n); storage=Shared)
B = Metal.rand(Float32, (n, p); storage=Shared)
C = Metal.zeros(Float32, (m, p); storage=Shared)

C_cpu = unsafe_wrap(Array{Float32}, C, size(C))

len = 1024*1024*100
n_threads = 1024
n_groups = cld(len, n_threads)

benchmark_results = @benchmark Metal.@sync begin
    @metal threads=$n_threads groups=$n_groups matrix_multiplication_kernel($A, $B, $C)
end

function generate_report(benchmark_results)
    println("Benchmark Details:")
    show(benchmark_results)
    println("\nAdditional Information:")
    println("  - Secs: ", minimum(benchmark_results.times))
    println("  - Memory estimate: ", benchmark_results.memory) 
end

generate_report(benchmark_results)
println(benchmark_results)
finalize(A)
finalize(B)
finalize(C)

# The resulting matrix can now be modified on the CPU
inv(C_cpu)