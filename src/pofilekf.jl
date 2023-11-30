using Pkg
metal_dir = dirname(@__DIR__)
Pkg.activate(; temp=true)
Pkg.add(["Metal", "BenchmarkTools"])
Pkg.develop(path=metal_dir)

println("Package Status:")
Pkg.status()
println()

ENV["METAL_CAPTURE_ENABLED"] = 1

using Metal, BenchmarkTools

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

len = 1024*1024*100
n_threads = 1024
n_groups = cld(len, n_threads)

Metal.@profile @metal threads=n_threads groups=n_groups matrix_multiplication_kernel(A, B, C)

synchronize()
finalize(A)
finalize(B)
finalize(C)

