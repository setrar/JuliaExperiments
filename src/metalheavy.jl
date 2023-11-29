using Pkg
metal_dir = dirname(@__DIR__)
Pkg.activate(; temp=true)
Pkg.add(["Metal", "BenchmarkTools"])
Pkg.develop(path=metal_dir)

using Metal, BenchmarkTools

function benchmark_matrix_multiplication(n_threads, n_groups)
    # size of each square matrix
    m, n, p = 80_000, 80_000, 80_000

    # If A is an m × n matrix and B is an n × p matrix,
    # https://en.wikipedia.org/wiki/Matrix_multiplication    

    # Create Metal arrays for matrices A, B, and C with shared storage
    A = Metal.rand(Float32, (m, n); storage=Shared)
    B = Metal.rand(Float32, (n, p); storage=Shared)
    C = Metal.zeros(Float32, (m, p); storage=Shared)

    # Benchmark matrix multiplication
    bench_result = @benchmark Metal.@sync begin
        # @metal = macro, used to indicate that the block of code contains GPU kernels taht will be excuted on the Metal GPU
        # threads = specifies the number of GPU threads to be launched for the kernel.
        # groups = the number of thread groups to be launched for the kernel. A thread group is a collection of threads that can cooperate and synchronize
        # $ = used to interpolate the values of the CPU arrays into the GPU kernel. Important because GPU kernels operate on GPU arrays.

        # In general, the total number of threads (threads multiplied by groups)
        # should be sufficient to cover the size of the data you are processing. 
        # If you have a 2D array of size MxN, you might choose threads and groups such that threads * groups is at least M * N to ensure that each element of the array is processed.

        @metal threads=$n_threads groups=$n_groups kernel_multiply($A, $B, $C)
    end

    # Cleanup memory
    finalize(A)
    finalize(B)
    finalize(C)

    return bench_result
end

function kernel_multiply(A, B, C)
    idx = thread_position_in_grid_1d()    
    # Perform matrix multiplication
    # create additional complexity here but multiplying
    # the matrix n times
    #  m, n, p = size(Aᵐ, 1), size(Bᵐ, 2), size(Cᵐ, 2)
    
     # Perform matrix multiplication
    for _ in 1:100000
        C[idx] += A[idx] * B[idx]
    end
    return
end


function generate_report(bench_result)
    println("Benchmark Details:")
    show(bench_result)

    # Additional information
    println("\nAdditional Information:")
    println("  - Number of samples: ", length(bench_result.times))
    println("  - Memory estimate: ", bench_result.memory)
end

# Run the benchmark and generate the report

# Choose the number of threads and groups based on the total number of elements
n_threads, n_groups = 1024, 1024
result = benchmark_matrix_multiplication(n_threads, n_groups)
generate_report(result)


# 64

# 128
# Benchmark Details:
# Trial(8.732 ms)
# Additional Information:
#   - Number of samples: 561
#   - Memory estimate: 5128

# 256
# Benchmark Details:
# Trial(27.637 ms)
# Additional Information:
#   - Number of samples: 171
#   - Memory estimate: 5128

# 512
# Trial(102.329 ms)
# Additional Information:
#   - Number of samples: 47
#   - Memory estimate: 5192

# 1024