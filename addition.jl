using CUDA
using KernelAbstractions

function broadcast_add!(a, b, c)
    @. a = b + c
    return nothing
end

@kernel function manual_add_kernel!(a, b, c)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = b[i, j, k] + c[i, j, k]
end

function manual_add!(a, b, c)
    kernel! = manual_add_kernel!(CUDADevice(), 256)
    kernel!(a, b, c, ndrange=size(a))
end

function run_broadcast(n)

    a = CUDA.zeros(n, n, n)
    b = CUDA.zeros(n, n, n)
    c = CUDA.zeros(n, n, n)

    add!(a, b, c)

    return nothing
end

function run_manual(n)

    a = CUDA.zeros(n, n, n)
    b = CUDA.zeros(n, n, n)
    c = CUDA.zeros(n, n, n)

    manual_add!(a, b, c)

    return nothing
end



n = 256
run_broadcast(n)
run_manual(n)
