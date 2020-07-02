using CUDA
using KernelAbstractions
using OffsetArrays

@kernel function i_stencil_kernel!(a, b, c)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = (b[i, j, k] + b[i+1, j, k]) * c[i, j, k]
end

@kernel function j_stencil_kernel!(a, b, c)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = (b[i, j, k] + b[i, j+1, k]) * c[i, j, k]
end

@kernel function k_stencil_kernel!(a, b, c)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] = (b[i, j, k] + b[i, j, k+1]) * c[i, j, k]
end

function i_stencil!(a, b, c)
    kernel! = i_stencil_kernel!(CUDADevice(), 16)
    nx, ny, nz = size(a)
    kernel!(a, b, c, ndrange=(nx-2, ny-2, nz-2))
end

function j_stencil!(a, b, c)
    kernel! = j_stencil_kernel!(CUDADevice(), 16)
    nx, ny, nz = size(a)
    kernel!(a, b, c, ndrange=(nx-2, ny-2, nz-2))
end

function k_stencil!(a, b, c)
    kernel! = k_stencil_kernel!(CUDADevice(), 16)
    nx, ny, nz = size(a)
    kernel!(a, b, c, ndrange=(nx-2, ny-2, nz-2))
end

function run(n)

    #a = OffsetArray(CUDA.zeros(n, n, n), 0:n-1, 0:n-1, 0:n-1)
    #b = OffsetArray(CUDA.zeros(n, n, n), 0:n-1, 0:n-1, 0:n-1)
    #c = OffsetArray(CUDA.zeros(n, n, n), 0:n-1, 0:n-1, 0:n-1)

    a = CUDA.zeros(n, n, n)
    b = CUDA.zeros(n, n, n)
    c = CUDA.zeros(n, n, n)

    i_stencil!(a, b, c)
    j_stencil!(a, b, c)
    k_stencil!(a, b, c)

    return nothing
end

n = 256
run(n)
