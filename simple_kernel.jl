using CUDA

function add!(a, b, c)
    @. a = b + c
    return nothing
end

function run(n)

    a = CUDA.zeros(n, n, n)
    b = CUDA.zeros(n, n, n)
    c = CUDA.zeros(n, n, n)

    add!(a, b, c)

    return nothing
end

n = 256
run(n)
