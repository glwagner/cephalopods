# cephalopods (Adventures in profiling)

Write

```
julia -L heptapus_example.jl
```

and then

```julia
julia> r
Roofline containing Table with 9 columns and 1 row:
     kernels                                                                                        arithmeticintensity  performance  kernelmaxempiricalbandwidth  maxgflopsestimate  maxempiricalbandwidth  haslocal  floptype  hasmixedflops
   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 1 │ julia_broadcast(CuKernelContext, CuDeviceArray<Float32, int=3, Global>, Broadcasted<void, Tu…  0.0316454            20.0196      2.80806e6                    12838.9            1.03244e7              true      Float32   false
 ```
