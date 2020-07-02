using Heptapus

#r = Roofline(`$(Base.julia_cmd()) addition.jl`)
r = Roofline(`$(Base.julia_cmd()) stencil.jl`)

