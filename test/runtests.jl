using Test
using LinearAlgebra
using GSWDesign

# run unit tests
include("unit-tests.jl")

# load in macro tests (small # units)
include("macro-tests.jl")
include("run-macro-tests.jl")