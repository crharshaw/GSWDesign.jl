using Test
using LinearAlgebra
using Random
using GSWDesign

# set this to see print messages when running tests
verbose=true

# run unit tests
include("unit-tests.jl")

# load in macro tests (small # units)
include("macro-tests.jl")
include("run-macro-tests.jl")