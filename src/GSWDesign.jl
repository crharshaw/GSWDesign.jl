module GSWDesign

using LinearAlgebra 
using Random
using Combinatorics

include("gsw-design.jl")
export sample_gs_walk

include("design-utils.jl")
export convert_to_pm1, empirical_assignment_mean_cov, build_stacked_matrix

include("enumerate-gsw-distribution.jl")
export gs_walk_entire_dist, exact_mean_cov

end # module
