# design-utils.jl
# Chris Harshaw, Fredrik Savje, Dan Spielman, Peng Zhang 
# January 2020
#
# Utilities for general designs, including
#   1. computing empirical mean and covariances
#   2. building stacked matrix
#
# This code is meant to be internal to the package. Use at your own risk.

using LinearAlgebra


"""
    topmr(z::BitArray) -> Array{Float64,N}

Transform assignments `z` from `{0, 1}` to `{-1.0, 1.0}`.

# Examples
```julia-repl
z = BitArray([1, 1, 0, 1, 0, 1])
topmr(z)
6-element Array{Float64,1}:
  1.0
  1.0
 -1.0
  1.0
 -1.0
  1.0
z = BitArray([1 1 0 1 0 1; 0 1 1 0 0 1])
topmr(z)
2×6 Array{Float64,2}:
  1.0  1.0  -1.0   1.0  -1.0  1.0
 -1.0  1.0   1.0  -1.0  -1.0  1.0
```
"""
topmr(z::BitArray) = 2.0 .* z .- 1.0


"""
    empirical_assignment_mean_cov(X::Array{<:AbstractFloat,2}, lambda::AbstractFloat, num_samples::Integer, ...)

Compute empirical mean and covariance matrix of +/- 1 assignment vectors

# Arguments
- `X`: an n by d matrix which has the covariate vectors x_1, x_2 ... x_n as rows
- `lambda`: a design parameter in (0,1] which determines the level of covariate balance
- `num_samples`: the number of assignment vectors to sample from the design 
- `treatment_probs`: n length vector of treatment probabilities in [0,1] (default: all entries = 1/2)
- `balanced`: set `true` to sample from balanced Gram--Schmidt Walk design. (default: `false`)

# Output 
- `empirical_mean`: n length vector of empirical means of +/- 1 assignment vectors 
- `empirical_cov`: n-by-n empirical covariance matrix of +/- 1 assignment vectors
"""
function empirical_assignment_mean_cov(
    X::Array{<:AbstractFloat,2}, 
    lambda::AbstractFloat, 
    num_samples::Integer; 
    balanced=false, 
    treatment_probs=0.5
)

    @assert(num_samples > 1) # otherwise, we'll have dimension mismatch
    n,d = size(X)

    # sample all assignments
    assignment_list = sample_gs_walk(X, lambda, num_samples, balanced=balanced, treatment_probs=treatment_probs)

    # compute empirical mean and covariance
    A = topmr(reduce(hcat, assignment_list)) 
    empirical_mean = reshape(sum(A, dims=2), n) ./ length(assignment_list)
    empirical_cov = (A*A' ./ length(assignment_list)) - (empirical_mean * empirical_mean')

    return empirical_mean, empirical_cov
end

"""
    build_stacked_matrix

Build stacked matrix used in Gram--Schmidt Walk Design. Includes automatic scaling.

# Arguments
- `X`: an n by d matrix which has the covariate vectors x_1, x_2 ... x_n as rows
- `lambda`: a design parameter in (0,1] which determines the level of covariate balance

# Output 
- `B`: (d+n)-by-n stacked matrix used in GSW design. 
"""
function build_stacked_matrix(X::Array{<:AbstractFloat,2}, lambda::AbstractFloat)
    n,d=size(X)
    max_cov_norm = maximum([norm(X[i,:]) for i=1:n])
    return vcat(sqrt(lambda)*I, (sqrt(1-lambda) / max_cov_norm )*X')
end