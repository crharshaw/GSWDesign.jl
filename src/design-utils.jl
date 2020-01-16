# design-utils.jl
# Chris Harshaw, Fredrik Savje, Dan Spielman, Peng Zhang 
# January 2020
#
# Utilities for general designs, including
#   1. format transforming 
#   2. computing empirical mean and covariances
#   3. brute force enumeration of Gram--Schmidt Walk Design distribution
#
# This code is meant to be internal to the package. Use at your own risk.

using LinearAlgebra

function convert_to_pm1(assign_list)

    # convert to float (for computation later)
    assign_list = convert(Array{Float64}, assign_list)

    # change from 0/1 to +/- 1
    num_samples, n = size(assign_list)
    for i=1:num_samples
        for j=1:n
            assign_list[i,j] = (assign_list[i,j] > 0.5) ? 1.0 : -1.0
        end
    end
    return assign_list
end

"""
    empirical_assignment_mean_cov(X, lambda, num_samples...)

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
function empirical_assignment_mean_cov(X, lambda, num_samples; balanced=false, treatment_probs=0.5*ones(size(X,1)))

    # sample all assignments, convert to +/- 1 vectors 
    n,d = size(X)
    assign_list = sample_gs_walk(X, lambda, treatment_probs=treatment_probs, balanced=balanced, num_samples=num_samples)
    assign_list = convert_to_pm1(assign_list)

    # compute empirical mean & covariance
    empirical_mean = sum(assign_list, dims=1) / num_samples         # mean 
    assign_list .-= empirical_mean                                  # centering assignments in place
    empirical_cov = (assign_list' * assign_list) / num_samples      # covariance

    return reshape(empirical_mean,n), empirical_cov
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
function build_stacked_matrix(X, lambda)
    n,d=size(X)
    max_cov_norm = maximum([norm(X[i,:]) for i=1:n])
    return vcat(sqrt(lambda)*I, (sqrt(1-lambda) / max_cov_norm )*X')
end