# gsw-design.jl
# Chris Harshaw, Fredrik Savje, Dan Spielman, Peng Zhang 
# January 2020
#
# An efficient implementation for sampling assignments from the Gram--Schmidt Walk Design.
#

using LinearAlgebra

"""
    sample_gs_walk(X::Array{AbstractFloat,2}, lambda::AbstractFloat ...)

Sample assignment vectors from the Gram--Schmidt Walk Design. 

A fast implementation which maintains a cholesky factorization of (I + X * X^T ) for faster repeated linear 
system solves and has a recursive component for more effective memory allocation.

# Arguments
- `X`: an n by d matrix which has the covariate vectors x_1, x_2 ... x_n as rows
- `lambda`: a design parameter in (0,1] which determines the level of covariate balance
- `balanced`: set `true` to sample from balanced Gram--Schmidt Walk design. (default: `false`)
- `treatment_probs`: a `Number` is interpreted as the marginal treatment probability for each unit. An `Array` is interpreted as array of marginal treatment probabilities.
- `num_samples`: the number of sample assignments to draw 

# Output 
- `assignment_list`: sampled +/- 1 assignment vectors. Integer array of dimension (n) if `num_samples==1`, otherwise dimensions are (`num_samples`,n)
"""
function sample_gs_walk(
    X::Array{<:AbstractFloat,2}, 
    lambda::AbstractFloat; 
    balanced=false, 
    treatment_probs = 0.5, 
    num_samples=1
)

    # transpose the covariate matrix so it has covariates as columns (this is a quick fix)
    X = copy(X')

    # get the dimensions, re-scale so covariate norm is equal to 1
    d, n = size(X)
    max_norm = maximum([norm(X[:,i]) for i=1:n])
    if max_norm > eps()
        X ./= max_norm
    end
    
    # transform treatment prob to means vector
    if isa(treatment_probs, Number)
        @assert( 0 < treatment_probs < 1.0)
        z0 = (2.0 * treatment_probs) * ones(n) .- 1.0
    else
        @assert(all(0 .<= treatment_probs .<= 1.0))
        @assert(length(treatment_probs) == n)
        z0 = (2.0 * treatment_probs) .- 1.0
    end

    # pre-processing: compute cholesky factorization of I + (1-a) X X^T
    M = (lambda / (1-lambda)) * I +  (X * X')
    MC = cholesky(M)

    # compute sum of covariances if necessary 
    if balanced 
        cov_sum = sum(X, dims=2)
    else
        cov_sum = nothing
    end

    # sample the num_sample assignments
    assignment_list = zeros(Int64, num_samples, n)
    for i=1:num_samples

        # run the recursive version of the walk 
        z = _gs_walk_recur(X, copy(MC), copy(z0), lambda, balanced, cov_sum)

        # update assignment list
        for j=1:n
            assignment_list[i,j] = (z[j] < 0) ? -1 : 1
        end
    end

    # return a 1-dimensional array if only one sample
    if num_samples == 1
        assignment_list = reshape(assignment_list, n)
    end

    return assignment_list
end

"""
    _gs_walk_recur

Run the iterative procedure of the Gram--Schmidt Walk.

This function recusively cals itself after sufficiently many variables have been frozen to achieve
better memory allocation. The cholesky factorization of (I + X * X^T ) is also maintained.

# Arguments
- `X`: an d by n matrix which has the alive covariate vectors x_1, x_2 ... x_n as columns
- `MC`: the relevant cholesky factorization 
- `z`: vector of fractional assignments 
- `lambda`: the design parameter, a real value in (0,1)
- `balanced`: (optional) bool, set `true` to run the balanced GSW. Default value is `false`
- `cov_sum`: (optional) length d vector, sum of the current alive covariates 

# Output
- `z`: the random +/- 1 vector, length n array of Float
"""
function _gs_walk_recur(
    X::Array{<:AbstractFloat,2}, 
    MC::Cholesky,
    z::Array{<:AbstractFloat,1}, 
    lambda::AbstractFloat, 
    balanced::Bool, 
    cov_sum)

    # get the dimensions, set tolerance
    d, n = size(X)
    tol = 100*eps()

    # initialize alive variablesa and pivot index
    live_not_pivot = trues(n) # bit array, space efficient
    p_alive = true

    # select pivot, update cholesky and covariate sum
    p = rand(1:n) 
    live_not_pivot[p] = false
    lowrankdowndate!(MC, X[:,p])
    if balanced 
        cov_sum -= X[:,p]
    end

    # iterate through the GS walk
    iter = 1

    # will recurse if freeze a large number of variables - need to optimize those parameters.
    num_frozen = 0
    targ_frozen = max(5, div(n,3))

    while any(live_not_pivot .!= false) || p_alive # while any alive variables

        # if pivot was previously frozen
        if !p_alive

            if num_frozen >= targ_frozen
                # println("recur: $(n), $(num_frozen), $(targ_frozen)")
                y = _gs_walk_recur(X[:,live_not_pivot], MC, z[live_not_pivot], lambda, balanced, cov_sum)
                z[live_not_pivot] = y
                break
            end

            # select a new pivot by pivot rule
            p = sample_pivot(live_not_pivot, n - num_frozen)
            p_alive = true
            live_not_pivot[p] = false

            # downdate cholesky factorization by a_ratio * (a_p a_p') now that p has been decided
            lowrankdowndate!(MC, X[:,p])
            if balanced 
                cov_sum -= X[:,p]
            end
        end

        # get the u vector (only defined on live no pivots) 
        u = compute_step_direction(MC, X, lambda, p, live_not_pivot, balanced, cov_sum)

        # get the step size delta
        del_plus, del_minus = compute_step_sizes(z, u, live_not_pivot, p)
        prob_plus = del_minus / (del_plus + del_minus)
        del = (rand() < prob_plus) ? del_plus : -del_minus # randomly choose + or -

        # update z
        z[live_not_pivot] += del * u
        z[p] += del

        # update indices if they are frozen 
        for i=1:n
            if live_not_pivot[i]

                # if frozen, update live not pivot array, cholesky factorization, and covariate sum
                if (abs(z[i]) >= 1. - tol)
                    live_not_pivot[i] = false
                    lowrankdowndate!(MC, X[:,i])
                    if balanced 
                        cov_sum -= X[:,i]
                    end
                    num_frozen += 1
                end

            elseif p == i
                # a flag for whether pivot is alive
                p_alive = (abs(z[i]) < 1. - tol)
                if !p_alive 
                    num_frozen += 1
                end
            end
        end

        # update iteration count
        iter += 1
    end

    return z
end

"""
    sample_pivot(live_not_pivot::BitArray{1}, num_alive::Integer)

Uniformly sample a pivot from the set of alive variables.

Note that this is only meant to be called when a pivot is frozen, so live_not_pivot == live

# Arguments
- `live_not_pivot`:  n length BitArray where a `true` entry means the variable is alive and not pivot
- `num_alive`: the number of alive variables, Integer

# Output 
- `p`: the randomly chosen pivot
"""
function sample_pivot(live_not_pivot::BitArray{1}, num_alive::Integer)

    ind = rand(1:num_alive)
    for p=1:length(live_not_pivot)
        if live_not_pivot[p]
           ind -= 1
            if ind == 0
                return p
            end
        end
    end
end

"""
    compute_step_direction

Efficiently compute step direction `u` for live not pivot variables using matrix factorizations.

# Arguments
- `MC`: the relevant cholesky factorization
- `X`: an d by n matrix which has the covariate vectors x_1, x_2 ... x_n as columns
- `lambda`: the design parameter, a real value in (0,1)
- `p`: pivot variable, Integer
- `balanced`: Bool, set `true` to run the balanced GSW and `false` for typical GSW
- `cov_sum`: if `balanced = true` then `cov_sum` must be length d vector, sum of the current alive covariates 

# Output 
- `u`           the step direction only defined on live not pivot variables i.e. length(u) == sum(live_not_pivot)
"""
function compute_step_direction(
    MC::Cholesky, 
    X::Array{<:AbstractFloat,2}, 
    lambda::AbstractFloat, 
    p::Integer, 
    live_not_pivot::BitArray{1}, 
    balanced::Bool, 
    cov_sum
)

    # Here is a description of the a, more clearly outlined in paper
    #   a(0) = X_k X_k' * z_p                                       O(d^2) using factorization
    #   a(1) = inv( lambda/(1-lambda) * I + X_k' * X_k ) * a(0)     O(d^2) using factorization 
    #   a(2) = (1-lambda)/(lambda) [ a(1) - v_p]                    O(d)
    #   a(3) = X_k * a(2)                                           O(nd) matrix-vector multiplication 

    a = (MC.L * (MC.U * X[:,p])) - (lambda / (1-lambda)) * X[:,p]   # a(0)
    ldiv!(MC, a)                                                    # a(1)

    mult_val = (1 - lambda) / lambda                                # a(2), in place
    for i=1:length(a)
        a[i] = mult_val * (a[i] - X[i,p]) 
    end
    a = (X' * a)[live_not_pivot]                                    # a(3)

    if balanced 

        # Here is a description of the b, more clearly outline in the paper 
        #   b(0) = X_k' * 1                                             O(1) look-up (pre-computed)
        #   b(1) = inv( lambda/(1-lambda) * I + X_k' * X_k ) * b(0)     O(d^2) using factorization 
        #   b(2) = X_k * b(1)                                           O(nd) matrix-vector multiplication 
        #   b(3) = (b(2) - 1) / (2 * lambda)                            O(n)

        b = copy(cov_sum)               # b(0)
        ldiv!(MC, b)                    # b(1) 
        b = (X' * b)[live_not_pivot]    # b(2) 
        
        div_val = 2*lambda              # b(3), in place 
        for i=1:length(b)
            b[i] = (b[i] - 1) / div_val
        end

        # compute scaling constant 
        scale = - (1 + sum(a)) / (sum(b))

        # compute u 
        u = a + (scale * b) 
    else 
        # u is a -- nothing to do, really
        u = a
    end

    return u
end


"""
    compute_step_sizes

Compute the positive and negative step sizes, without unecessary allocations & calculations.

# Arguments
- `z`: n vector in [-1,1]
- `u`: m vector where m = # of non-pivot alive variables
- `live_not_pivot`: n length BitArray where a `true` entry means the variable is alive and not pivot
- `p`: pivot variable, Integer

# Output
- `del_plus`    the positive step size
- `del_minus`   the negative step size
"""
function compute_step_sizes(z::Array{<:AbstractFloat,1}, u::Array{<:AbstractFloat,1}, live_not_pivot::BitArray{1}, p::Integer)

    # initialize + and - step sizes
    del_plus = Inf
    del_minus = Inf

    # set tolerance 
    zero_tol = 10*eps()

    # go through all coordinates, finding best
    ind = 0
    for i=1:length(z)

        if live_not_pivot[i]

            ind += 1

            # skip the case where u is numerically zero 
            if abs(u[ind]) <= zero_tol
                continue
            end 

            # these are the step sizes delta_+ and delta_- that yield integrality
            dp = (sign(u[ind]) - z[i]) / u[ind]
            dm = (sign(u[ind]) + z[i]) / u[ind]

            # update step sizes to z[i] is always within +/- 1
            del_plus = (dp < del_plus) ? dp : del_plus
            del_minus = (dm < del_minus) ? dm : del_minus

        elseif p == i

                # these are step sizes delta_+ and delta_- that yield integrality
                dp = 1 - z[i]
                dm = 1 + z[i]

                # update step sizes to z[i] is always within +/- 1
                del_plus = (dp < del_plus) ? dp : del_plus
                del_minus = (dm < del_minus) ? dm : del_minus
        end
    end

    # return largest possible +/- step sizes
    return del_plus, del_minus
end