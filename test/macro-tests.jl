# macro-tests.jl
# Chris Harshaw, Fredrik Savje, Dan Spielman, Peng Zhang 
# January 2020
#
# These are macro test functions for the Gram--Schmidt Walk
#

using Random 

function matches_known_design_cov(rseed, n, balanced, num_samples, tol; verbose=false)
    """
    # matches_known_design_cov
    # Tests whether GSW run on orthogonal covariates yields the known Bernoulli design in
    # the unbalanced case and the complete randomization design in the unbalanced case.
    # More precisely, the test is whether the empirical covariance is close to the true known
    # covariance in the operator norm sense.
    #
    # Note that it is up to the user to provide num_samples and tol parameters which make sense
    # given n. A reasonable choice is to first pick tol then set num_samples = 2 * ( n / tol )^2 .
    # lambda is not an input parameter because it doesn't matter in these orthogonal instances.
    #
    # Input
    #   rseed       the random seed used for the test 
    #   n           the number of units
    #   balanced    set `true` to run the balanced GSW; otherwise, leave false 
    #   num_samples   number of samples used for empirical covariance matrix
    #   tol         highest passable value of || empirical_cov - exact_cov ||
    #   
    # Output 
    #   test_pass       true if || empirical_cov - exact_cov || < tol; false otherwise
    #   empirical_cov   the computed empirical covariance matrix
    #   exact_cov       the (known) exact covariance matrix
    """

    # set random seed
    Random.seed!(rseed)

    # construct orthogonal covariates
    X = Matrix{Float64}(I, n, n)

    # obtain empirical covariance of +/- 1 assignments 
    _, empirical_cov = empirical_assignment_mean_cov(X, 0.5, num_samples, balanced=balanced)

    # construct the true covariance matrix 
    if !balanced
        exact_cov = Matrix{Float64}(I, n, n)
    else
        exact_cov = (n*I - ones(n,n)) / (n-1)
    end

    # compute the norm error 
    norm_err = norm(empirical_cov - exact_cov)
    test_pass = norm_err < tol 

    # print statements if verbose 
    if verbose 
        println("Testing whether GSW assignment covariance matches known design")
        println("\tn = ", n)
        println("\tbalanced = ", balanced)
        println("\t# of samples = ", num_samples)
        println("\ttolerance = ", tol)
        println("\nNorm Error: ", norm_err)
        if test_pass
            println("Test passed")
        else
            println("Test failed")
        end
    end

    return test_pass, empirical_cov, exact_cov
end

function matches_enumerated_design_cov(rseed, X, lambda, balanced, num_samples, tol; verbose=false)
    """
    # matches_enumerated_design_cov
    # Tests whether our Gram--Schmidt Walk implementation has an empirical covariane matrix which
    # matches what is produced by the brute-force enumeration. Brute force enumeration runs quickly 
    # only for small n, say n <= 10. 
    # More precisely, the test is whether the empirical covariance matrix is close to the true brute
    # force calculated covariance matrix in the operator norm sense.
    #
    # Note that it is up to the user to provide num_samples and tol parameters which make sense
    # given n. A reasonable choice is to first pick tol then set num_samples = 2 * ( n / tol )^2 .
    #
    # Input
    #   rseed       the random seed used for the test
    #   X           the d-by-n matrix of covariates, where covariates are columns 
    #   lambda       the GSW trade-off parameter
    #   balanced    set `true` to run the balanced GSW; otherwise, leave false 
    #   num_samples   number of samples used for empirical covariance matrix
    #   tol         highest passable value of || empirical_cov - true_cov ||
    #   
    # Output 
    #   test_pass       true if || empirical_cov - true_cov || < tol; false otherwise
    #   empirical_cov   the computed empirical covariance matrix
    #   exact_cov       the (brute-force computed) exact covariance matrix
    """

    # set random seed
    Random.seed!(rseed)

    # get dimensions, check col norms at most 1
    n,d = size(X)
    # @assert(all([norm(X[i,:]) for i=1:n] .< 1.0))

    # obtain true covariance by brute force enumeration
    B = build_stacked_matrix(X, lambda)
    assign_list, prob_list = gs_walk_entire_dist(B, balanced=balanced)
    exact_mean, exact_cov = exact_mean_cov(assign_list, prob_list)

    # obtain empirical covariance of +/- 1 assignments 
    _, empirical_cov = empirical_assignment_mean_cov(X, lambda, num_samples, balanced=balanced)

    # compute error
    norm_err = norm(empirical_cov - exact_cov)
    test_pass = norm_err < tol

    # print statements if verbose 
    if verbose 
        println("Testing whether GSW assignment covariance matches brute-force computed covariance")
        println("\tn = ", n, "\td = ", d)
        println("\tbalanced = ", balanced)
        println("\t# of samples = ", num_samples)
        println("\ttolerance = ", tol)
        println("\nNorm Error: ", norm_err)
        if test_pass
            println("Test passed")
        else
            println("Test failed")
        end
    end

    return test_pass, empirical_cov, exact_cov
end

function correct_treatment_probabilities(rseed, treatment_probs, X, lambda, balanced, num_samples, tol; verbose=false)
    """
    # correct_treatment_probabilities
    # Tests whether our Gram--Schmidt Walk implementation has correct marginal assignment probabilities.
    # More precisely, the test is whether the empirical probabilities of assignment match the specified 
    # treatment probabilities, in the infinity norm sense.
    #
    # Note that it is up to the user to provide num_samples and tol parameters which make sense
    # given n. A reasonable choice is to first pick tol then set num_samples = 2 * n * (log(n) / tol)^2
    #
    # Input
    #   rseed           the random seed used for the test
    #   treatment_probs an n length vector in [0, 1]
    #   X               the n-by-d matrix of covariates, where covariates are columns 
    #   lambda           the GSW trade-off parameter
    #   balanced        set `true` to run the balanced GSW; otherwise, leave false 
    #   num_samples       number of samples used for empirical covariance matrix
    #   tol             highest passable value of || target_mean - empirical_mean ||_inf
    #   
    # Output 
    #   test_pass       true if || target_mean - empirical_mean ||_inf < tol; false otherwise
    #   empirical_mean  the computed empirical mean
    """

    # set random seed
    Random.seed!(rseed)

    # get dimensions
    n,d = size(X)

    # check that this test instance is legitimate
    # @assert(all([norm(X[i,:]) for i=1:n] .< 1.0))   # covariate vectors have norm at most 1
    @assert(all(0.0 .<= treatment_probs .<= 1.0))         # target mean has values in [-1, 1]
    @assert(length(treatment_probs) == n)               # dimensions match 

    # obtain empirical mean of +/- 1 assignments, then empirical probs
    empirical_mean, empirical_cov = empirical_assignment_mean_cov(X, lambda, num_samples, balanced=balanced, treatment_probs=treatment_probs)
    empirical_probs = 0.5 * (empirical_mean .+ 1.0)

    # compute error 
    mean_err = abs.(empirical_probs - treatment_probs)
    test_pass = all(mean_err .< tol)

    # print statements if verbose 
    if verbose 
        println("Testing whether GSW produces the right marginal assignment probabilities")
        println("\tn = ", n, "\td = ", d)
        println("\tbalanced = ", balanced)
        println("\t# of samples = ", num_samples)
        println("\ttolerance = ", tol)
        println("\nMaximum Error: ", maximum(mean_err))
        if test_pass
            println("Test passed")
        else
            println("Test failed")
        end
    end

    return test_pass, empirical_probs
end

function test_cov_bound(rseed, X, lambda, num_samples, tol; verbose=false)
    """
    # test_cov_bound
    # Tests whether our Gram--Schmidt Walk implementation yields a +/- 1 assignment covariance 
    # matrix which satisfies the bound that we proved. We obtain an empirical covariance and 
    # test that our upper bound holds approximately in the sense that the maximum eigenvalue of
    # the difference is not too large.
    #
    # Note that it is up to the user to provide num_samples and tol parameters which make sense
    # given n. A reasonable choice is to first pick tol then set num_samples = 2 * (n / tol)^2
    #
    # Input
    #   rseed           the random seed used for the test
    #   X               the n-by-d matrix of covariates, where covariates are columns 
    #   lambda           the GSW trade-off parameter
    #   num_samples       number of samples used for empirical covariance matrix
    #   tol             highest passable value of eigmax(cov_upper_bound)
    #   
    # Output 
    #   test_pass       true if eigmax(cov_upper_bound) < tol; false otherwise
    #   empirical_cov   the computed empirical covariance matrix
    #   cov_ub          the proven upper bound on the covariance matrix
    """

    # set random seed
    Random.seed!(rseed)

    # get dimensions
    n,d = size(X)

    # check that this test instance is legitimate
    # @assert(all([norm(X[i,:]) for i=1:n] .< 1.0))   # covariate vectors have norm at most 1
    @assert(10*eps() < lambda <= 1.0)                   # lambda is in the range (0, 1]

    # obtain empirical covariance of +/- 1 assignments
    _, empirical_cov = empirical_assignment_mean_cov(X, lambda, num_samples, balanced=balanced)

    # compute covariance upper bound
    cov_ub = pinv( lambda * I + (1-lambda)*X*X')

    # compute difference matrix: this is negative semidefinite if bound holds
    diff_mat = empirical_cov - cov_ub
    bound_err = eigmax(diff_mat)
    test_pass = bound_err < tol

    # print statements if verbose 
    if verbose 
        println("Testing whether GSW assignment covariance satisfies proven upper bound")
        println("\tn = ", n, "\td = ", d)
        println("\t# of samples = ", num_samples)
        println("\ttolerance = ", tol)
        println("\nBound Error: ", bound_err)
        if test_pass
            println("Test passed")
        else
            println("Test failed")
        end
    end

    return test_pass, empirical_cov, cov_ub
end
