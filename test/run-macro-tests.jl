# set verbose to be true for these experiments
verbose = true

# Macro Test 1: recovering Bernoulli design
n = 10
tol = 0.1
num_samples = 10 * convert(Int64, ceil((n / tol)^2))
balance = false
test_pass, empirical_cov, exact_cov = matches_known_design_cov(1234, n, balance, num_samples, tol, verbose=verbose)
@test test_pass

# Macro Test 2: recovering complete randomization design
n = 10
tol = 0.1
num_samples = 10 * convert(Int64, ceil((n / tol)^2))
balance = true
test_pass, empirical_cov, exact_cov = matches_known_design_cov(1234, n, balance, num_samples, tol, verbose=verbose)
@test test_pass

# load the following matrix of covariates & GSW params for the next examples
X =   [0.580234   -0.603244  -0.330794;
      -0.31985     0.306207   0.783539; 
       0.613942   -0.313111   0.578823; 
      -0.610001   -0.661348  -0.0227628;
       0.0566846   0.820089  -0.366389; 
       0.215253   -0.491084   0.722843]
n,d = size(X)
lambda = 0.5
balanced = false

# Macro Test 3: empirical covariance looks as expected for a small example
tol = 0.1
num_samples = 10 * convert(Int64, ceil((n / tol)^2))
test_pass, empirical_cov, exact_cov = matches_enumerated_design_cov(1234, X, lambda, balanced, num_samples, tol, verbose=verbose)
@test test_pass

# Macro Test 4: marginal treatment probabilities are correct
treatment_probs = 0.7 * ones(n)
tol = 0.1
num_samples = 10 * convert(Int64, ceil(n * (log(n) / tol)^2))
test_pass, empirical_probs = correct_treatment_probabilities(1234, treatment_probs, X, lambda, balanced, num_samples, tol; verbose=verbose)
@test test_pass

# Macro Test 5:  covariance satisifes proven upper bound for a small example
tol = 0.1
num_samples = 10 * convert(Int64, ceil((n / tol)^2))
test_pass, empirical_cov, cov_ub = test_cov_bound(1234, X, lambda, num_samples, tol; verbose=verbose)
@test test_pass 

