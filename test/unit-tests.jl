# unit-tests.jl
# Chris Harshaw, Fredrik Savje, Dan Spielman, Peng Zhang 
# January 2020
#
# These are unit tests for functions in the Gram--Schmidt Walk
#

include("../src/gsw-design.jl")

# set random seed for unit tests
Random.seed!(1234)

#==============================
# Unit Test 1:
# Check step size calculation.
===============================#
if verbose 
    println("Unit Test: step size calculation")
end

# To make it harder, use a u has near-zero values 
p = 5                           # 5 is pivot vector
u = [1.0 2.0 0.0 -eps() -3.0]   # last 5 alive not pivot
z = vcat(ones(4), zeros(6))     # ten total variables
live_not_pivot = vcat(falses(5), trues(5))

# compute step sizes
del_plus, del_minus = compute_step_sizes(z, u, live_not_pivot, p)

# check that they are equal to 1/3
@test isapprox(del_plus, 1/3)
@test isapprox(del_minus, 1/3) 

#================================
# Unit Test 2:
# Check step direction computation
==================================#
if verbose 
    println("Unit Test: step direction calculation")
end

# initial set up
n = 10
d = 4
balanced = false
cov_sum = false
lambda = 0.5

# create covariate (scaled) and the factorization
X = randn(d,n)
maxnorm = maximum([norm(X[:,i]) for i=1:n])
X /= maxnorm
M = (lambda / (1-lambda)) * I +  (X * X')
MC = cholesky(M)
cov_sum = sum(X, dims=2)

# Set Up at itereation 3: live_not_pivot lists and factorizations should be updated.
p = 3
live_not_pivot = vcat(falses(3), trues(n-3))
lowrankdowndate!(MC, X[:,1]); cov_sum -= X[:,1]                 
lowrankdowndate!(MC, X[:,2]); cov_sum -= X[:,2]  
lowrankdowndate!(MC, X[:,3]); cov_sum -= X[:,3]  

# get the necessary variables for the larger (slower) linear system solve 
B = build_stacked_matrix(X', lambda)
Bp = B[:,live_not_pivot]
bp = B[:,p]

# First: the usual GSW setting
u = compute_step_direction(MC, X, lambda, p, live_not_pivot, false, nothing)
true_u = - inv(Bp' * Bp) * (Bp' * bp)

# test length of u is corect, only defined on live not pivot vectors
@test (length(u) == n-3)
@test norm(u - true_u) <= 1e-2

# Second: the balanced GSW setting 
u = compute_step_direction(MC, X, lambda, p, live_not_pivot, true, cov_sum)

# construct system to solve in balanced case
k = sum(live_not_pivot)
        
# construct coefficient matrix
A = zeros(k+1, k+1)
A[1:k, 1:k] = Bp' * Bp
A[1:k,k+1] = ones(k)/2
A[k+1,1:k] = ones(k)

# construct rhs coefficient vector 
b = zeros(k+1)
b[1:k] = - Bp' * bp
b[k+1] = - 1

# solve the system 
y = pinv(A)*b
true_u = y[1:k]

@test (length(u) == n-3)
@test norm(u - true_u) <= 1e-2

#==============================
# Unit Test 3:
# Run Gram Schmidt Walk once
===============================#
if verbose 
    println("Unit Test: run GSW")
end

d = 10
n = 50
X = randn(n,d)
for i=1:n
    X[i,:] /= norm(X[i,:])
end
assign_list = sample_gs_walk(X, 0.5, num_samples=200)