# unit-tests.jl
# Chris Harshaw, Fredrik Savje, Dan Spielman, Peng Zhang 
# January 2020
#
# These are unit tests for functions in the Gram--Schmidt Walk
#

#  run the Gram--Schmidt walk on a small example
d = 10
n = 50
X = randn(n,d)
for i=1:n
    X[i,:] /= norm(X[i,:])
end
assign_list = sample_gs_walk(X, 0.5, num_samples=200)