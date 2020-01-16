# GSWDesign.jl
GSWDesign is a Julia package which contains a fast implementation of the Gram--Schmidt Walk 
for balancing covariates in randomized experiments. 
The Gram--Schmidt Walk design allows experimenters the flexibilty to control the amount of covariate balancing.
See the references below for details of the Gram--Schmidt Walk design and its analysis.

1. Christopher Harshaw, Fredrik S&auml;vje, Daniel Spielman, Peng Zhang. "Balancing covariates in randomized experiments
using the Gram–Schmidt walk". Arxiv 1911.03071, 2019. [arxiv](https://arxiv.org/abs/1911.03071)
2. Nikhil Bansal, Daniel Dadush, Shashwat Garg, and Shachar Lovett. "The Gram–Schmidt walk: a
cure for the Banaszczyk blues". In STOC, 2018. [arxiv](https://arxiv.org/abs/1708.01079)

## Installing this package
To install this package, you must first have installed the Julia programming language.
If you have not done this, visit [julialang.org](https://julialang.org/) for instructions on how to do so.

The best way to install this package is using Julia's builtin package manager, `Pkg`. 
GSWDesign is currently an unregistered package and so the command for adding it is slightly different than for registered packages.
We discuss how to do this for Julia versions v1.0 and higher.
1. At the command line, type `julia` to enter an interactive Julia session.
2. Enter the package manager by pressing `]`.
3. To download our GSWDesign package, enter the command `add https://github.com/crharshaw/GSWDesign.jl`
4. Exit the package manager by pressing press backspace or ^C.

Now GSWDesign is installed with your version of Julia and you can use it.

## How to use this package
The main functionality of this package is the function `sample_gs_walk` which is a fast implementation to sample
from the Gram--Schmidt Walk design. 
The function `sample_gs_walk` maintains a Cholesky factorization so the runtime for producing one sample from the design scales like $O(n^2 d)$ for $n > d$, where $n$ is the number of units and $d$ is the number of covariates for each unit.
Moreover, a recursive implementation is used for more efficient memory allocation.

Here is an example of how to use the function `sample_gs_walk`.

```julia
# import the package
using GSWDesign

# generate a random matrix of covariates
n = 20
d = 4
X = randn(n,d)

# run the Gram--Schmidt walk
lambda = 0.5
assignment_list = sample_gs_walk(X, lambda, num_samples=5)
```
Note that covariate matrices are n-by-d, so the covariates vectors for each unit are the *rows* of the matrix $X$.
The return variable `assignment_list` is an array of size `num_samples`-by-n with entries in 0 and 1.
The function `sample_gs_walk` has several other features, including the ability to set individual assignment probabilities and the option for strictly balanced assignments. 
A complete description of these features is available by typing `?sample_gs_walk` in the interactive terminal.

Please see our Jupyter notebooks in the `notebooks` directory for more example usage.

