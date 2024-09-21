# Tests for the nonsymmetric paper

We are going to solve the following problem within __Clarabel.jl__:

$$
\begin{array}{r}
\text{minimize} &  q^T x\\\\[2ex]
\text{subject to} & Ax + s = b \\\\[1ex]
        & s \in \mathcal{K}
\end{array}

$$

with decision variables$x \in \mathbb{R}^n$, $s \in \mathbb{R}^m$ and $q \in \mathbb{R}^n$, $A \in \mathbb{R}^{m \times n}$, and $b \in \mathbb{R}^m$.
The convex set $\mathcal{K}$ is a composition of convex cones. Currently, we support the dual cones of nonsymmetric cones detailed in the paper:

```
@misc{chen2023efficient,
      title={An Efficient IPM Implementation for A Class of Nonsymmetric Cones}, 
      author={Yuwen Chen and Paul Goulart},
      year={2023},
      eprint={2305.12275},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```

## Installation

- The code can be downloaded by calling `pkg> dev https://github.com/yuwenchen95/Clarabel-genpowcone.jl.git` under the [Pkg mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Pkg-mode) of Julia [REPL](https://docs.julialang.org/en/v1/stdlib/REPL/#The-Julia-REPL).

## Running tests in the paper

All tests for the paper is in the folder `.\test\Nonsymmetric_paper_tests\`, where we can call the follwing command to run the follwing tests:

#### Maximum likelihood estimation

```julia
include("run_maximum_likelihood_estimator_genpow.jl")     #implementation for generalized power cones
include("run_maximum_likelihood_estimator_powmean.jl")    #implementation for power mean cones
```

#### Volume maximization problems

```julia
include("run_maxvolume_genpow.jl")                      #implementation for generalized power cones
include("run_maxvolume_powmean.jl")                     #implementation for power mean cones
```

#### Signomial problems

```julia
include("run_signomial.jl")
```

#### Barcenter problems

```julia
include("run_barcenter.jl")
```
