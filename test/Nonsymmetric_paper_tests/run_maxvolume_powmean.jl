using LinearAlgebra, SparseArrays
using JuMP
using Revise
using Random
using Clarabel
using Hypatia

"""
Maximum volume hypercube} from Hypatia.jl,

https://github.com/chriscoey/Hypatia.jl/tree/master/examples/maxvolume,
"""

n = 2500
rng = Random.MersenneTwister(1)
# ensure there will be a feasible solution
x = randn(rng,n)
A = sparse(1.0*I(n))
gamma = norm(A * x) / sqrt(n)
freq = ones(n) #+ rand(rng,n)
normalize!(freq,1)

#######################################################################
# YC: Benchmarking should be implemented separately if you want to 
#     obtain the plot.
#######################################################################
#Result from Clarabel's power mean cone
println("power mean cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, t)
@variable(model, x[1:n])
@objective(model, Max, t)
At = spdiagm(0 =>[freq; -1.0])
@constraint(model, At*vcat(x,t) in Clarabel.MOI.DualPowerMeanCone(freq))
@constraint(model, vcat(gamma, A * x) in MOI.NormInfinityCone(n + 1))
@constraint(model, vcat(sqrt(n) * gamma, A * x) in MOI.NormOneCone(n + 1))
set_optimizer_attribute(model,"min_switch_step_length",0.001)

set_optimizer_attribute(model,"tol_gap_abs", 1e-7)
set_optimizer_attribute(model,"tol_gap_rel", 1e-7)
set_optimizer_attribute(model,"tol_feas", 1e-7)
set_optimizer_attribute(model,"tol_ktratio", 1e-5)
optimize!(model)
t3 = solution_summary(model).solve_time
iter3 = solution_summary(model).barrier_iterations

#Result from Hypatia
println("power mean cones via Hypatia")
model = Model(Hypatia.Optimizer)
@variable(model, t)
@variable(model, x[1:n])
@objective(model, Max, t)
@constraint(model, vcat(t,x) in Hypatia.HypoPowerMeanCone(freq,false))
@constraint(model, vcat(gamma, A * x) in MOI.NormInfinityCone(n + 1))
@constraint(model, vcat(sqrt(n) * gamma, A * x) in MOI.NormOneCone(n + 1))
# MOI.set(model, MOI.Silent(), true)
optimize!(model)
t4 = solution_summary(model).solve_time
iter4 = solution_summary(model).barrier_iterations

(t3,iter3,t4,iter4)