using LinearAlgebra, SparseArrays
using JuMP
using Revise
using Random
using Clarabel
using Hypatia
# using BenchmarkTools

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
#Result from Clarabel's generalized power cone
println("generalized power cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, t)
@variable(model, x[1:n])
@objective(model, Max, t)
At = spdiagm(0 =>[freq; -1.0])
@constraint(model, At*vcat(x,t) in Clarabel.MOI.DualPowerMeanCone(freq))
# @constraint(model, vcat(gamma, A * x) in MOI.SecondOrderCone(n + 1))
@constraint(model, vcat(gamma, A * x) in MOI.NormInfinityCone(n + 1))
@constraint(model, vcat(sqrt(n) * gamma, A * x) in MOI.NormOneCone(n + 1))
# MOI.set(model, MOI.Silent(), true)      #Disable printing information
set_optimizer_attribute(model,"cratio",1.0)
set_optimizer_attribute(model,"max_iter",2000)
set_optimizer_attribute(model,"min_switch_step_length",0.001)
# set_optimizer_attribute(model,"equilibrate_enable", false)

set_optimizer_attribute(model,"tol_gap_abs", 1e-7)
set_optimizer_attribute(model,"tol_gap_rel", 1e-7)
set_optimizer_attribute(model,"tol_feas", 1e-7)
set_optimizer_attribute(model,"tol_ktratio", 1e-5)
optimize!(model)
t3 = solution_summary(model).solve_time
iter3 = solution_summary(model).barrier_iterations
# @assert isapprox(opt_val,objective_value(model),atol = 1e-4)
# solver = model.moi_backend.optimizer.model.optimizer.solver
xsol = value.(x)

# #Use different precision
# T = BigFloat
# Pb = T.(solver.data.P)
# qb = T.(solver.data.q)
# Ab = T.(solver.data.A)
# bb = T.(solver.data.b)

# cones = [Clarabel.NonnegativeConeT(4*n+1),
#         Clarabel.GenPowerConeT(freq,1)
#         # Clarabel.PowerMeanConeT(freq)
#         ]

# settings = Clarabel.Settings{BigFloat}(
#     verbose = true,
#     direct_kkt_solver = true,
#     direct_solve_method = :qdldl,
#     cratio = 0.95,
#     up_barrier = 1,
#     low_barrier = 1,
#     min_switch_step_length = 0.1,
#     equilibrate_enable = false
#     )
# setprecision(BigFloat,128)

# solver   = Clarabel.Solver{BigFloat}()
# Clarabel.setup!(solver, Pb, qb, Ab, bb, cones, settings)
# result = Clarabel.solve!(solver)

#Result from Hypatia
println("generalized power cones via Hypatia")
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