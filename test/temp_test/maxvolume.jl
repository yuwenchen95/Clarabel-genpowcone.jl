using LinearAlgebra, SparseArrays
using JuMP
# using Mosek,MosekTools
# using JLD,JLD2
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
coef = 1.0
#######################################################################
# YC: Benchmarking should be implemented separately if you want to 
#     obtain the plot.
#######################################################################

#Result from Mosek
println("Three-dimensional cones via Mosek")
model = Model(Mosek.Optimizer)
set_optimizer_attribute(model,"MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-7)
set_optimizer_attribute(model,"MSK_DPAR_INTPNT_CO_TOL_MU_RED", 1e-7)
set_optimizer_attribute(model,"MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-7)
set_optimizer_attribute(model,"MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-7)
@variable(model, p[1:n])
@variable(model,q[1:n-1])
@variable(model,r[1:n-2])
@objective(model, Max, q[end])
# trnasform a general power cone into a product of 3x3 power cones
power = freq[1] + freq[2]
@constraint(model, vcat(p[2],p[1],q[1]) in MOI.PowerCone(freq[2]/power))
for i = 1:n-2
    global power += freq[i+2]
    @constraint(model, r[i] == q[i])
    @constraint(model, vcat(p[i+2],r[i],q[i+1]) in MOI.PowerCone(freq[i+2]/power))
end
@constraint(model, vcat(gamma, coef*A * p) in MOI.NormInfinityCone(n + 1))
@constraint(model, vcat(sqrt(n) * gamma, A * p) in MOI.NormOneCone(n + 1))
# MOI.set(model, MOI.Silent(), true)
# set_optimizer_attribute(model,"MSK_IPAR_PRESOLVE_USE",false)
optimize!(model)
opt_val = objective_value(model)
psol = value.(p)
t1 = solution_summary(model).solve_time
iter1 = solution_summary(model).barrier_iterations

#Result from Clarabel's generalized power cone
println("Three-dimensional cones via Clarabel")
model = Model(Clarabel.Optimizer)
set_optimizer_attribute(model,"tol_gap_abs", 1e-7)
set_optimizer_attribute(model,"tol_gap_rel", 1e-7)
set_optimizer_attribute(model,"tol_feas", 1e-7)
set_optimizer_attribute(model,"tol_ktratio", 1e-5)
set_optimizer_attribute(model,"max_iter",2000)
set_optimizer_attribute(model,"min_switch_step_length", 1e-1)
@variable(model, p[1:n])
@variable(model,q[1:n-1])
@variable(model,r[1:n-2])
@objective(model, Max, q[end])
# trnasform a general power cone into a product of 3x3 power cones
power = freq[1] + freq[2]
@constraint(model, vcat(p[2],p[1],q[1]) in MOI.PowerCone(freq[2]/power))
for i = 1:n-2
    global power += freq[i+2]
    @constraint(model, r[i] == q[i])
    @constraint(model, vcat(p[i+2],r[i],q[i+1]) in MOI.PowerCone(freq[i+2]/power))
end
@constraint(model, vcat(gamma, coef * A * p) in MOI.NormInfinityCone(n + 1))
@constraint(model, vcat(sqrt(n) * gamma, A * p) in MOI.NormOneCone(n + 1))
# MOI.set(model, MOI.Silent(), true)
# set_optimizer_attribute(model,"MSK_IPAR_PRESOLVE_USE",false)
optimize!(model)
opt_val = objective_value(model)
psol = value.(p)
t2 = solution_summary(model).solve_time
iter2 = solution_summary(model).barrier_iterations
solver = model.moi_backend.optimizer.model.optimizer.solver

#Result from Clarabel's generalized power cone
println("generalized power cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, t)
@variable(model, x[1:n])
@objective(model, Max, t)
At = spdiagm(0 =>[freq; 1.0])
@constraint(model, At*vcat(x,t) in Clarabel.MOI.DualGenPowerCone(freq,1))
# At = spdiagm(0 =>[freq; -1.0])
# @constraint(model, At*vcat(x,t) in Clarabel.MOI.DualPowerMeanCone(freq))
# @constraint(model, vcat(gamma, A * x) in MOI.SecondOrderCone(n + 1))
@constraint(model, vcat(gamma, coef * A * x) in MOI.NormInfinityCone(n + 1))
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
@constraint(model, vcat(x,t) in Hypatia.GeneralizedPowerCone(freq,1,false))
# @constraint(model, vcat(gamma, A * x) in MOI.SecondOrderCone(n + 1))
@constraint(model, vcat(gamma, coef * A * x) in MOI.NormInfinityCone(n + 1))
@constraint(model, vcat(sqrt(n) * gamma, A * x) in MOI.NormOneCone(n + 1))
# MOI.set(model, MOI.Silent(), true)
optimize!(model)
t4 = solution_summary(model).solve_time
iter4 = solution_summary(model).barrier_iterations

(t3,iter3,t4,iter4,t2,iter2,t1,iter1)