using JuMP
using Revise
using Clarabel
using MosekTools
using Distributions, Random
using LinearAlgebra, SparseArrays
# using StatProfilerHTML

dist = Exponential(1.0)
# dist = BetaPrime(1.0,2.0) 

# Generate random samples from the distribution
n = Int(1e4)  # Number of samples

rng = Random.MersenneTwister(1)
y = rand(rng,dist, n)
sort!(y)

# # Preprocessing data, remove data points that are too close to the neighborhood
# ynew = [y[1]]
# prev = 1
# for i in 2:n
#     if (y[i] - y[prev]) > 1e-2
#         append!(ynew,y[i])
#         global prev = i
#     end
# end
# y = ynew[2:end] 
# n = length(y)


# ind = Int(ceil(0.1*n))
# y = y[ind:end]
# n = n-ind+1
# freq = ones(n)+ rand(rng,n)
freq = ones(n)
normalize!(freq,1)

#Result from Clarabel's generalized power cone
println("Three-dimensional cones via Mosek")
model = Model(Mosek.Optimizer)
set_optimizer_attribute(model,"MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-7)
set_optimizer_attribute(model,"MSK_DPAR_INTPNT_CO_TOL_MU_RED", 1e-7)
set_optimizer_attribute(model,"MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-7)
set_optimizer_attribute(model,"MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-7)
@variable(model, x[1:n])
@variable(model,z[1:n-1])
@variable(model,r[1:n-2])
@objective(model, Max, z[end])
# trnasform a general power cone into a product of three-dimensional power cones
power = freq[1] + freq[2]
@constraint(model, vcat(x[2],x[1],z[1]) in MOI.PowerCone(freq[2]/power))
for i = 1:n-2
    global power += freq[i+2]
    @constraint(model, r[i] == z[i])
    @constraint(model, vcat(x[i+2],r[i],z[i+1]) in MOI.PowerCone(freq[i+2]/power))
end
@constraint(model, sum((y[i+1] - y[i])*(x[i] + x[i+1])/2 for i in 1:(n-1)) == 1)
for i = 1:n-2
    @constraint(model, (x[i+1]-x[i])/(y[i+1]-y[i]) - (x[i+2]-x[i+1])/(y[i+2]-y[i+1])<= 0)
end
# set_optimizer_attribute(model,"MSK_IPAR_PRESOLVE_USE",false)
# MOI.set(model, MOI.Silent(), true)      #Disable printing information
optimize!(model)
xsol = value.(x)
t1 = solution_summary(model).solve_time
iter1 = solution_summary(model).barrier_iterations

# #Result from Clarabel's generalized power cone
println("Three-dimensional cones via Clarabel")
#Preprocessing


#Result from Clarabel's generalized power cone
println("generalized power cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, x[1:n])
@variable(model, t)
@objective(model, Max, t)

using SparseArrays
# @constraint(model, vcat(x,t) in Clarabel.MOI.GenPowerCone(freq,1))
@constraint(model, sum((y[i+1] - y[i])*(x[i] + x[i+1])/2 for i in 1:(n-1)) == 1)
for i = 1:n-2
    @constraint(model, (x[i+1]-x[i])/(y[i+1]-y[i]) - (x[i+2]-x[i+1])/(y[i+2]-y[i+1])<= 0)
end
# @constraint(model, x .>= 0)

set_optimizer_attribute(model,"equilibrate_enable",false)
set_optimizer_attribute(model,"verbose",false)
optimize!(model)
solver = model.moi_backend.optimizer.model.optimizer.solver

At = SparseMatrixCSC(solver.data.A')
b = solver.data.b 
q = solver.data.q
len_nn = n-2    #number of constraints
#Generate new problem
model = Model(Clarabel.Optimizer)
@variable(model, x[1:n-1])
@objective(model, Min, dot(b,x))
@constraint(model, q + At*x in Clarabel.MOI.DualGenPowerCone(freq,1))
# @constraint(model, q + At*x in Clarabel.MOI.DualPowerMeanCone(freq))
@constraint(model, x[2:n-1] .>= 0)
set_optimizer_attribute(model,"neighborhood", 1e-4)
set_optimizer_attribute(model,"tol_gap_abs", 1e-6)
set_optimizer_attribute(model,"tol_gap_rel", 1e-6)
set_optimizer_attribute(model,"tol_feas", 1e-6)
set_optimizer_attribute(model,"tol_ktratio", 1e-4)
set_optimizer_attribute(model,"min_terminate_step_length",1e-3)
optimize!(model)
solver = model.moi_backend.optimizer.model.optimizer.solver