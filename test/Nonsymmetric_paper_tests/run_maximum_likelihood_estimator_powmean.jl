using JuMP
using Revise
using Clarabel
using MosekTools
using Distributions, Random
using LinearAlgebra, SparseArrays

dist = Exponential(2.0)

# Generate random samples from the distribution
n = Int(1e4)  # Number of samples, choose 1e4,1e6,1e8

rng = Random.MersenneTwister(1)
y = rand(rng,dist, n)
sort!(y)

# Preprocessing data, remove data points that are too close to the neighborhood
ynew = [y[1]]
prev = 1
for i in 2:n
    if (y[i] - y[prev]) > 1e-2
        append!(ynew,y[i])
        global prev = i
    end
end
y = ynew[2:end] 
n = length(y)

freq = ones(n)
normalize!(freq,1)

#Result from Clarabel's power mean cone
println("power mean cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, x[1:n])
@variable(model, t)
@objective(model, Max, t)

using SparseArrays
At = spdiagm(0 =>[freq; -1.0])
@constraint(model, At*vcat(x,t) in Clarabel.MOI.DualPowerMeanCone(freq))
@constraint(model, sum((y[i+1] - y[i])*(x[i] + x[i+1])/2 for i in 1:(n-1)) == 1)
for i = 1:n-2
    @constraint(model, (x[i+1]-x[i])/(y[i+1]-y[i]) - (x[i+2]-x[i+1])/(y[i+2]-y[i+1])<= 0)
end

set_optimizer_attribute(model,"min_terminate_step_length", 1e-3)
set_optimizer_attribute(model,"max_iter", 5000)
set_optimizer_attribute(model,"tol_gap_abs", 1e-7)
set_optimizer_attribute(model,"tol_gap_rel", 1e-7)
set_optimizer_attribute(model,"tol_feas", 2e-7)
set_optimizer_attribute(model,"tol_ktratio", 1e-5)

optimize!(model)
psol = value.(x)
t3 = solution_summary(model).solve_time
iter3 = solution_summary(model).barrier_iterations

using Hypatia
#Result from Hypatia's generalized power cone
println("generalized power cones via Hypatia")
model = Model(Hypatia.Optimizer)
@variable(model, t)
@variable(model, x[1:n])
@objective(model, Max, t)
@constraint(model, vcat(t,x) in Hypatia.HypoPowerMeanCone(freq,false))
@constraint(model, sum((y[i+1] - y[i])*(x[i] + x[i+1])/2 for i in 1:(n-1)) == 1)
for i = 1:n-2
    @constraint(model, (x[i+1]-x[i])/(y[i+1]-y[i]) - (x[i+2]-x[i+1])/(y[i+2]-y[i+1])<= 0)
end

optimize!(model)
t4 = solution_summary(model).solve_time
iter4 = solution_summary(model).barrier_iterations

(t3,iter3,t4,iter4)