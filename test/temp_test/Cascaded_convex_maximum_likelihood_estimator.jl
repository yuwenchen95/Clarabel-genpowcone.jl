using JuMP
# using Clarabel
using Distributions, Random
using LinearAlgebra

dist = Exponential(2.0)
# dist = BetaPrime(1.0,2.0) 

# Generate random samples from the distribution
nbase = 100
sect = 10
n = nbase*sect  # Number of samples

rng = Random.MersenneTwister(1)
y = rand(rng,dist, n)
sort!(y)

# ind = Int(ceil(0.1*n))
# y = y[ind:end]
# n = n-ind+1
freq = ones(n)+ rand(rng,n)
# freq = ones(n)
normalize!(freq,1)


#Result from Clarabel's generalized power cone
println("Three-dimensional cones via Mosek")
model = Model(Clarabel.Optimizer)
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
# @constraint(model, x .>= 0)
# MOI.set(model, MOI.Silent(), true)      #Disable printing information
optimize!(model)
xsol = value.(x)

#Result from Clarabel's generalized power cone
println("Three-dimensional cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, x[1:n])
@variable(model,z[1:n-1])
@variable(model,r[1:n-2])
@objective(model, Max, z[end])
# trnasform a general power cone into a product of three-dimensional power cones
power = freq[1] + freq[2]
@constraint(model, [freq[2]/power; 1-freq[2]/power; 1.0].*vcat(x[2],x[1],z[1]) in Clarabel.MOI.GenPowerCone([freq[2]/power, 1-freq[2]/power],1))
for i = 1:n-2
    global power += freq[i+2]
    @constraint(model, r[i] == z[i])
    @constraint(model, [freq[i+2]/power; 1-freq[i+2]/power; 1.0].*vcat(x[i+2],r[i],z[i+1]) in Clarabel.MOI.GenPowerCone([freq[i+2]/power, 1-freq[i+2]/power],1))
end
@constraint(model, sum((y[i+1] - y[i])*(x[i] + x[i+1])/2 for i in 1:(n-1)) == 1)
for i = 1:n-2
    @constraint(model, (x[i+1]-x[i])/(y[i+1]-y[i]) - (x[i+2]-x[i+1])/(y[i+2]-y[i+1])<= 0)
end
# @constraint(model, x .>= 0)

# set_optimizer_attribute(model,"equilibrate_enable",false)
set_optimizer_attribute(model,"up_barrier", 1.0)
set_optimizer_attribute(model,"low_barrier", 0.5)
# set_optimizer_attribute(model,"static_regularization_constant",0.0)
# set_optimizer_attribute(model,"equilibrate_max_iter",100)
set_optimizer_attribute(model,"min_terminate_step_length", 1e-4)
set_optimizer_attribute(model,"cratio",0.95)
set_optimizer_attribute(model,"max_iter", 5000)
set_optimizer_attribute(model,"tol_gap_abs", 1e-6)
set_optimizer_attribute(model,"tol_gap_rel", 1e-6)
set_optimizer_attribute(model,"tol_feas", 1e-6)
set_optimizer_attribute(model,"tol_ktratio", 1e-4)
set_optimizer_attribute(model,"equilibrate_max_iter",3)
set_optimizer_attribute(model,"equilibrate_min_scaling",1e-2)
set_optimizer_attribute(model,"equilibrate_max_scaling",1e2)
optimize!(model)

solver = model.moi_backend.optimizer.model.optimizer.solver
