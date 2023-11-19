using JuMP
# using Clarabel
using Distributions, Random
using LinearAlgebra, SparseArrays

dist = Exponential(2.0)
# dist = BetaPrime(1.0,2.0) 

# Generate random samples from the distribution
n = 2000000  # Number of samples

rng = Random.MersenneTwister(1)
y = rand(rng,dist, n)
sort!(y)

#Preprocessing data, remove data points that are too close to the neighborhood
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


# ind = Int(ceil(0.1*n))
# y = y[ind:end]
# n = n-ind+1
# freq = ones(n)+ rand(rng,n)
freq = ones(n)
normalize!(freq,1)

#Result from Clarabel's generalized power cone
println("Three-dimensional cones via Mosek")
model = Model(Mosek.Optimizer)
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
# @constraint(model, x .>= 0)
# MOI.set(model, MOI.Silent(), true)      #Disable printing information
optimize!(model)
xsol = value.(x)

# #Result from Clarabel's generalized power cone
# println("Three-dimensional cones via Clarabel")
# model = Model(Clarabel.Optimizer)
# @variable(model, x[1:n])
# @variable(model,z[1:n-1])
# @variable(model,r[1:n-2])
# @objective(model, Max, z[end])
# # trnasform a general power cone into a product of three-dimensional power cones
# power = freq[1] + freq[2]
# @constraint(model, vcat(x[2],x[1],z[1]) in MOI.PowerCone(freq[2]/power))
# # @constraint(model, spdiagm(0=> [freq[2]/power, 1-freq[2]/power, 1.0])*vcat(x[2],x[1],z[1]) in Clarabel.MOI.GenPowerCone([freq[2]/power, 1-freq[2]/power],1))
# for i = 1:n-2
#     global power += freq[i+2]
#     @constraint(model, r[i] == z[i])
#     @constraint(model, vcat(x[i+2],r[i],z[i+1]) in MOI.PowerCone(freq[i+2]/power))
#     # @constraint(model, spdiagm(0=> [freq[i+2]/power, 1-freq[i+2]/power, 1.0])*vcat(x[i+2],r[i],z[i+1]) in Clarabel.MOI.GenPowerCone([freq[i+2]/power, 1-freq[i+2]/power],1))
# end
# @constraint(model, sum((y[i+1] - y[i])*(x[i] + x[i+1])/2 for i in 1:(n-1)) == 1)
# for i = 1:n-2
#     @constraint(model, (x[i+1]-x[i])/(y[i+1]-y[i]) - (x[i+2]-x[i+1])/(y[i+2]-y[i+1])<= 0)
# end
# # @constraint(model, x .>= 0)
# # MOI.set(model, MOI.Silent(), true)      #Disable printing information
# set_optimizer_attribute(model,"min_switch_step_length",0.01)
# # set_optimizer_attribute(model,"linesearch_backtrack_step",0.9)
# set_optimizer_attribute(model,"max_iter", 50)
# optimize!(model)

#Result from Clarabel's generalized power cone
println("generalized power cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, x[1:n])
@variable(model, t)
@objective(model, Max, t)

using SparseArrays
At = spdiagm(0 =>[freq; 1.0])
@constraint(model, At*vcat(x,t) in Clarabel.MOI.GenPowerCone(freq,1))
# @constraint(model, vcat(x,t) in Clarabel.MOI.PowerMeanCone(freq))
@constraint(model, sum((y[i+1] - y[i])*(x[i] + x[i+1])/2 for i in 1:(n-1)) == 1)
for i = 1:n-2
    @constraint(model, (x[i+1]-x[i])/(y[i+1]-y[i]) - (x[i+2]-x[i+1])/(y[i+2]-y[i+1])<= 0)
end
# @constraint(model, x .>= 0)

# set_optimizer_attribute(model,"equilibrate_enable",false)
set_optimizer_attribute(model,"up_barrier", 1.0)
set_optimizer_attribute(model,"low_barrier", 0.5)
# set_optimizer_attribute(model,"static_regularization_constant",0.0)
set_optimizer_attribute(model,"min_terminate_step_length", 1e-3)
set_optimizer_attribute(model,"cratio",0.95)
set_optimizer_attribute(model,"max_iter", 500)
set_optimizer_attribute(model,"tol_gap_abs", 1e-7)
set_optimizer_attribute(model,"tol_gap_rel", 1e-7)
set_optimizer_attribute(model,"tol_feas", 1e-7)
set_optimizer_attribute(model,"tol_ktratio", 1e-5)
# set_optimizer_attribute(model,"equilibrate_max_iter",20)
# set_optimizer_attribute(model,"equilibrate_min_scaling",1e-4)
# set_optimizer_attribute(model,"equilibrate_max_scaling",1e4)
# set_optimizer_attribute(model,"barrier", -n-0.5)
optimize!(model)
psol = value.(x)

solver = model.moi_backend.optimizer.model.optimizer.solver

# #Use different precision
# T = BigFloat
# Pb = T.(solver.data.P)
# qb = T.(solver.data.q)
# Ab = T.(solver.data.A)
# bb = T.(solver.data.b)

# cones = [Clarabel.ZeroConeT(1),           
#         Clarabel.NonnegativeConeT(n-2),
#         Clarabel.GenPowerConeT(freq,1)
#         # Clarabel.PowerMeanConeT(freq)
#         ]

# settings = Clarabel.Settings{BigFloat}(
#     verbose = true,
#     direct_kkt_solver = true,
#     direct_solve_method = :qdldl,
#     up_barrier = 1.0,
#     low_barrier = 0.5,
#     min_terminate_step_length = 1e-3,
#     cratio = 0.95,
#     max_iter = 5000,
#     equilibrate_enable = false
#     )
# setprecision(BigFloat,128)

# solver   = Clarabel.Solver{BigFloat}()
# Clarabel.setup!(solver, Pb, qb, Ab, bb, cones, settings)
# result = Clarabel.solve!(solver)


# using Hypatia
# #Result from Hypatia's generalized power cone
# println("generalized power cones via Hypatia")
# model = Model(Hypatia.Optimizer)
# @variable(model, t)
# @variable(model, x[1:n])
# @objective(model, Max, t)
# @constraint(model, vcat(x,t) in Hypatia.GeneralizedPowerCone(freq,1,false))
# # @constraint(model, vcat(t,x) in Hypatia.HypoPowerMeanCone(freq,false))
# @constraint(model, sum((y[i+1] - y[i])*(x[i] + x[i+1])/2 for i in 1:(n-1)) == 1)
# for i = 1:n-2
#     @constraint(model, (x[i+1]-x[i])/(y[i+1]-y[i]) - (x[i+2]-x[i+1])/(y[i+2]-y[i+1])<= 0)
# end
# @constraint(model, x .>= 0)

# optimize!(model)


# function barriergenpow(x)
#     result = one(eltype(x))

#     for i in 1:length(x)-1
#         result *= (x[i]/a[i])^(2*a[i])
#     end

#     result -= x[end]^2
#     println("residual is ", result)
#     res = -log(result)

#     for i in 1:length(x)-1
#         res -= (1-a[i])*log(x[i]/a[i])
#     end

#     return res
# end

# function gradientClarabel(z)
    
#     # ϕ = ∏_{i ∈ dim1}(ui/αi)^(2*αi), ζ = ϕ - ||w||^2
#     phi = one(eltype(z))
#     @inbounds for i = 1:(length(z)-1)
#         phi *= (z[i]/a[i])^(2*a[i])
#     end
#     norm2w = z[end]*z[end]
#     ζ = phi - norm2w
#     @assert ζ > zero(T)

#     # compute the gradient at z
#     grad = similar(z)
#     τ = similar(z)

#     @inbounds for i = 1:(length(z)-1)
#         τ[i] = 2*a[i]/z[i]
#         grad[i] = -τ[i]*phi/ζ - (1-a[i])/z[i]
#     end
    
#     grad[end] = 2*z[end]/ζ

#     return grad
# end

# function hessianClarabel(
#     z::AbstractVector{T}
# ) where {T}
    
#     len = length(z) 
#     p = similar(z)
#     q = zeros(len-1)
#     r = zero(T)
#     d1 = similar(q)

#     # ϕ = ∏_{i ∈ dim1}(ui/αi)^(2*αi), ζ = ϕ - ||w||^2
#     phi = one(T)
#     @inbounds for i = 1:(len-1)
#         phi *= (z[i]/a[i])^(2*a[i])
#     end
#     norm2w = z[end]*z[end]
#     ζ = phi - norm2w
#     @assert ζ > zero(T)

#     # compute the gradient at z
#     grad = similar(z)
#     τ = q           # τ shares memory with q

#     @inbounds for i = 1:(len-1)
#         τ[i] = 2*a[i]/z[i]
#         grad[i] = -τ[i]*phi/ζ - (1-a[i])/z[i]
#     end
    
#     grad[end] = 2*z[end]/ζ
    

#     # compute Hessian information at z 
#     p0 = sqrt(phi*(phi+norm2w)/2)
#     p1 = -2*phi/p0
#     q0 = sqrt(ζ*phi/2)
#     r1 = 2*sqrt(ζ/(phi+norm2w))

#     # compute the diagonal d1,d2
#     @inbounds for i = 1:(len-1)
#         d1[i] = τ[i]*phi/(ζ*z[i]) + (1-a[i])/(z[i]*z[i])
#     end   
#     d2 = 2/ζ

#     # compute p, q, r where τ shares memory with q
#     @. p[1:(len-1)] = p0*τ/ζ
#     p[end] = p1*z[end]/ζ

#     q .*= q0/ζ      #τ is abandoned
#     r = r1*z[end]/ζ

#     return (p,q,r,d1,d2)

# end

