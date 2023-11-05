using JuMP
# using Clarabel
using Distributions, Random
using LinearAlgebra

dist = Exponential(2.0)
# dist = BetaPrime(1.0,2.0) 

# Generate random samples from the distribution
nbase = 50
level = 2
n = nbase^level  # Number of samples

rng = Random.MersenneTwister(1)
y = rand(rng,dist, n)
sort!(y)

# ind = Int(ceil(0.1*n))
# y = y[ind:end]
# n = n-ind+1
# freq = ones(n)+ rand(rng,n)
freq = ones(n)
normalize!(freq,1)

#Cascading frequency
freqTable = Vector{Vector{Float64}}()
function generate_cascaded_freq(
    freqTable::Vector{Vector{T}},
    freq::Vector{T},
    nbase::Int
) where {T}
    push!(freqTable,freq)
    len = length(freqTable[end])
    while (len > 1)
        len = Int(length(freqTable[end])/nbase)
        prevfreq = freqTable[end]
        newfreq = zeros(T, len)

        for i in 1:len
            @views slice = prevfreq[(i-1)*nbase+1:i*nbase]
            newfreq[i] = sum(slice)
            @. slice /= newfreq[i]
        end

        push!(freqTable,newfreq)
    end
end
generate_cascaded_freq(freqTable,deepcopy(freq),nbase)




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
@constraint(model, x .>= 0)
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
# for i = 1:n-2
#     global power += freq[i+2]
#     @constraint(model, r[i] == z[i])
#     @constraint(model, vcat(x[i+2],r[i],z[i+1]) in MOI.PowerCone(freq[i+2]/power))
# end
# @constraint(model, sum((y[i+1] - y[i])*(x[i] + x[i+1])/2 for i in 1:(n-1)) == 1)
# for i = 1:n-2
#     @constraint(model, (x[i+1]-x[i])/(y[i+1]-y[i]) - (x[i+2]-x[i+1])/(y[i+2]-y[i+1])<= 0)
# end
# @constraint(model, x .>= 0)
# # MOI.set(model, MOI.Silent(), true)      #Disable printing information
# set_optimizer_attribute(model,"min_switch_step_length",0.01)
# set_optimizer_attribute(model,"max_iter", 500)
# optimize!(model)

#Result from Clarabel's generalized power cone
println("generalized power cones via Clarabel")
model = Model(Clarabel.Optimizer)
total_len = sum(length(a) for a in freqTable)
@variable(model,x[1:total_len])
@variable(model,t[1:(total_len-length(freq))])
@objective(model, Max, t[end])
@constraint(model,x[length(freq)+1:end] .== t)

using SparseArrays
start_ind = 0
start_t = 0
for (depth,cur_freq) in enumerate(freqTable)
    cur_len = length(cur_freq)

    if (depth == length(freqTable))
        break
    end

    for i in 1:Int(cur_len/nbase)
        inds = (i-1)*nbase+1:(i*nbase)
        @views slice = cur_freq[inds]
        At = spdiagm(0 =>[deepcopy(slice); 1.0])
        global start_t += 1
        @constraint(model, At*vcat(x[start_ind .+ (inds)],t[start_t]) in Clarabel.MOI.GenPowerCone(slice,1))
    end

    global start_ind += cur_len
end

# @constraint(model, vcat(x,t) in Clarabel.MOI.PowerMeanCone(freq))
@constraint(model, sum((y[i+1] - y[i])*(x[i] + x[i+1])/2 for i in 1:(n-1)) == 1)
for i = 1:n-2
    @constraint(model, (x[i+1]-x[i])/(y[i+1]-y[i]) - (x[i+2]-x[i+1])/(y[i+2]-y[i+1])<= 0)
end
@constraint(model, x .>= 0)

# set_optimizer_attribute(model,"equilibrate_enable",false)
set_optimizer_attribute(model,"up_barrier", 1.0)
set_optimizer_attribute(model,"low_barrier", 0.5)
# set_optimizer_attribute(model,"static_regularization_constant",0.0)
# set_optimizer_attribute(model,"equilibrate_max_iter",100)
set_optimizer_attribute(model,"min_terminate_step_length", 1e-4)
set_optimizer_attribute(model,"cratio",0.95)
set_optimizer_attribute(model,"max_iter", 5000)
# set_optimizer_attribute(model,"barrier", -n-0.5)
optimize!(model)

solver = model.moi_backend.optimizer.model.optimizer.solver

# #Use different precision
# T = BigFloat
# Pb = T.(solver.data.P)
# qb = T.(solver.data.q)
# Ab = T.(solver.data.A)
# bb = T.(solver.data.b)

# cones = [Clarabel.ZeroConeT(1),           
#         Clarabel.NonnegativeConeT(2*n-2),
#         Clarabel.GenPowerConeT(freq,1)
#         # Clarabel.PowerMeanConeT(freq)
#         ]

# settings = Clarabel.Settings{BigFloat}(
#     verbose = true,
#     direct_kkt_solver = true,
#     direct_solve_method = :qdldl,
#     # equilibrate_enable = false
#     )
# setprecision(BigFloat,256)

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

