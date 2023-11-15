using Mosek, MosekTools
using JuMP
# using Clarabel
using Distributions, Random
using LinearAlgebra

dist = Exponential(2.0)
dist = BetaPrime(1.0,2.0) 

# Generate random samples from the distribution
n = 5  # Number of samples

rng = Random.MersenneTwister(1)
y = rand(rng,dist, n)
sort!(y)

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
# @constraint(model, x .>= 0)
# MOI.set(model, MOI.Silent(), true)      #Disable printing information
optimize!(model)
xsol = value.(x)

#Clarabel
println("generalized power cones via Clarabel")
model = Model(Clarabel.Optimizer)
@variable(model, t)
@variable(model, x[1:n])
@objective(model, Max, t)

using SparseArrays
At = spdiagm(0 =>[freq; 1.0])
@constraint(model, At*vcat(x,t) in Clarabel.MOI.GenPowerCone(freq,1))
# @constraint(model, vcat(x,t) in Clarabel.MOI.PowerMeanCone(freq))
@constraint(model, sum((y[i+1] - y[i])*(x[i] + x[i+1])/2 for i in 1:(n-1)) == 1)
for i = 1:n-2
    @constraint(model, (x[i+1]-x[i])/(y[i+1]-y[i]) - (x[i+2]-x[i+1])/(y[i+2]-y[i+1])<= 0)
end
@constraint(model, x .>= 0)
set_optimizer_attribute(model,"equilibrate_enable",false)
set_optimizer_attribute(model,"max_iter", 1)
# set_optimizer_attribute(model,"barrier", -n-0.5)
optimize!(model)

solver = model.moi_backend.optimizer.model.optimizer.solver

A = solver.data.A 
q = solver.data.q
b = solver.data.b 

(m,n) = size(A)
#Mosek solver

csub = [1]
cval = [-1.0]
numvar = n
numcon = m-n

Am = deepcopy(A[1:numcon,:])
bm = deepcopy(b[1:numcon])

# task = model.moi_backend.optimizer.model.task
maketask() do task

    putstreamfunc(task,MSK_STREAM_LOG,msg -> print(msg))
    appendvars(task,numvar)
    
    appendcons(task,numcon)
    # Set up the linear part of the problem
    putclist(task,csub, cval)
    bkc = [MSK_BK_FX;
    repeat([MSK_BK_UP],numcon-1)]
    # Bound values for constraints
    blc = [bm[1]; repeat([-Inf],numcon-1)]
    buc = bm
    putacolslice(task,1,numvar+1,Am)
    putconboundslice(task,1,numcon+1,bkc,blc,buc)

    #No constraints for variables
    putvarboundsliceconst(task,1, numvar+1,MSK_BK_FR,-Inf,Inf)

    # Input the cones
    pc1 = appendprimalpowerconedomain(task,n, freq)
    # pc2 = appendprimalpowerconedomain(task,3,[4.0, 6.0])
    appendafes(task,n)
    putafefentrylist(task,
    [i for i in 1:n], # Rows
    [[i+1 for i in 1:(n-1)]; 1], #Columns
    [1.0 for i in 1:n])
    putafegslice(task,1, n+1, repeat([0.0],n))
    # Append the two conic constraints
    appendacc(task,
    pc1, # Domain
    [i for i in 1:n], # Rows from F
    nothing)

    # Input the objective sense (minimize/maximize)
    putobjsense(task,MSK_OBJECTIVE_SENSE_MINIMIZE)
    # Optimize the task
    optimize(task)
    solutionsummary(task,MSK_STREAM_MSG)

end