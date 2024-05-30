using LinearAlgebra
using Plots
pyplot()

using JuMP

using Mosek
using MosekTools
using Clarabel

using Statistics

#Define the number of images for the barycenter calculation.
n = 2

using MLDatasets
data, labels = MNIST(split=:train)[:]

#Select the images.
mask = labels .== 3
train_ones = data[:,:,mask]
train = train_ones[:,:,1:n]

x = [i for i=1:28]
y = reverse(x)
f,ax = PyPlot.plt.subplots(2,5,sharey=true,sharex=true,figsize=(10,5))
PyPlot.plt.xticks([5,10,15,20,25])

for i = 1:10
    rand_pick = rand(1:size(train_ones)[3])
    ax[i].pcolormesh(x,y,transpose(train_ones[:,:,rand_pick]))
end

function single_pmf(data)
    #Takes a list of images and extracts the probability mass function.
    v = vec(data[:,:,1])
    v = v./cumsum(v)[length(v)]
    for im_k in 2:size(data)[3]
        image = data[:,:,im_k]
        arr = vec(image)
        v_size = size(arr)[1]
        v = hcat(v, arr./cumsum(arr)[length(arr)])
    end
    return v,size(v)[1]
end

function ms_distance(m,n)
    #Squared Euclidean distance calculation between the pixels.
    d = ones(m,m)
    coor_I = []
    for c_i in 1:n
        append!(coor_I,ones(Int,n).*c_i)
    end    
    coor_J = repeat(1:n,n)
    coor = hcat(coor_I,coor_J)
    for i in 1:m
        for j in 1:m
            d[i,j] = norm(coor[i,:]-coor[j,:]).^2
        end
    end
    return d
end

function reg_wasserstein_barycenter(M,data,lambda,relgap;entropyform=false)
    #Calculation of wasserstein barycenter by solving an entropy regularised minimization problem.
    #Direct mode model
    #M = direct_model(Mosek.Optimizer(MSK_DPAR_INTPNT_CO_TOL_REL_GAP=relgap))
    
    if length(size(data))==3
        K = size(data)[3]
    else
        K = 1
    end
    v,N = single_pmf(data)
    d = ms_distance(N,size(data)[2])
    
    if isnothing(lambda)
        lambda = 60/median(vec(d))
    end
    
    #Define indices
    M_i = 1:N
    M_j = 1:N
    M_k = 1:K

    #Adding variables
    M_pi = @variable(M, M_pi[i = M_i, j = M_j, k = M_k] >= 0.0)
    M_mu = @variable(M, M_mu[i = M_i] >= 0.0)

    
    #Adding constraints
    @constraint(M, c3_expr[k = M_k, j = M_j], sum(M_pi[:,j,k]) == v[j,k])
    @constraint(M, c2_expr[k = M_k, i = M_i], sum(M_pi[i,:,k]) == M_mu[i])


    if entropyform
        #YC:Clarabel
        #Auxiliary variable for the conic constraint
        M_aux = @variable(M,M_aux[j = M_j, k = M_k])

        #Adding conic constraint
        #YC: -1 is needed for it
        @constraint(M,cExp_cone[j=M_j, k=M_k],vcat(-M_aux[j,k],ones(N),M_pi[:,j,k]) in Clarabel.MOI.EntropyCone(2*N + 1))
        
        #Non-linear objective in the case of Regularized barycenters.
        W_obj = @objective(M, Min,(sum(d[i,j]*M_pi[i,j,k] for i=M_i,j=M_j,k=M_k) - 
                sum(M_aux[j,k] for j=M_j,k=M_k)/lambda)/K)
    else
        #Auxiliary variable for the conic constraint
        M_aux = @variable(M,M_aux[i = M_i, j = M_j, k = M_k])

        #Adding conic constraint
        @constraint(M,cExp_cone[i=M_i, j=M_j, k=M_k],[M_aux[i,j,k],M_pi[i,j,k],1] in MOI.ExponentialCone())
        
        #Non-linear objective in the case of Regularized barycenters.
        W_obj = @objective(M, Min,(sum(d[i,j]*M_pi[i,j,k] for i=M_i,j=M_j,k=M_k) - 
                sum(M_aux[i,j,k] for i=M_i,j=M_j,k=M_k)/lambda)/K)
    end

    return M,M_mu
end

function hypatia_reg_wasserstein_barycenter(M,data,lambda,relgap;entropyform=false)
    #Calculation of wasserstein barycenter by solving an entropy regularised minimization problem.
    #Direct mode model
    #M = direct_model(Mosek.Optimizer(MSK_DPAR_INTPNT_CO_TOL_REL_GAP=relgap))
    
    if length(size(data))==3
        K = size(data)[3]
    else
        K = 1
    end
    v,N = single_pmf(data)
    d = ms_distance(N,size(data)[2])
    
    if isnothing(lambda)
        lambda = 60/median(vec(d))
    end
    
    #Define indices
    M_i = 1:N
    M_j = 1:N
    M_k = 1:K

    #Adding variables
    M_pi = @variable(M, M_pi[i = M_i, j = M_j, k = M_k] >= 0.0)
    M_mu = @variable(M, M_mu[i = M_i] >= 0.0)

    
    #Adding constraints
    @constraint(M, c3_expr[k = M_k, j = M_j], sum(M_pi[:,j,k]) == v[j,k])
    @constraint(M, c2_expr[k = M_k, i = M_i], sum(M_pi[i,:,k]) == M_mu[i])

    #Auxiliary variable for the conic constraint
    M_aux = @variable(M,M_aux[j = M_j, k = M_k])

    #Adding conic constraint
    #YC: -1 is needed for it
    @constraint(M,cExp_cone[j=M_j, k=M_k],vcat(-M_aux[j,k],ones(N),M_pi[:,j,k]) in  MOI.RelativeEntropyCone(2*N + 1))
    
    #Non-linear objective in the case of Regularized barycenters.
    W_obj = @objective(M, Min,(sum(d[i,j]*M_pi[i,j,k] for i=M_i,j=M_j,k=M_k) - 
            sum(M_aux[j,k] for j=M_j,k=M_k)/lambda)/K)


    return M,M_mu
end


function run_regularised_model(model,data,lambda=nothing,relgap=1e-7)
    @time begin
        #Automatic mode model 
        model,M_mu = reg_wasserstein_barycenter(model,data,lambda,relgap)
        optimize!(model)
    end
    println("Solution status = ",termination_status(model))
    println("Primal objective value = ",objective_value(model))
    mu_level = value.(M_mu)
    # return mu_level
    return model
end

function clarabel_run_regularised_model(model,data,lambda=nothing,relgap=1e-7)
    @time begin
        #Automatic mode model 
        model,M_mu = reg_wasserstein_barycenter(model,data,lambda,relgap;entropyform=true)
        optimize!(model)
    end
    println("Solution status = ",termination_status(model))
    println("Primal objective value = ",objective_value(model))
    mu_level = value.(M_mu)
    # return mu_level
    return model
end

function show_barycenter(bary_center)
    bary_center = reshape(bary_center,(28,28))
    x = [i for i=1:28]
    y = reverse(x)
    PyPlot.plt.pcolormesh(x,y,transpose(bary_center))
    PyPlot.plt.title("Regularized Barycenter")
    PyPlot.plt.show()
end

# bary_center = run_regularised_model(train)
#Automatic mode model
model = Model(Clarabel.Optimizer)
set_optimizer_attribute(model,"equilibrate_enable",false)
set_optimizer_attribute(model,"max_iter", 1)
# set_optimizer_attribute(model,"neighborhood", 1e-6)
# set_optimizer_attribute(model,"tol_gap_abs", 1e-6)
# set_optimizer_attribute(model,"tol_gap_rel", 1e-6)
# set_optimizer_attribute(model,"tol_feas", 1e-6)
# set_optimizer_attribute(model,"tol_ktratio", 1e-4)
# set_optimizer_attribute(M,"MSK_IPAR_PRESOLVE_USE",0)
rimg = 11:18
model = clarabel_run_regularised_model(model,train[rimg,rimg,:])
solver = model.moi_backend.optimizer.model.optimizer.solver
println("******")

#Generate new problem
using SparseArrays
At = SparseMatrixCSC(solver.data.A')
n_dual = size(At,2)
b = solver.data.b 
q = solver.data.q
len_linear = solver.cones.rng_cones[2][end]    #number of linear constraints
len_entropy = solver.cones.rng_cones[3][end] - solver.cones.rng_cones[2][end]
n_cones = length(solver.cones.rng_cones)

model = Model(Clarabel.Optimizer)
@variable(model, x[1:n_dual])
@objective(model, Min, dot(b,x))
@constraint(model, q + At*x .== 0)
@constraint(model, x[1:Clarabel.degree(solver.cones[1])] >= 0)

start_idx = len_linear
for i in 1:(n_cones-2)
    @constraint(model, x[start_idx+1:start_idx+len_entropy] in Clarabel.MOI.DualEntropyCone(len_entropy))
    global start_idx += len_entropy
end

set_optimizer_attribute(model,"direct_solve_method",:qdldl)
set_optimizer_attribute(model,"tol_gap_abs", 1e-7)
set_optimizer_attribute(model,"tol_gap_rel", 1e-7)
set_optimizer_attribute(model,"tol_feas", 1e-7)
set_optimizer_attribute(model,"tol_ktratio", 1e-5)
set_optimizer_attribute(model,"min_terminate_step_length", 1e-3)
set_optimizer_attribute(model,"min_switch_step_length", 1e-3)
set_optimizer_attribute(model,"max_iter", 500)
optimize!(model)
solver = model.moi_backend.optimizer.model.optimizer.solver

modelm = Model(Mosek.Optimizer)
set_optimizer_attribute(modelm,"MSK_IPAR_PRESOLVE_USE",0)
set_optimizer_attribute(modelm,"MSK_IPAR_NUM_THREADS", 1)#thread number
set_optimizer_attribute(modelm,"MSK_DPAR_INTPNT_CO_TOL_DFEAS",1e-7)
set_optimizer_attribute(modelm,"MSK_DPAR_INTPNT_CO_TOL_MU_RED",1e-7)
set_optimizer_attribute(modelm,"MSK_DPAR_INTPNT_CO_TOL_PFEAS",1e-7)
set_optimizer_attribute(modelm,"MSK_DPAR_INTPNT_CO_TOL_REL_GAP",1e-7)

# using Hypatia
# modelH = Model(Hypatia.Optimizer)
# # set_optimizer_attribute(modelH,"syssolver", Hypatia.Solvers.SymIndefSparseSystemSolver{Float64}())
# set_optimizer_attribute(modelH,"tol_feas",1e-7)
# set_optimizer_attribute(modelH,"tol_rel_opt", 1e-7)
# set_optimizer_attribute(modelH,"tol_abs_opt", 1e-7)
# set_optimizer_attribute(modelH,"tol_inconsistent", 1e-4)
hypatia_reg_wasserstein_barycenter(modelm,train[rimg,rimg,:],nothing,1e-7)
optimize!(modelm)