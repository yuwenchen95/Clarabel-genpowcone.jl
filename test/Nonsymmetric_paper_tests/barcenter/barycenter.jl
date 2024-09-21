using LinearAlgebra
using Plots
pyplot()

using JuMP

using Mosek
using MosekTools
using Clarabel

using Statistics

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

