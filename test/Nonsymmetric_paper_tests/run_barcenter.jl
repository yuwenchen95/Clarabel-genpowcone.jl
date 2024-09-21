include("barcenter/barycenter.jl")

#Define the number of images for the barycenter calculation.
num_img = 2

using MLDatasets
data, labels = MNIST(split=:train)[:]

#Select the images.
mask = labels .== 3
train_ones = data[:,:,mask]
train = train_ones[:,:,1:num_img]

x = [i for i=1:28]
y = reverse(x)
f,ax = PyPlot.plt.subplots(2,5,sharey=true,sharex=true,figsize=(10,5))
PyPlot.plt.xticks([5,10,15,20,25])

for i = 1:10
    rand_pick = rand(1:size(train_ones)[3])
    ax[i].pcolormesh(x,y,transpose(train_ones[:,:,rand_pick]))
end

#Generate the primal model
# For (m,n) in the paper, we set rimg[1] = 14-floor((n-1)/2) and rimg[2] = 14+floor(n/2)
# We also set m = num_img*n in default for this case, i.e. choose num_img = 2 images from the data set for the Barcenter problem
n = 6
rimg = Int(14 - floor((n-1)/2)):Int(14 + floor(n/2))   
model = Model(Clarabel.Optimizer)
set_optimizer_attribute(model,"equilibrate_enable",false)
set_optimizer_attribute(model,"verbose",false)
set_optimizer_attribute(model,"max_iter", 1)    
model = clarabel_run_regularised_model(model,train[rimg,rimg,:])
solver = model.moi_backend.optimizer.model.optimizer.solver
println("******")

#Solve the equivalent dual problem
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

using Hypatia
modelH = Model(Hypatia.Optimizer)
# set_optimizer_attribute(modelH,"syssolver", Hypatia.Solvers.SymIndefSparseSystemSolver{Float64}())
set_optimizer_attribute(modelH,"tol_feas",1e-7)
set_optimizer_attribute(modelH,"tol_rel_opt", 1e-7)
set_optimizer_attribute(modelH,"tol_abs_opt", 1e-7)
set_optimizer_attribute(modelH,"tol_inconsistent", 1e-4)
hypatia_reg_wasserstein_barycenter(modelH,train[rimg,rimg,:],nothing,1e-7)
optimize!(modelH)