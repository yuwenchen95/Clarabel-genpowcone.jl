using JuMP
include("signomial_test/JuMP.jl")
using Revise
using Clarabel
m = 50
n = 5      #YC: fails for n = 40,50

#For generating the example
model = clarabel_build(SignomialMinJuMP{Float64}(m,n))
set_optimizer_attribute(model,"equilibrate_enable",false)
set_optimizer_attribute(model,"tol_gap_abs", 1e-7)
set_optimizer_attribute(model,"tol_gap_rel", 1e-7)
set_optimizer_attribute(model,"tol_feas", 1e-7)
set_optimizer_attribute(model,"tol_ktratio", 1e-5)
set_optimizer_attribute(model,"min_terminate_step_length", 1e-3)
set_optimizer_attribute(model,"min_switch_step_length", 1e-3)
set_optimizer_attribute(model,"max_iter", 1)
set_optimizer_attribute(model,"verbose", false)
optimize!(model)
solver = model.moi_backend.optimizer.model.optimizer.solver


#Generate new problem
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
@constraint(model, x[1] >= 0)

start_idx = len_linear
for i in 1:(n_cones-2)
    @constraint(model, x[start_idx+1:start_idx+len_entropy] in Clarabel.MOI.DualEntropyCone(len_entropy))
    global start_idx += len_entropy
end

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
modelH = build(SignomialMinJuMP{Float64}(m,n),modelH)
optimize!(modelH)