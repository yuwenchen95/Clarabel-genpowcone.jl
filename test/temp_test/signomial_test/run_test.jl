using JuMP
include("JuMP.jl")
using Revise
using Clarabel
m = 40
n = 3      #YC: fails for n = 40,50
model = clarabel_build(SignomialMinJuMP{Float64}(m,n))
set_optimizer_attribute(model,"up_barrier", 1.0)
set_optimizer_attribute(model,"low_barrier", 0.5)
set_optimizer_attribute(model,"min_terminate_step_length", 1e-3)
set_optimizer_attribute(model,"min_switch_step_length", 1e-3)
set_optimizer_attribute(model,"cratio",1.0)
set_optimizer_attribute(model,"max_iter", 2000)
optimize!(model)
solver = model.moi_backend.optimizer.model.optimizer.solver


# using Hypatia
# modelH = Model(Hypatia.Optimizer)
# modelH = build(SignomialMinJuMP{Float64}(m,n),modelH)
# optimize!(modelH)

# # using ECOS
# modelC = Model(Clarabel.Optimizer)
# # set_optimizer_attribute(modelC,"min_switch_step_length", 1e-3)
# # set_optimizer_attribute(modelC,"static_regularization_constant", 1e-9)     
# # set_optimizer_attribute(modelC,"iterative_refinement_abstol", 1e-15)
# # set_optimizer_attribute(modelC,"iterative_refinement_reltol", 1e-15)
# # set_optimizer_attribute(modelC,"MSK_IPAR_PRESOLVE_USE",false)
# # set_optimizer_attribute(modelC,"max_iter", 5000)
# modelC = build(SignomialMinJuMP{Float64}(m,n),modelC)
# optimize!(modelC)