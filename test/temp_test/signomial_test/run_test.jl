using JuMP
include("JuMP.jl")

m = 10
n = 10
model = clarabel_build(SignomialMinJuMP{Float64}(m,n))
set_optimizer_attribute(model,"up_barrier", 1.0)
set_optimizer_attribute(model,"low_barrier", 0.5)
set_optimizer_attribute(model,"min_terminate_step_length", 1e-3)
set_optimizer_attribute(model,"cratio",0.95)
set_optimizer_attribute(model,"max_iter", 5000)
optimize!(model)
solver = model.moi_backend.optimizer.model.optimizer.solver


using Hypatia
modelH = Model(Hypatia.Optimizer)
modelH = build(SignomialMinJuMP{Float64}(m,n),modelH)
optimize!(modelH)

modelC = Model(Mosek.Optimizer)
modelC = build(SignomialMinJuMP{Float64}(m,n),modelC)
optimize!(modelC)