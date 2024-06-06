using Hypatia

N = 10
d = 20

function random_state(d)
    x = randn(ComplexF64, (d, d))
    y = x * x'
    return LinearAlgebra.Hermitian(y / LinearAlgebra.tr(y))
end

ρ = [random_state(d) for i in 1:N]

# model = Model(Hypatia.Optimizer)
model = Model(Clarabel.Optimizer)
# set_silent(model)
E = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N-1]
E_N = LinearAlgebra.Hermitian(LinearAlgebra.I - sum(E))
@constraint(model, E_N in HermitianPSDCone())
push!(E, E_N)
@objective(model, Max, real(LinearAlgebra.dot(ρ, E)) / N)
optimize!(model)
# @assert is_solved_and_feasible(model)
solution_summary(model)