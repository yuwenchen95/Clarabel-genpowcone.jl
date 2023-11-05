using ForwardDiff
using Random
using LinearAlgebra
# using Clarabel

#power 
m1 = 5
dim2 = 1
rng = Random.MersenneTwister(1)
rng2 = Random.MersenneTwister(10)
rng3 = Random.MersenneTwister(100)
a = rand(rng, m1)
normalize!(a,1)
K = Clarabel.GenPowerCone{Float64}(a,dim2)

#barrier for generalized power cone
foo(x) = -log(exp(sum(2*a[i]*log(x[i]) for i in 1:Clarabel.dim1(K))) - x[Clarabel.dim1(K)+1]^2) - sum((1-a[i])*log(x[i]) for i in 1:Clarabel.dim1(K))

function tensor(f, x)
    n = length(x)
    out = ForwardDiff.jacobian(y -> ForwardDiff.hessian(f, y), x)
    return reshape(out, n, n, n)
end

#Verification
n = Clarabel.dim(K)
x0 = rand(rng2,n)
dx = rand(rng3,n)
ten = tensor(foo,x0)

tmpm = zeros(n,n)
for i in 1:n
    for j in 1:n
        tmpm[i,j] = dot(ten[i,j,:],dx)
    end
end

result = tmpm*dx
result .*= -0.5

#from Clarabel 
result2 = similar(result)
K.data.z .= x0
Clarabel.update_dual_grad_H(K,x0)
Clarabel.higher_correction!(K,result2,dx,dx)

#check Hessian
using BlockDiagonals
H = BlockDiagonal([diagm(0 => K.data.d1), diagm(0 => K.data.d2*ones(Float64,Clarabel.dim2(K)))]) + K.data.p*K.data.p' - BlockDiagonal([K.data.q*K.data.q', K.data.r*K.data.r'])
Hc = ForwardDiff.hessian(foo,x0)
gradc = ForwardDiff.gradient(foo,x0)
@assert(all(isapprox.(K.data.grad, gradc)))
@assert(all(isapprox.(H,Hc)))

#check primal gradient
g = K.data.work
Clarabel.gradient_primal!(K,g,x0) 
@assert(isapprox(dot(g,x0),-Clarabel.degree(K)))

#check tensor product 
@assert(all(isapprox.(result,result2)))