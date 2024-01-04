using ForwardDiff
using Random
using LinearAlgebra
using Clarabel

#power 
m1 = 5
dim2 = 1
rng = Random.MersenneTwister(1)
rng2 = Random.MersenneTwister(10)
rng3 = Random.MersenneTwister(100)
rng4 = Random.MersenneTwister(1000)
a = ones(m1) + rand(rng, m1)
normalize!(a,1)
# a =  solver.cones.cones[2].α
K = Clarabel.DualPowerMeanCone{Float64}(a)

#barrier for generalized power cone
foo(x) = -log(exp(sum(a[i]*log(x[i]) for i in 1:Clarabel.dim1(K))) - x[end]) - sum(log(x[i]) for i in 1:Clarabel.dim1(K))
f_phi(x) = exp(sum(a[i]*log(x[i]) for i in 1:Clarabel.dim1(K)))

function tensor(f, x)
    n = length(x)
    out = ForwardDiff.jacobian(y -> ForwardDiff.hessian(f, y), x)
    return reshape(out, n, n, n)
end

#Update# update gradient and Hessian at dual z = (u,w)
function update_dual_grad_H(
    K,
    z::AbstractVector{T}
) where {T}
    
    α = K.α
    p = K.data.p
    q = K.data.q        
    d1 = K.data.d1

    dim1 = K.data.d

    # ϕ = ∏_{i ∈ dim1}(ui)^(αi), ζ = φ - w
    ϕ = one(T)
    @inbounds for i = 1:dim1
        ϕ *= (z[i])^(α[i])
    end
    ζ = ϕ - z[end]
    @assert ζ > zero(T)

    # compute the gradient at z
    grad = K.data.grad
    τ = K.data.q           # τ shares memory with K.q
    @inbounds for i = 1:dim1
        τ[i] = α[i]/(z[i]*ζ)
        grad[i] = -τ[i]*ϕ - inv(z[i])
    end
    grad[end] = inv(ζ)

    # compute Hessian information at z 
    p0 = ϕ
    p1 = -one(T)
    q0 = sqrt(ζ*ϕ)

    # compute the diagonal d1,d2
    @inbounds for i = 1:dim1
        d1[i] = -grad[i]/z[i]
    end   
    K.data.d2 = zero(T)

    # compute p, q, r where τ shares memory with q
    p[1:dim1] .= p0*τ
    p[end] = p1/ζ

    q .*= q0      #τ is abandoned
    K.data.r[1] = zero(T)

    K.data.phi = ϕ 
    K.data.ζ = ζ
    # println("dual ζ is: ", ζ)
end


#Verification
n = Clarabel.dim(K)
x0 = rand(rng2,n)
x0[end] = f_phi(x0)-1e-6
# x0 = deepcopy(solver.variables.z.views[2])
dx = rand(rng3,n)
tx = rand(rng4,n)
ten = tensor(foo,x0)


#from Clarabel 
# result2 = similar(result)
K.data.z .= x0
update_dual_grad_H(K,x0)
# Clarabel.higher_correction!(K,result2,dx,dx)

#check Hessian
using BlockDiagonals
H = BlockDiagonal([diagm(0 => K.data.d1), diagm(0 => [K.data.d2])]) + K.data.p*K.data.p' - BlockDiagonal([K.data.q*K.data.q', K.data.r*K.data.r'])
Hc = ForwardDiff.hessian(foo,x0)
invH = inv(H)
# gradc = ForwardDiff.gradient(foo,x0)
# @assert(all(isapprox.(K.data.grad, gradc)))
# @assert(all(isapprox.(H,Hc)))

# #check primal gradient
# g = K.data.work
# Clarabel.gradient_primal!(K,g,x0) 
# @assert(isapprox(dot(g,x0),-Clarabel.degree(K)))

# #check tensor product 
# @assert(all(isapprox.(result,result2)))


####################################
# H^{-1}*x 
####################################
grad = K.data.grad
α = K.α

z = K.data.z
@assert(all(z .== x0))
z = K.data.z
dim1 = Clarabel.dim1(K)
u = @view z[1:dim1]

ϕ = K.data.phi
ζ = K.data.ζ

work = K.data.work_pp
@views worku = work[1:dim1]
@views xu = tx[1:dim1]

@. worku = ζ/(ζ+α*ϕ)        #inv(k1)
k2 = mapreduce((i,j)->i*i*j,+,α,worku)
k3 = 1 - ϕ/ζ*k2

c1 = 0
@inbounds for i = 1:dim1
    global c1 += xu[i]*α[i]*z[i]*worku[i]
end
c2 = ϕ/k3


#compute the H^{-1} explicitly
Hr = zeros(dim1+1,dim1+1)
for i in 1:dim1
    for j in 1:dim1
        Hr[i,j] = ϕ/ζ/k3*α[i]*α[j]*z[i]*z[j]*worku[i]*worku[j]

        if (i == j)
            Hr[i,j] += z[i]^2*worku[i]
        end
    end
end
for i in 1:dim1
    for j in 1:dim2
        Hr[i,dim1+j] = ϕ/k3*α[i]*z[i]*worku[i]
        Hr[dim1+j,i] = Hr[i,dim1+j]
    end
end

Hr[end,end] = ζ^2 + k2/k3*ϕ^2


function mul_Hinv!(
    K,
    y::AbstractVector{T},
    x::AbstractVector{T}
) where {T}
    α = K.α

    z = K.data.z
    dim1 = Clarabel.dim1(K)
    u = @view z[1:dim1]

    ϕ = K.data.phi
    ζ = K.data.ζ

    @views yu = y[1:dim1]
    work = K.data.work_pp
    @views worku = work[1:dim1]
    @views xu = x[1:dim1]

    @. worku = ζ/(ζ+α*ϕ)        #inv(k1)
    k2 = mapreduce((i,j)->i*i*j,+,α,worku)
    k3 = 1 - ϕ/ζ*k2

    c1 = zero(T)
    @inbounds for i = 1:dim1
        c1 += xu[i]*α[i]*u[i]*worku[i]
    end
    c2 = ϕ/k3
    c3 = c2*x[end]
    c4 = c2*c1/ζ
    @. yu = u*u*worku*xu + α*u*worku*c3 + c4*α*u*worku
    y[end] = (ζ^2+k2/k3*ϕ^2)*x[end] + c2*c1

    return nothing
end

function mul_Hs!(
    K,
    y::AbstractVector{T},
    x::AbstractVector{T}
) where {T}

    # Hs = μ*(D + pp' -qq' -rr')

    data = K.data

    rng1 = 1:Clarabel.dim1(K)
    rng2 = (Clarabel.dim1(K)+1):Clarabel.dim(K)

    coef_p = dot(data.p,x)
    @views coef_q = dot(data.q,x[rng1])
    @views coef_r = dot(data.r,x[rng2])
    
    @. y[rng1] = data.d1*x[rng1] - coef_q*K.data.q
    @. y[rng2] = data.d2*x[rng2] - coef_r*K.data.r

    @. y += coef_p*data.p
    @. y *= data.μ

end

#Verification for the error of multiplying Hs and Hs^{-1}
mul_Hinv!(K,K.data.work,tx)
K.data.μ = 1.0
mul_Hs!(K,K.data.work_pb,K.data.work)
norm(tx - K.data.work_pb,Inf)

# #Verify F′′(x)x =−F′(x)
# K.data.μ = 1.0
# mul_Hs!(K,K.data.work_pb,x0)
# norm(K.data.grad + K.data.work_pb,Inf)


# ####################################################
# # More general tensor product
# ####################################################
# derivative w.r.t. wi,wj,wk
-z[end]*ϕ*tmp[i]*tmp[j]*tmp[k]*(2*z[end]+ζ) - (i==j)*(z[end]*ϕ*tmp[i]*tmp[k]/u[i] + (i==k)*(2*ϕ*tmp[i]/u[i]^2 + 2/u[i]^3)) - z[end]*ϕ*tmp[i]*tmp[j]*((i==k)/u[i] + (j==k)/u[j]) - ten[i,j,k]
# tensor product
dim1 = Clarabel.dim1(K)
dim2 = Clarabel.dim2(K)
dim = Clarabel.dim(K)
du = @view dx[1:dim1]
dw = dx[end]
tu = @view tx[1:dim1]
tw = tx[end]
u = @view x0[1:dim1]
w = x0[end]

#workspace
τ = similar(u)
@. τ = a/u/ζ
# normd = similar(u)
# @. normd = du/u

Hwork = zeros(dim,dim)

#constants
τdu = dot(τ,du)
# wdw = dot(w,dw)
τtu = dot(τ,tu)
# wtw = dot(w,tw)
# dtw = dot(dw,tw)
c0 = 0.0
t0 = 0.0
s0 = 0.0
for i in 1:dim1
    global c0 += τ[i]*du[i]
    global t0 += τ[i]*tu[i]
    global s0 += τ[i]*du[i]*tu[i]/u[i]
end

#Verifying tensor product
for i in 1:dim1
    for j in 1:dim1
        Hwork[i,j] = ϕ*(2*ϕ/ζ-1)*τ[i]*τ[j]*dw - w*ϕ*τ[i]*τ[j]*((2*w+ζ)*c0+du[i]/u[i]+du[j]/u[j])

        if (i==j)
            Hwork[i,j] += ϕ/ζ*dw*τ[i]/u[i] - w*ϕ*τ[i]/u[i]*c0 - 2*ϕ*τ[i]*du[i]/u[i]^2 - 2*du[i]/u[i]^3
        end
    end
end
for i in 1:dim1
    Hwork[i,end] = -2*ϕ*τ[i]*dw/ζ^2 + ϕ*(2*ϕ/ζ-1)*τ[i]*c0 + ϕ*τ[i]*du[i]/(ζ*u[i])
    Hwork[end,i] = Hwork[i,end]
end

Hwork[end,end] = (2*dw/ζ - 2*ϕ*c0)/(ζ^2)

tmpm = zeros(dim,dim)
for i in 1:dim
    for j in 1:dim
        tmpm[i,j] = dot(ten[i,j,:],dx)
    end
end

#Final Verification
dir0 = tmpm*tx          #result from ForwardDiff

dir = similar(x0)
diru = @view dir[1:dim1]


#constant 
cu1 = -((2*w+ζ)*c0*t0 + s0)*w - 2*dw*tw/ζ^2 + (2*ϕ/ζ-1)*(tw*c0 + dw*t0)

@. diru  = ϕ*cu1*τ
for i in 1:dim1
    diru[i] += -2*du[i]*tu[i]*(1/u[i]+τ[i]*ϕ)/(u[i]*u[i]) + τ[i]*ϕ/u[i]*((tw*du[i] + dw*tu[i])/ζ - w*(tu[i]*c0+du[i]*t0))
end

dir[end] = 2/ζ^2*(dw*tw/ζ - ϕ*(c0*tw+t0*dw)) + ϕ*(2*ϕ/ζ-1)*c0*t0 + ϕ*s0/ζ
norm(dir0 - dir,Inf)

function higher_correction_mosek!(
    K,
    η::AbstractVector{T},
    ds::AbstractVector{T},
    dz::AbstractVector{T}
) where {T}

    #data.work = H^{-1}*ds
    mul_Hinv!(K,K.data.work,ds)
    # tensor product
    α = K.α
    dim1 = Clarabel.dim1(K)

    du = @view K.data.work[1:dim1]
    dw = K.data.work[end]
    tu = @view dz[1:dim1]
    tw = dz[end]

    z = K.data.z
    u = @view z[1:dim1]
    w = z[end]
    
    #workspace
    τ = similar(u)
    @. τ = α/u/ζ
    
    #constants
    c0 = dot(τ,du)
    t0 = dot(τ,tu)
    s0 = zero(T)
    @inbounds for i in 1:dim1
        s0 += τ[i]*du[i]*tu[i]/u[i]
    end

    #search direction
    diru = @view  η[1:dim1]

    #constant 
    cu1 = -((2*w+ζ)*c0*t0 + s0)*w - 2*dw*tw/ζ^2 + (2*ϕ/ζ-1)*(tw*c0 + dw*t0)

    @. diru  = ϕ*cu1*τ
    for i in 1:dim1
        diru[i] += -2*du[i]*tu[i]*(1/u[i]+τ[i]*ϕ)/(u[i]*u[i]) + τ[i]*ϕ/u[i]*((tw*du[i] + dw*tu[i])/ζ - w*(tu[i]*c0+du[i]*t0))
    end

    η[end] = 2/ζ^2*(dw*tw/ζ - ϕ*(c0*tw+t0*dw)) + ϕ*(2*ϕ/ζ-1)*c0*t0 + ϕ*s0/ζ

    @. η /= -2
    
    return nothing
end


mul_Hinv!(K,tx,dx)
dir0 = -tmpm*tx/2          #result from ForwardDiff
dir = similar(x0)
higher_correction_mosek!(K,dir,dx,dx)
norm(dir0-dir,Inf)
