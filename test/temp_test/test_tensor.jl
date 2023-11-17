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
rng4 = Random.MersenneTwister(1000)
a = ones(m1) + rand(rng, m1)
normalize!(a,1)
# a =  solver.cones.cones[2].α
K = Clarabel.GenPowerCone{Float64}(a,dim2)

#barrier for generalized power cone
foo(x) = -log(exp(sum(2*a[i]*log(x[i]) for i in 1:Clarabel.dim1(K))) - x[Clarabel.dim1(K)+1]^2) - sum((1-a[i])*log(x[i]) for i in 1:Clarabel.dim1(K))
f_phi(x) = exp(sum(2*a[i]*log(x[i]) for i in 1:Clarabel.dim1(K)))

function tensor(f, x)
    n = length(x)
    out = ForwardDiff.jacobian(y -> ForwardDiff.hessian(f, y), x)
    return reshape(out, n, n, n)
end

#Verification
n = Clarabel.dim(K)
x0 = rand(rng2,n)
x0[end] = sqrt(f_phi(x0)-1e-10)
# x0 = deepcopy(solver.variables.z.views[2])
dx = rand(rng3,n)
tx = rand(rng4,n)
# ten = tensor(foo,x0)

# tmpm = zeros(n,n)
# for i in 1:n
#     for j in 1:n
#         tmpm[i,j] = dot(ten[i,j,:],dx)
#     end
# end

# result = tmpm*dx
# result .*= -0.5

#from Clarabel 
result2 = similar(result)
K.data.z .= x0
Clarabel.update_dual_grad_H(K,x0)
# Clarabel.higher_correction!(K,result2,dx,dx)

#check Hessian
using BlockDiagonals
H = BlockDiagonal([diagm(0 => K.data.d1), diagm(0 => K.data.d2*ones(Float64,Clarabel.dim2(K)))]) + K.data.p*K.data.p' - BlockDiagonal([K.data.q*K.data.q', K.data.r*K.data.r'])
# Hc = ForwardDiff.hessian(foo,x0)
invH = inv(H)
# gradc = ForwardDiff.gradient(foo,x0)
# @assert(all(isapprox.(K.data.grad, gradc)))
# @assert(all(isapprox.(H,Hc)))

#check primal gradient
g = K.data.work
Clarabel.gradient_primal!(K,g,x0) 
@assert(isapprox(dot(g,x0),-Clarabel.degree(K)))

# #check tensor product 
# @assert(all(isapprox.(result,result2)))


####################################
# H^{-1}*x 
####################################
grad = K.data.grad
z = x0
dim1 = Clarabel.dim1(K)
u = @view x0[1:dim1]
w = @view x0[dim1+1:end]
w2 = Clarabel.sumsq(w)
ϕ = f_phi(u)
ζ = ϕ - w2
d = Clarabel.dim(K)

@views gradu = grad[1:dim1]
@views gradw = grad[dim1+1:end]

y = K.data.work
x = dx
@views yu = y[1:dim1]
@views yw = y[dim1+1:end]
@views xu = x[1:dim1]
@views xw = x[dim1+1:end]

k1 = 0.0
c1 = 0.0
for i in 1:dim1
    global k1 += a[i]*a[i]/(grad[i]*z[i])
    global c1 += a[i]*xu[i]/grad[i]
end

k2 = -(1 + w2/ϕ) - 4*k1*w2/ζ
c2 = dot(w,xw)

c11 = -4*c1*w2/(k2*ζ) + 2*c2/k2
@. yu = -u/gradu*xu + c11*a/gradu
@. yw = ζ/2*xw + ((ζ/ϕ+4*k1)*c2/k2 + 2*c1/k2)*w

# result = similar(x0)
# Hr = mul_Hinv!(K,result,dx)
# norm(result0 - result,Inf)
# τ = 2*a./u 
# coef = ϕ/ζ
# U = [coef*τ τ; -2/ζ*w zeros(dim2,1)]
# A = BlockDiagonal([diagm(0 => -gradu./u), 2/ζ*diagm(0=> ones(dim2))])
# work = U'*inv(A)*U
# work[1,1] += 1
# work[2,2] += -1/coef
# work = inv(work)

#compute the H^{-1} explicitly
Hr = zeros(d,d)
for i in 1:dim1
    for j in 1:dim1
        Hr[i,j] = -4/k2*(w2/ζ)*(a[i]*a[j])/(gradu[i]*gradu[j])

        if (i == j)
            Hr[i,j] -= z[i]/gradu[i]
        end
    end
end
for i in 1:dim1
    for j in 1:dim2
        Hr[i,dim1+j] = 2/k2*a[i]/gradu[i]*w[j]
        Hr[dim1+j,i] = Hr[i,dim1+j]
    end
end
for i in 1:dim2
    for j in 1:dim2
        Hr[i+dim1,j+dim1] = (ζ/ϕ+4*k1)/k2*w[i]*w[j]

        if (i == j)
            Hr[i+dim1,j+dim1] += ζ/2
        end
    end
end

function mul_Hinv!(
    K,
    y::AbstractVector{T},
    x::AbstractVector{T}
) where {T}
    α = K.α

    grad = K.data.grad
    z = K.data.z
    dim1 = Clarabel.dim1(K)
    u = @view z[1:dim1]
    w = @view z[dim1+1:end]
    w2 = K.data.w2
    ϕ = K.data.phi
    ζ = K.data.ζ

    @views gradu = grad[1:dim1]
    @views gradw = grad[dim1+1:end]

    @views yu = y[1:dim1]
    @views yw = y[dim1+1:end]
    @views xu = x[1:dim1]
    @views xw = x[dim1+1:end]

    k1 = zero(T)
    c1 = zero(T)
    @inbounds for i in 1:dim1
        k1 += α[i]*α[i]/(grad[i]*z[i])
        c1 += α[i]*xu[i]/grad[i]
    end

    k2 = -(1 + w2/ϕ) - 4*k1*w2/ζ
    c2 = dot(w,xw)

    c11 = -4*c1*w2/(k2*ζ) + 2*c2/k2
    @. yu = -u/gradu*xu + c11*α/gradu
    @. yw = ζ/2*xw + ((ζ/ϕ+4*k1)*c2/k2 + 2*c1/k2)*w
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


# ####################################################
# # More general tensor product
# ####################################################
# # tensor product
# dim1 = Clarabel.dim1(K)
# dim2 = Clarabel.dim2(K)
# dim = Clarabel.dim(K)
# du = @view dx[1:dim1]
# dw = @view dx[dim1+1:end]
# tu = @view tx[1:dim1]
# tw = @view tx[dim1+1:end]
# u = @view x0[1:dim1]
# w = @view x0[dim1+1:end]

# #workspace
# τ = similar(u)
# @. τ = 2*a/u
# normd = similar(u)
# @. normd = du/u

# Hwork = zeros(dim,dim)

# #constants
# ϕ = f_phi(u)
# ζ = ϕ - Clarabel.sumsq(w)
# ϕdζ = ϕ/ζ
# τdu = dot(τ,du)
# wdw = dot(w,dw)
# τtu = dot(τ,tu)
# wtw = dot(w,tw)
# dtw = dot(dw,tw)
# c0 = 0.0
# for i in 1:dim1
#     global c0 += τ[i]*normd[i]*tu[i]
# end

# #Verifying tensor product
# for i in 1:dim1
#     for j in 1:dim1
#         Hwork[i,j] = τ[i]*τ[j]*ϕdζ*(1-ϕdζ)*((2*ϕdζ-1)*τdu + normd[i] + normd[j]) + 2*τ[i]*τ[j]*ϕdζ/ζ*(2*ϕdζ-1)*wdw

#         if (i==j)
#             Hwork[i,j] += τ[i]*ϕdζ/u[i]*(1-ϕdζ)*τdu -2*du[i]*((1-a[i])/u[i]+τ[i]*ϕdζ)/(u[i]*u[i]) + 2*τ[i]*ϕdζ/(ζ*u[i])*wdw
#         end
#     end
# end
# for i in 1:dim1
#     for j in 1:dim2
#         Hwork[i,j+dim1] = 2*τ[i]*ϕdζ/ζ*(w[j]*((2*ϕdζ-1)*τdu + normd[i]) - dw[j]) - 8*w[j]*τ[i]*ϕdζ/(ζ*ζ)*wdw
#     end
# end
# for i in 1:dim2
#     for j in 1:dim1
#         Hwork[i+dim1,j] = 2*τ[j]*ϕdζ/ζ*(w[i]*((2*ϕdζ-1)*τdu + normd[j]) - dw[i]) - 8*w[i]*τ[j]*ϕdζ/(ζ*ζ)*wdw
#     end
# end
# for i in 1:dim2
#     for j in 1:dim2
#         Hwork[i+dim1,j+dim1] = 8*w[i]*w[j]/(ζ^3)*(2*wdw-ϕ*τdu) + 4*(w[i]*dw[j]+w[j]*dw[i])/(ζ^2)

#         if (i==j)
#             Hwork[i+dim1,j+dim1] += 2/(ζ^2)*(2*wdw - ϕ*τdu)
#         end
#     end
# end

# #Final Verification
# dir0 = tmpm*tx          #result from ForwardDiff

# dir = similar(x0)
# diru = @view dir[1:dim1]
# dirw = @view dir[dim1+1:end]

# #constant 
# cu1 = ϕdζ*(2*ϕdζ-1)*((1-ϕdζ)*τdu*τtu + 2/ζ*(wdw*τtu + τdu*wtw)) - 2*ϕdζ/ζ*(4*wdw*wtw/ζ + dtw) + ϕdζ*(1-ϕdζ)*c0
# cud = 2*ϕdζ*wtw/ζ + ϕdζ*(1-ϕdζ)*τtu
# cut = 2*ϕdζ*wdw/ζ + ϕdζ*(1-ϕdζ)*τdu

# for i in 1:dim1
#     diru[i] = cu1 + (cud*du[i] + cut*tu[i])/u[i]
# end
# @. diru  *= τ
# for i in 1:dim1
#     diru[i] -= 2*du[i]*tu[i]*((1-a[i])/u[i]+τ[i]*ϕdζ)/(u[i]*u[i])
# end

# cw1 = 2*ϕ*((2ϕdζ-1)*τdu*τtu + c0 - 4/ζ*(wdw*τtu+τdu*wtw)) + (16*wdw*wtw/ζ + 4*dtw)
# cwd = 4*wtw - 2*ϕ*τtu
# cwt = 4*wdw - 2*ϕ*τdu
# for i in 1:dim2
#     dirw[i] = (cw1*w[i] + (cwd*dw[i]+cwt*tw[i]))/(ζ*ζ) 
# end


# function higher_correction_mosek!(
#     K,
#     η::AbstractVector{T},
#     ds::AbstractVector{T},
#     dz::AbstractVector{T}
# ) where {T}

#     #data.work = H^{-1}*ds
#     mul_Hinv!(K,K.data.work,ds)
#     # tensor product
#     α = K.α
#     dim1 = Clarabel.dim1(K)
#     dim2 = Clarabel.dim2(K)

#     du = @view K.data.work[1:dim1]
#     dw = @view K.data.work[dim1+1:end]
#     tu = @view dz[1:dim1]
#     tw = @view dz[dim1+1:end]

#     z = K.data.z
#     u = @view z[1:dim1]
#     w = @view z[dim1+1:end]

#     #workspace
#     τ = similar(u)
#     @. τ = 2*α/u
#     normd = similar(u)
#     @. normd = du/u

#     #constants
#     ϕ = K.data.phi
#     ζ = K.data.ζ
#     ϕdζ = ϕ/ζ
#     τdu = dot(τ,du)
#     wdw = dot(w,dw)
#     τtu = dot(τ,tu)
#     wtw = dot(w,tw)
#     dtw = dot(dw,tw)
#     c0 = zero(T)
#     @inbounds for i in 1:dim1
#         c0 += τ[i]*normd[i]*tu[i]
#     end

#     #search direction
#     diru = @view  η[1:dim1]
#     dirw = @view  η[dim1+1:end]

#     #constant 
#     cu1 = ϕdζ*(2*ϕdζ-1)*((1-ϕdζ)*τdu*τtu + 2/ζ*(wdw*τtu + τdu*wtw)) - 2*ϕdζ/ζ*(4*wdw*wtw/ζ + dtw) + ϕdζ*(1-ϕdζ)*c0
#     cud = 2*ϕdζ*wtw/ζ + ϕdζ*(1-ϕdζ)*τtu
#     cut = 2*ϕdζ*wdw/ζ + ϕdζ*(1-ϕdζ)*τdu

#     @inbounds for i in 1:dim1
#         diru[i] = cu1 + (cud*du[i] + cut*tu[i])/u[i]
#     end
#     @. diru  *= τ
#     @inbounds for i in 1:dim1
#         diru[i] -= 2*du[i]*tu[i]*((1-α[i])/u[i]+τ[i]*ϕdζ)/(u[i]*u[i])
#     end
    
#     cw1 = 2*ϕ*((2ϕdζ-1)*τdu*τtu + c0 - 4/ζ*(wdw*τtu+τdu*wtw)) + (16*wdw*wtw/ζ + 4*dtw)
#     cwd = 4*wtw - 2*ϕ*τtu
#     cwt = 4*wdw - 2*ϕ*τdu
#     @inbounds for i in 1:dim2
#         dirw[i] = (cw1*w[i] + (cwd*dw[i]+cwt*tw[i]))/(ζ*ζ) 
#     end

#     @. η /= -2
    
#     return nothing
# end


# mul_Hinv!(K,tx,dx)
# dir0 = -tmpm*tx/2          #result from ForwardDiff
# dir = similar(x0)
# higher_correction_mosek!(K,dir,dx,dx)
# norm(dir0-dir,Inf)
