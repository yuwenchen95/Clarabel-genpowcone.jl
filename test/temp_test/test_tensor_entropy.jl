using Random
using ForwardDiff
using LinearAlgebra

d = 3
n = 2*d+1

T = Float64
H = ones(T,n,n)

#data point 
z = [1.435316653, 1.272492397, 1.172492397, 1.372492397, 0.860840992, 0.760840992, 0.960840992]
z[3] = 1.2
z[5] = 1
rng = Random.MersenneTwister(1)
rng2 = Random.MersenneTwister(10)
δ = rand(rng,n)
t = rand(rng2,n)
p = zeros(T,n)

v = @view z[2:d+1]
w = @view z[d+2:end]
σ = @view p[2:d+1]
τ = @view p[d+2:end]
δv = @view δ[2:d+1]
δw = @view δ[d+2:end]
tv = @view t[2:d+1]
tw = @view t[d+2:end]

ζ = z[1]
p[1] = one(T)
@inbounds for i = 1:d
    global c1 = w[i]/v[i]
    c2 = log(c1)
    global τ[i] = -c2 - 1
    global ζ -= w[i]*c2
    global σ[i] = c1
end
@. p /= ζ

dotpδ = dot(p,δ)
dotpt = dot(p,t)
dotσδv = dot(σ,δv)
dotτδw = dot(τ,δw)
dotδ1 = sum(δw)
for i in 1:n
    if i == 1
        H[1,1] = -2/(ζ^2)*dotpδ
        for j in 1:d
            H[1,j+1] = -2*σ[j]/(ζ)*dotpδ - σ[j]*δv[j]/(v[j]*ζ) + δw[j]/(v[j]*ζ^2)
            H[1,d+1+j] = -2τ[j]/ζ*dotpδ + δv[j]/(v[j]*ζ^2) - δw[j]/(w[j]*ζ^2)
        end
    end

    for j in 1:d 
        H[1+j,1] = H[1,1+j]
        H[d+1+j,1] = H[1,d+1+j]

        for k in 1:d 
            H[1+j,1+k] = -2*σ[j]*σ[k]*dotpδ + σ[j]*δw[k]/(v[k]*ζ) + σ[k]*δw[j]/(v[j]*ζ) - σ[k]*σ[j]*δv[j]/v[j] - σ[k]*σ[j]*δv[k]/v[k]
            H[d+1+j,d+1+k] = -2τ[j]*τ[k]*dotpδ + τ[j]*δv[k]/(v[k]*ζ) + τ[k]*δv[j]/(v[j]*ζ) - τ[j]δw[k]/(w[k]*ζ) - τ[k]*δw[j]/(w[j]*ζ)
            H[1+j,d+1+k] = -2σ[j]*τ[k]*dotpδ - τ[k]*σ[j]*δv[j]/v[j] + σ[j]*δv[k]/(v[k]*ζ) - σ[j]*δw[k]/(w[k]*ζ) + τ[k]*δw[j]/(v[j]*ζ)
            
            if k == j
                H[1+j,1+k] += -(σ[j]*δ[1]/(v[j]*ζ) + σ[j]/v[j]*dotσδv + 2*δv[j]*(σ[j]/(v[j]^2) + 1/(v[j]^3)) + σ[j]/v[j]*dotτδw) + δw[j]/(v[j]^2*ζ) 
                H[d+1+j,d+1+k] += -dotτδw/(w[j]*ζ) - δw[j]*(1/(w[j]^2*ζ) + 2/(w[j]^3)) - δ[1]/(w[j]*ζ^2) - dotσδv/(w[j]*ζ)
                H[1+j,d+1+k] += δ[1]/(v[j]*ζ*ζ) + dotσδv/(v[j]*ζ) + δv[j]/(v[j]^2*ζ) + dotτδw/(v[j]*ζ) 
            end

            H[d+1+k,1+j] = H[1+j,d+1+k]
        end
    end
end

#From simulation
foo(x) = -log(x[1] - x[5]*log(x[5]/x[2]) - x[6]*log(x[6]/x[3]) - x[7]*log(x[7]/x[4])) - log(x[2]) - log(x[3]) - log(x[4]) - log(x[5])- log(x[6]) - log(x[7])

function tensor(f, x)
    n = length(x)
    out = ForwardDiff.jacobian(y -> ForwardDiff.hessian(f, y), x)
    return reshape(out, n, n, n)
end

ten = tensor(foo,z)

tmpm = zeros(n,n)
for i in 1:n
    for j in 1:n
        tmpm[i,j] = dot(ten[i,j,:],δ)
    end
end

#H^3[δ,t]
tmpc = tmpm*t
correction = ones(T,n)
correction[1] = -2/ζ*dotpδ*dotpt
for i in 1:d
    correction[1] += (-σ[i]*δv[i]*tv[i]/v[i] + (δw[i]*tv[i]/v[i] + δv[i]*tw[i]/v[i]-δw[i]*tw[i]/w[i])/ζ)/ζ 
end

for i in 1:d
    correction[i+1] = -2*σ[i]*dotpδ*dotpt + δw[i]/(v[i]*ζ)*dotpt - σ[i]*δv[i]/v[i]*dotpt-tv[i]*(σ[i]/v[i]*dotpδ+2*δv[i]*(σ[i]+1/v[i])/(v[i]^2) -δw[i]/(v[i]^2*ζ)) + tw[i]*(dotpδ+δv[i]/v[i])/(v[i]*ζ)
    correction[i+d+1] = -2*τ[i]*dotpδ*dotpt + δv[i]/(v[i]*ζ)*dotpt - δw[i]/(w[i]*ζ)*dotpt + tv[i]*(dotpδ+δv[i]/v[i])/(v[i]*ζ) - tw[i]*(dotpδ/(w[i]*ζ) + δw[i]*(1/(w[i]^2*ζ)+2/(w[i]^3)))

    for j in 1:d
        correction[i+1] += σ[i]*(-σ[j]*δv[j]*tv[j]/v[j] + ((δw[j]*tv[j] + δv[j]*tw[j])/v[j] - δw[j]*tw[j]/w[j])/ζ) 
        correction[i+d+1] += τ[i]*(-σ[j]*δv[j]*tv[j]/v[j] + (δw[j]*tv[j]/v[j]-δw[j]*tw[j]/w[j]+δv[j]*tw[j]/v[j])/ζ)
    end
end


# function tensorijk(i,j,k,σ,τ,v,w,ζ)
#     tmp = 0
#     if (i==j)
#         if (j==k)
#             tmp = -3*σ[i]^2/v[i] - 2*σ[i]/(v[i]^2) - 2/(v[i]^3)
#         else
#             tmp = -σ[i]*σ[k]/v[i]
#         end
#     else
#         if (j==k)
#             tmp = -σ[i]*σ[k]/v[j]
#         else
#             tmp = 0
#         end
#     end

#     return tmp -2*σ[i]*σ[j]*σ[k]
# end

# function tensorijk(i,j,k,σ,τ,v,w,ζ)
#     tmp = 0
#     if (i==j)
#         if (j==k)
#             tmp = 1/(v[i]^2*ζ) - τ[i]*σ[i]/v[i] + 2*σ[i]/(v[i]*ζ)
#         else
#             tmp = -τ[k]*σ[i]/v[i]
#         end
#     else
#         if (j==k)
#             tmp = σ[i]/(v[j]*ζ)
#         elseif (i==k)
#             tmp = σ[j]/(v[i]*ζ)
#         else
#             tmp = 0
#         end
#     end

#     return tmp -2*σ[i]*σ[j]*τ[k]
# end

# function tensorijk(i,j,k,σ,τ,v,w,ζ)
#     tmp = 0
#     if (i==j)
#         if (j==k)
#             tmp = 2τ[i]/(v[i]*ζ) - 1/(v[i]*ζ^2)
#         else
#             tmp = τ[k]/(v[i]*ζ)
#         end
#     else
#         if (j==k)
#             tmp = -σ[i]/(w[j]*ζ)
#         elseif (k==i)
#             tmp = τ[j]/(v[i]*ζ)
#         else
#             tmp = 0
#         end
#     end

#     return tmp -2*σ[i]*τ[j]*τ[k]
# end
