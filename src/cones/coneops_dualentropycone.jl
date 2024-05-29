# ----------------------------------------------------
# Relative Entropy Cone
# ----------------------------------------------------

# degree of the cone is the dim of power vector + 1
dim(K::DualEntropyCone{T}) where {T} = K.dim
degree(K::DualEntropyCone{T}) where {T} = 2*K.d + 1
numel(K::DualEntropyCone{T}) where {T} = dim(K)

function is_sparse_expandable(::DualEntropyCone{T}) where{T}
    # we do not curently have a way of representing
    # this cone in non-expanded form
    return true
end

is_symmetric(::DualEntropyCone{T}) where {T} = false
allows_primal_dual_scaling(::DualEntropyCone{T}) where {T} = false

function shift_to_cone!(
    K::DualEntropyCone{T},
    z::AbstractVector{T}
) where{T}

    # We should never end up shifting to this cone, since 
    # asymmetric problems should always use unit_initialization!
    error("This function should never be reached.");
    # 
end

# # Generate an initial point following Hypatia
# # primal variable s0
# function get_central_ray_epirelentropy(w_dim::Int)
#     if w_dim <= 10
#         return central_rays_epirelentropy[w_dim, :]
#     end
#     # use nonlinear fit for higher dimensions
#     rtw_dim = sqrt(w_dim)
#     if w_dim <= 20
#         u = 1.2023 / rtw_dim - 0.015
#         v = 0.432 / rtw_dim + 1.0125
#         w = -0.3057 / rtw_dim + 0.972
#     else
#         u = 1.1513 / rtw_dim - 0.0069
#         v = 0.4873 / rtw_dim + 1.0008
#         w = -0.4247 / rtw_dim + 0.9961
#     end
#     return [u, v, w]
# end

# const central_rays_epirelentropy = [
#     0.827838399 1.290927714 0.805102005
#     0.708612491 1.256859155 0.818070438
#     0.622618845 1.231401008 0.829317079
#     0.558111266 1.211710888 0.838978357
#     0.508038611 1.196018952 0.847300431
#     0.468039614 1.183194753 0.854521307
#     0.435316653 1.172492397 0.860840992
#     0.408009282 1.163403374 0.866420017
#     0.38483862 1.155570329 0.871385499
#     0.364899122 1.148735192 0.875838068
# ]


function unit_initialization!(
    K::DualEntropyCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T}
 ) where{T}
 
     d = K.d
     dim  = K.dim
 
    # initialization from Hypatia
    (z[1], v, w) = get_central_ray_epirelentropy(dim)
    @views z[2:d+1] .= v
    @views z[d+2:end] .= w
    # find z such that s = -g*(z)

    gradient_dual!(K,K.grad,z)   
    @. s = -K.grad

    return nothing
 end

function set_identity_scaling!(
    K::DualEntropyCone{T},
) where {T}

    # We should never use identity scaling because 
    # we never want to allow symmetric initialization
    error("This function should never be reached.");
end

function update_scaling!(
    K::DualEntropyCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    # update both gradient and Hessian for function f*(z) at the point z
    _update_dual_grad_H(K,z)

    # update the scaling matrix Hs
    # YC: dual-scaling at present; we time μ to the diagonal here,
    # but this could be implemented elsewhere; μ is also used later 
    # when updating the off-diagonal terms of Hs; Recording μ is redundant 
    # for the dual scaling as it is a global parameter
    K.μ = μ

    # K.z .= z
    dim = K.dim
    @inbounds for i = 1:dim
        K.z[i] = z[i]
    end

    return is_scaling_success = true
end

# YC: may need to _allocate_kkt_Hsblocks()
# Stop it here
function Hs_is_diagonal(
    K::DualEntropyCone{T}
) where{T}
    return false
end

# return μH*(z) for generalized power cone
function get_Hs!(
    K::DualEntropyCone{T},
    Hsblock::AbstractVector{T}
) where {T}

    #NB: we are returning here the diagonal dd and offd
    μ = K.μ
    dim = K.dim
    Hsblock[1:dim]    .= μ*K.dd
    Hsblock[dim+1:end]    .= μ*K.offd

end

# compute the product y = Hs*x = μH(z)x
function mul_Hs!(
    K::DualEntropyCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    workz::AbstractVector{T}
) where {T}

    d = K.d
    p = K.p

    dotpx = dot(p,x)

    x1 = @view x[2:d+1]
    x2 = @view x[d+2:end]
    y1 = @view y[2:d+1]
    y2 = @view y[d+2:end]
    
    @. y = K.dd*x + dotpx*p
    @. y1 += K.offd*x2
    @. y2 += K.offd*x1
    
    @. y *= K.μ

end

function affine_ds!(
    K::DualEntropyCone{T},
    ds::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

    # @. x = y
    @inbounds for i = 1:K.dim
        ds[i] = s[i]
    end
end

function combined_ds_shift!(
    K::DualEntropyCone{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}

    η = K.work_pb
    # #3rd order correction requires input variables z
    if (all(step_z .== zero(T)))
        η .= zero(T)
    else
        higher_correction_mosek!(K,η,step_s,step_z)  
    end     

    @inbounds for i = 1:K.dim
        shift[i] = K.grad[i]*σμ + η[i]
    end

    return nothing
end

function Δs_from_Δz_offset!(
    K::DualEntropyCone{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    @inbounds for i = 1:K.dim
        out[i] = ds[i]
    end

    return nothing
end

#YC: an extension for _step_length_3d_cone, we could merge two together
function _step_length_n_cone(
    K::DualEntropyCone{T},
    dq::AbstractVector{T},
    q::AbstractVector{T},
    α_init::T,
    α_min::T,
    backtrack::T,
    is_in_cone_fcn::Function
) where {T}

    dim = K.dim
    wq = K.work
    α = α_init
    while true
        #@. wq = q + α*dq
        @inbounds for i = 1:dim
            wq[i] = q[i] + α*dq[i]
        end

        if is_in_cone_fcn(wq)
            break
        end
        if (α *= backtrack) < α_min
            α = zero(T)
            break
        end
    end
    return α
end

#return maximum allowable step length while remaining in the generalized power cone
function step_length(
    K::DualEntropyCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     settings::Settings{T},
     αmax::T,
) where {T}

    backtrack = settings.linesearch_backtrack_step
    αmin      = settings.min_terminate_step_length

    #switch the primal dual defined w.r.t. the entropy cone
    is_primal_feasible_fcn = s -> _is_dual_feasible_entropycone(s,K.d)
    is_dual_feasible_fcn   = s -> _is_primal_feasible_entropycone(s,K.d)

    αz = _step_length_n_cone(K, dz, z, αmax, αmin, backtrack, is_dual_feasible_fcn)
    αs = _step_length_n_cone(K, ds, s, αmax, αmin, backtrack, is_primal_feasible_fcn)

    # println(typeof(K), " with α", αz, " and ", αs)
    return (αz,αs)
end

function compute_barrier(
    K::DualEntropyCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T
) where {T}

    dim = K.dim

    barrier = zero(T)

    # we want to avoid allocating a vector for the intermediate 
    # sums, so the two barrier functions are written to accept 
    # both vectors and MVectors. 
    wq = K.work

    #primal barrier
    @inbounds for i = 1:dim
        wq[i] = s[i] + α*ds[i]
    end
    barrier += _barrier_primal(K, wq)

    #dual barrier
    @inbounds for i = 1:dim
        wq[i] = z[i] + α*dz[i]
    end
    barrier += _barrier_dual(K, wq)

    return barrier
end

function check_neighbourhood(
    K::DualEntropyCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T},  
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    μ::T,
    thr::T
) where {T}   
    d = K.d

    work = K.work
    work_pp = K.work_pp
    cur_z = K.work_pb
    @. cur_z = z+α*dz

    #overwrite g with the new gradient
    gradz = K.work_pp
    gradient_dual!(K,gradz,cur_z)
    @assert(isapprox(dot(gradz,cur_z),-degree(K)))

    #proximity check like Hypatia
    @. work = s+α*ds+μ*gradz

    mul_Hinv!(K,cur_z,work_pp,work)

    neighbourhood = dot(work,work_pp)

    if (neighbourhood < 0.9)
        return true
    end

    return false
end

# # ----------------------------------------------
# #  internal operations for dual relative entropy cones
# #
# # Primal relative entropy cone: u - ∑_{i ∈ [d]} w_i log(w_i/v_i) ≥ 0
# # Dual relative entropy cone: w_i ≥ u(log(u/v_i)-1), ∀ i ∈ [d]
# # We use the primal barrier function: 
# # f*(z) = -log(u - ∑_{i ∈ [d]}w_i*log(w_i/v_i)) -∑_{i ∈ [d]}(log(v_i) + log(w_i)):

function gradient_dual!(
    K::DualEntropyCone{T},
    grad::AbstractVector{T},
    z::AbstractVector{T}
) where {T}
    d = K.d
    v = @view z[2:d+1]
    w = @view z[d+2:end]

    # compute the gradient at z
    ζ = z[1]
    grad[1] = -one(T)
    @inbounds for i = 1:d
        c1 = w[i]/v[i]
        c2 = logsafe(c1)
        grad[d+1+i] = c2 + 1
        ζ -= w[i]*c2
        grad[i+1] = -c1
    end
    @. grad /= ζ

    @inbounds for i = 1:d
        grad[i+1] += - inv(v[i])
        grad[d+1+i] += -inv(w[i])
    end
end

# update gradient and Hessian at dual z = (u,w)
function _update_dual_grad_H(
    K::DualEntropyCone{T},
    z::AbstractVector{T}
) where {T}

    d = K.d
    offd = K.offd
    dd = K.dd
    p = K.p
    v = @view z[2:d+1]
    w = @view z[d+2:end]
    σ = @view p[2:d+1]
    τ = @view p[d+2:end]

    ζ = z[1]
    p[1] = one(T)
    @inbounds for i = 1:d
        c1 = w[i]/v[i]
        c2 = logsafe(c1)
        τ[i] = -c2 - 1
        ζ -= w[i]*c2
        σ[i] = c1
    end
    @. p /= ζ

    # compute the gradient at z
    grad = K.grad

    grad[1] = -p[1]
    @inbounds for i = 1:d
        grad[i+1] = -σ[i] - inv(v[i])
        grad[d+1+i] = -inv(w[i]) - τ[i]
    end
    
    # compute Hessian information at z 
    dd[1] = zero(T)
    @inbounds for i = 1:d
        offd[i] = -inv(ζ*v[i])
        dd[i+1] = σ[i]/v[i] + inv(v[i]*v[i])
        dd[d+1+i] = inv(ζ*w[i]) + inv(w[i]*w[i])
    end    
end

####################################
# 3rd-order correction
####################################
#H^{-1}*x 
function mul_Hinv!(
    K::DualEntropyCone{T},
    z::AbstractVector{T},
    y::AbstractVector{T},
    x::AbstractVector{T}
) where {T}
    d = K.d

    x1 = @view x[2:d+1]
    x2 = @view x[d+2:end]

    Hiuv = K.Hiuv
    Hiuw = K.Hiuw
    Hivw = K.Hivw
    Hiww = K.Hiww
    Hivv = K.Hivv

    ζ = z[1]
    @inbounds for i = 1:d
        ζ -= z[i+d+1]*logsafe(z[i+d+1]/z[i+1])
    end

    HiuHu = zero(T)
    @inbounds for i in 1:K.d
        wi = z[d+1+i]
        vi = z[1+i]
        lwvi = logsafe(wi/vi)
        zwi = ζ + wi
        z2wi = zwi + wi
        wz2wi = wi / z2wi
        vz2wi = vi / z2wi
        uvvi = wi * (wi * lwvi - ζ)
        uwwi = wi * (ζ + lwvi * zwi) * wz2wi
        HiuHu += wz2wi * uvvi - uwwi * (lwvi + 1)
        Hiuv[i] = vz2wi * uvvi
        Hiuw[i] = uwwi
        Hivw[i] = wi * vi * wz2wi
        Hiww[i] = wi * zwi * wz2wi
        Hivv[i] = vi * zwi * vz2wi
    end
    Hiuu = abs2(ζ) - HiuHu

    #compute neighborhood value
    y[1] = Hiuu*x[1] + dot(Hiuv,x1) + dot(Hiuw,x2)
    @. y[2:d+1] = Hiuv*x[1] + Hivv*x1 + Hivw*x2
    @. y[d+2:end] = Hiuw*x[1] + Hiww*x2 + Hivw*x1

    return ζ

end

function higher_correction_mosek!(
    K::DualEntropyCone{T},
    η::AbstractVector{T},
    ds::AbstractVector{T},
    dz::AbstractVector{T}
) where {T}
    d = K.d 
    p = K.p
    z = K.z
    δ = K.work
    t = dz

    #data.work = H^{-1}*ds
    ζ = mul_Hinv!(K,z,δ,ds)
    # tensor product
    v = @view z[2:d+1]
    w = @view z[d+2:end]
    σ = @view p[2:d+1]
    τ = @view p[d+2:end]
    δv = @view δ[2:d+1]
    δw = @view δ[d+2:end]
    tv = @view t[2:d+1]
    tw = @view t[d+2:end]
    
    dotpδ = dot(p,δ)
    dotpt = dot(p,t)

    η[1] = -2/ζ*dotpδ*dotpt
    @inbounds for i in 1:d
        η[1] += (-σ[i]*δv[i]*tv[i]/v[i] + (δw[i]*tv[i]/v[i] + δv[i]*tw[i]/v[i]-δw[i]*tw[i]/w[i])/ζ)/ζ 
    end

    @inbounds for i in 1:d
        η[i+1] = -2*σ[i]*dotpδ*dotpt + δw[i]/(v[i]*ζ)*dotpt - σ[i]*δv[i]/v[i]*dotpt-tv[i]*(σ[i]/v[i]*dotpδ+2*δv[i]*(σ[i]+1/v[i])/(v[i]^2) -δw[i]/(v[i]^2*ζ)) + tw[i]*(dotpδ+δv[i]/v[i])/(v[i]*ζ)
        η[i+d+1] = -2*τ[i]*dotpδ*dotpt + δv[i]/(v[i]*ζ)*dotpt - δw[i]/(w[i]*ζ)*dotpt + tv[i]*(dotpδ+δv[i]/v[i])/(v[i]*ζ) - tw[i]*(dotpδ/(w[i]*ζ) + δw[i]*(1/(w[i]^2*ζ)+2/(w[i]^3)))

        @inbounds for j in 1:d
            η[i+1] += σ[i]*(-σ[j]*δv[j]*tv[j]/v[j] + ((δw[j]*tv[j] + δv[j]*tw[j])/v[j] - δw[j]*tw[j]/w[j])/ζ) 
            η[i+d+1] += τ[i]*(-σ[j]*δv[j]*tv[j]/v[j] + (δw[j]*tv[j]/v[j]-δw[j]*tw[j]/w[j]+δv[j]*tw[j]/v[j])/ζ)
        end
    end

    @. η /= -2
    
    return nothing
end