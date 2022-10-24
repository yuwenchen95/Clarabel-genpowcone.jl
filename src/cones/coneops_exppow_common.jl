# find the maximum step length α≥0 so that
# q + α*dq stays in an exponential or power
# cone, or their respective dual cones.
#
# NB: Not for use as a general checking
# function because cone lengths are hardcoded
# to R^3 for faster execution.  

function _step_length_3d_cone(
    K::Union{PowerCone{T},ExponentialCone{T}},
    dq::AbstractVector{T},
    q::AbstractVector{T},
    α_init::T,
    α_min::T,
    backtrack::T,
    is_in_cone_fcn::Function
) where {T}

    wq = similar(K.grad)
    α = α_init
    while true
        #@. wq = q + α*dq
        @inbounds for i = 1:3
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

#YC: an extension for _step_length_3d_cone, we could merge two together
function _step_length_n_cone(
    K::GenPowerCone{T},
    dq::AbstractVector{T},
    q::AbstractVector{T},
    α_init::T,
    α_min::T,
    backtrack::T,
    is_in_cone_fcn::Function
) where {T}

    dim = K.dim
    wq = similar(K.grad)
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

#-------------------------------------
# primal-dual scaling
#-------------------------------------

# Implementation sketch
# 1) only need to replace μH by W^TW, where
#    W^TW is the primal-dual scaling matrix 
#    generated by BFGS, i.e. W^T W*[z,\tilde z] = [s,\tile s]
#   \tilde z = -f'(s), \tilde s = - f*'(z)


# update the scaling matrix Hs
function _update_Hs(
    K::Union{PowerCone{T},ExponentialCone{T}},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

# Choose the scaling strategy
    if(scaling_strategy == Dual::ScalingStrategy)
        # Dual scaling: Hs = μ*H
        _use_dual_scaling(K,μ)
    else
        # Primal-dual scaling
        _use_primal_dual_scaling(K,s,z)
    end 

end


# use the dual scaling strategy
function _use_dual_scaling(
    K::Union{PowerCone{T},ExponentialCone{T}},
    μ::T
) where {T}
    @inbounds for i = 1:9
        K.Hs[i] = μ*K.H_dual[i]
    end
end


# use the primal-dual scaling strategy
function _use_primal_dual_scaling(
    K::Union{PowerCone{T},ExponentialCone{T}},
    s::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    (Hs,H_dual) = (K.Hs,K.H_dual)

    st = K.grad
    δs = similar(st)
    tmp = similar(st) #shared for δz, tmp, axis_z

    # compute zt,st,μt locally
    # NB: zt,st have different sign convention wrt Mosek paper
    zt = _gradient_primal(K,s)
    dot_sz = dot(z,s)
    μ = dot_sz/3
    μt = dot(zt,st)/3

    δz = tmp
    @inbounds for i = 1:3
        δs[i] = s[i] + μ*st[i]
        δz[i] = z[i] + μ*zt[i]
    end    
    dot_δsz = dot(δs,δz)

    de1 = μ*μt-1
    de2 = dot(zt,H_dual,zt) - 3*μt*μt

    # use the primal-dual scaling
    if (abs(de1) > sqrt(eps(T))      # too close to central path
        && abs(de2) > eps(T) && dot_sz > 0 && dot_δsz > 0)  # for numerical stability
       
        # compute t
        # tmp = μt*st - H_dual*zt
        @inbounds for i = 1:3
            tmp[i] = μt*st[i] - H_dual[i,1]*zt[1] - H_dual[i,2]*zt[2] - H_dual[i,3]*zt[3]
        end

        # Hs as a workspace
        copyto!(Hs,H_dual)
        @inbounds for i = 1:3
            @inbounds for j = 1:3
                Hs[i,j] -= st[i]*st[j]/3 + tmp[i]*tmp[j]/de2
            end
        end

        t = μ*norm(Hs)  #Frobenius norm

        # @assert dot_sz > 0
        # @assert dot_δsz > 0
        @assert t > 0

        # generate the remaining axis
        # axis_z = cross(z,zt)
        axis_z = tmp
        axis_z[1] = z[2]*zt[3] - z[3]*zt[2]
        axis_z[2] = z[3]*zt[1] - z[1]*zt[3]
        axis_z[3] = z[1]*zt[2] - z[2]*zt[1]
        normalize!(axis_z)

        # Hs = s*s'/⟨s,z⟩ + δs*δs'/⟨δs,δz⟩ + t*axis_z*axis_z'
        @inbounds for i = 1:3
            @inbounds for j = i:3
                Hs[i,j] = s[i]*s[j]/dot_sz + δs[i]*δs[j]/dot_δsz + t*axis_z[i]*axis_z[j]
            end
        end
        # symmetrize matrix
        Hs[2,1] = Hs[1,2]
        Hs[3,1] = Hs[1,3]
        Hs[3,2] = Hs[2,3]

        return nothing
    else
        # Hs = μ*H_dual when s,z are on the central path
        _use_dual_scaling(K,μ)

        return nothing
    end
    
end


#------------------------------------------------------------
# Numerical sub-routines for primal barrier computation
#------------------------------------------------------------
function _newton_raphson_onesided(x0::T,f0::Function,f1::Function) where {T}

    #implements NR method from a starting point assumed to be to the 
    #left of the true value.   Once a negative step is encountered 
    #this function will halt regardless of the calculated correction.

    x = x0
    iter = 0

    while iter < 100

        iter += 1
        dfdx  =  f1(x)  
        dx    = -f0(x)/dfdx

        if (dx < eps(T)) ||
            (abs(dx/x) < sqrt(eps(T))) ||
            (abs(dfdx) < eps(T))
            break
        end
        x += dx
    end
    return x
end