# ----------------------------------------------------
# Power Mean Cone
# ----------------------------------------------------

#dimensions of the subcomponents
dim1(K::PowerMeanCone{T}) where {T} = length(K.α)
dim2(K::PowerMeanCone{T}) where {T} = 1

# degree of the cone is the dim of power vector + 1
dim(K::PowerMeanCone{T}) where {T} = K.dim
degree(K::PowerMeanCone{T}) where {T} = K.dim
numel(K::PowerMeanCone{T}) where {T} = dim(K)

function is_sparse_expandable(::PowerMeanCone{T}) where{T}
    # we do not curently have a way of representing
    # this cone in non-expanded form
    return true
end

is_symmetric(::PowerMeanCone{T}) where {T} = false
allows_primal_dual_scaling(::PowerMeanCone{T}) where {T} = false


function shift_to_cone!(
    K::PowerMeanCone{T},
    z::AbstractVector{T}
) where{T}

    # We should never end up shifting to this cone, since 
    # asymmetric problems should always use unit_initialization!
    error("This function should never be reached.");
    # 
end

function get_central_ray_powermean(α::AbstractVector{T},s::AbstractVector{T}) where {T <: Real}
    d = length(α)
    # predict w given α and d
    w = view(s,1:d)
    if d == 1
        w .= 1.306563
    elseif d == 2
        @. w = 1.0049885 + 0.2986276 * α
    elseif d <= 5
        @. w = 1.0040142949 - 0.0004885108 * d + 0.3016645951 * α
    elseif d <= 20
        @. w = 1.001168 - 4.547017e-05 * d + 3.032880e-01 * α
    elseif d <= 100
        @. w = 1.000069 - 5.469926e-07 * d + 3.074084e-01 * α
    else
        @. w = 1 + 3.086535e-01 * α
    end
    # get u in closed form from w
    p = exp(sum(α_i * log(w_i) for (α_i, w_i) in zip(α, w)))
    s[end] = p - p / d * sum(α_i / (abs2(w_i) - 1) for (α_i, w_i) in zip(α, w))

end

function unit_initialization!(
    K::PowerMeanCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T}
 ) where{T}
 
    α = K.α
 
    # init s as in Hypatia
    get_central_ray_powermean(α,s)
 
    #set @. z = -g(s)
    gradient_primal!(K,z,s) 
    z .*= -one(T)  
 
    return nothing
 end

function set_identity_scaling!(
    K::PowerMeanCone{T},
) where {T}

    # We should never use identity scaling because 
    # we never want to allow symmetric initialization
    error("This function should never be reached.");
end

function update_scaling!(
    K::PowerMeanCone{T},
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
    K.data.μ = μ

    # K.z .= z
    dim = K.dim
    @inbounds for i = 1:dim
        K.data.z[i] = z[i]
    end

    return is_scaling_success = true
end

function Hs_is_diagonal(
    K::PowerMeanCone{T}
) where{T}
    return true
end

# return μH*(z) for power mean cone
function get_Hs!(
    K::PowerMeanCone{T},
    Hsblock::AbstractVector{T}
) where {T}

    #NB: we are returning here the diagonal D = [d1; d2] block from the
    #sparse representation of W^TW, but not the
    #extra 3 entries at the bottom right of the block.
    #The ConicVector for s and z (and its views) don't
    #know anything about the 3 extra sparsifying entries
    dim1 = K.data.d
    μ = K.data.μ
    @. Hsblock[1:dim1]    = μ*K.data.d1
    Hsblock[end] = μ*K.data.d2

end

# compute the product y = Hs*x = μH(z)x
function mul_Hs!(
    K::PowerMeanCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    workz::AbstractVector{T}
) where {T}

    # Hs = μ*(D + pp' -qq' -rr')
    d1 = K.data.d1
    d2 = K.data.d2
    dim1 = K.data.d

    coef_p = dot(K.data.p,x)
    coef_q = dot(K.data.q,x[1:dim1])

    x1 = @view x[1:dim1]
    y1 = @view y[1:dim1]
    
    @. y = coef_p*K.data.p
    @. y1 += d1*x1 - coef_q*K.data.q
    r = K.data.r[1]
    y[end] += d2*x[end] - x[end]*r*r
    
    @. y *= K.data.μ

end

function affine_ds!(
    K::PowerMeanCone{T},
    ds::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

    # @. x = y
    @inbounds for i = 1:K.dim
        ds[i] = s[i]
    end
end

function combined_ds_shift!(
    K::PowerMeanCone{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}
    
    #YC: No 3rd order correction at present

    # #3rd order correction requires input variables z
    # η = _higher_correction!(K,step_s,step_z)     

    @inbounds for i = 1:K.dim
        shift[i] = K.data.grad[i]*σμ # - η[i]
    end

    return nothing
end

function Δs_from_Δz_offset!(
    K::PowerMeanCone{T},
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


function _step_length_n_cone(
    K::PowerMeanCone{T},
    dq::AbstractVector{T},
    q::AbstractVector{T},
    α_init::T,
    α_min::T,
    backtrack::T,
    is_in_cone_fcn::Function
) where {T}

    dim = K.dim
    wq = K.data.work
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

#return maximum allowable step length while remaining in the power mean cone
function step_length(
    K::PowerMeanCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     settings::Settings{T},
     αmax::T,
) where {T}

    backtrack = settings.linesearch_backtrack_step
    αmin      = settings.min_terminate_step_length

    #need functions as closures to capture the power K.α
    #and use the same backtrack mechanism as the expcone
    is_primal_feasible_fcn = s -> _is_primal_feasible_powmeancone(s,K.α,K.data.d)
    is_dual_feasible_fcn   = s -> _is_dual_feasible_powmeancone(s,K.α,K.data.d)

    αz = _step_length_n_cone(K, dz, z, αmax, αmin, backtrack, is_dual_feasible_fcn)
    αs = _step_length_n_cone(K, ds, s, αmax, αmin, backtrack, is_primal_feasible_fcn)

    return (αz,αs)
end

function compute_barrier(
    K::PowerMeanCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T
) where {T}

    dim = K.dim
    wq = K.data.work

    barrier = zero(T)

    #primal barrier
    @inbounds for i = 1:dim
        wq[i] = s[i] + α*ds[i]
    end
    barrier += barrier_primal(K, wq)

    #dual barrier
    @inbounds for i = 1:dim
        wq[i] = z[i] + α*dz[i]
    end
    barrier += barrier_dual(K, wq)

    return barrier
end

function check_neighbourhood(
    K::PowerMeanCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T},  
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    μ::T,
    thr::T
) where {T}   

    work = K.data.work
    @. work = s+α*ds
    g = K.data.work_pb
    @. g = z+α*dz
    # cur_μ = dot(work,g)
    cur_μ = μ

    #overwrite g with the new gradient
    gradz = K.data.work_pp
    gradient_dual!(K,gradz,g)
    # @assert(isapprox(dot(gradz,g),-degree(K)))
    gradient_primal!(K,g,work) 
    
    μt = dot(gradz,g)    
    neighbourhood = degree(K)/(μt*cur_μ)
    # println("neighbourhood is ", neighbourhood)
    if (neighbourhood < thr)
        return false
    end

    return true
end

# ----------------------------------------------
#  internal operations for power mean cones
#
# Primal power mean cone: ∏_{i ∈ [d1]}s[i]^{α[i]} ≥ s[end], s ≥ 0
# Dual power mean cone: ∏_{i ∈ [d1]}(z[i]/α[i])^{α[i]} + z[end] ≥ 0, z[1:d1] ≥ 0, z[end] ≤ 0
# We use the dual barrier function: 
# f*(z) = -log((∏_{i ∈ [d1]}(z[i]/α[i])^{α[i]} + z[end]) - ∑_{i ∈ [d1]} (1-α[i])*log(z[i]) - log(-z[end]):
# Evaluates the gradient of the dual power mean cone ∇f*(z) at z, 
# and stores the result at g


@inline function barrier_dual(
    K::PowerMeanCone{T},
    z::Union{AbstractVector{T}, NTuple{N,T}}
) where {N<:Integer,T}

    # Dual barrier
    dim1 = K.data.d
    α = K.α

    res = zero(T)
    @inbounds for i = 1:dim1
        res += α[i]*logsafe(z[i]/α[i])
    end
    res = exp(res) + z[end]
    barrier = -logsafe(res) 
    @inbounds for i = 1:dim1
        barrier -= (one(T)-α[i])*logsafe(z[i])
    end
    barrier -= logsafe(-z[end])

    return barrier

end

@inline function barrier_primal(
    K::PowerMeanCone{T},
    s::Union{AbstractVector{T}, NTuple{N,T}}
) where {N<:Integer,T}

    # Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    # NB: ⟨s,g(s)⟩ = -(dim1+1) = - ν

    minus_g = K.data.work_pb
    gradient_primal!(K,minus_g,s)     #compute g(s)
    minus_g .*= -one(T)

    #YC: need to consider the memory issue later
    return -barrier_dual(K,minus_g) #- degree(K)
end



# Returns true if s is primal feasible
function _is_primal_feasible_powmeancone(
    s::AbstractVector{T},
    α::AbstractVector{T},
    dim1::Int
) where {T}

    if (all(s[1:dim1] .> zero(T)))
        res = zero(T)
        @inbounds for i = 1:dim1
            res += α[i]*logsafe(s[i])
        end
        res = exp(res) - s[end]
        if res > zero(T)
            return true
        end
    end

    return false
end

# Returns true if z is dual feasible
function _is_dual_feasible_powmeancone(
    z::AbstractVector{T},
    α::AbstractVector{T},
    dim1::Int
) where {T}

    if (all(z[1:dim1] .> zero(T)) && z[end] < zero(T))
        res = zero(T)
        @inbounds for i = 1:dim1
            res += α[i]*(logsafe(z[i]) - logsafe(α[i]))
        end
        res = exp(res) + z[end]
        if res > zero(T)
            return true
        end
    end
    
    return false
end

# Compute the primal gradient of f(s) at s
# solve it by the Newton-Raphson method
function gradient_primal!(
    K::PowerMeanCone{T},
    g::Union{AbstractVector{T}, NTuple{N,T}},
    s::Union{AbstractVector{T}, NTuple{N,T}},
) where {N<:Integer,T}

    α = K.α
    dim1 = K.data.d

    # obtain g0 from the Newton-Raphson method
    p = @view s[1:dim1]
    gp = @view g[1:dim1]

    if s[end] > zero(T)
        g0 = _newton_raphson_powmeancone_pos(dim1,p,s[end],α)
        g[end] = 1/g0
    else
        invϕ = _newton_raphson_powmeancone_nonpos(dim1,p,s[end],α)
        ϕ = inv(invϕ)
        invr = inv(s[end])
        g[end] = (ϕ-sqrt(ϕ*ϕ + 4*invr*invr))/2 - invr
    end

    @. gp = -(1+α+α*s[end]*g[end])/p

    @assert dot(g,s) ≈ -degree(K)

end

function gradient_dual!(
    K::PowerMeanCone{T},
    grad::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    α = K.α

    dim1 = K.data.d

    # ϕ = ∏_{i ∈ dim1}(ui/αi)^(αi), ζ = φ + w
    ϕ = one(T)
    @inbounds for i = 1:dim1
        ϕ *= (z[i]/α[i])^(α[i])
    end
    ζ = ϕ + z[end]
    @assert ζ > zero(T)

    # compute the gradient at z
    ϕdivζ = ϕ/ζ
    @inbounds for i = 1:dim1
        grad[i] = -α[i]/z[i]*ϕdivζ - (1-α[i])/z[i]
    end
    grad[end] = - inv(ζ) - inv(z[end])

end
# Newton-Raphson method:
# solve a one-dimensional equation f(x) = 0
# x(k+1) = x(k) - f(x(k))/f'(x(k))
# When we initialize with x0 = 0 for the power mean cone, 
# the Newton-Raphson method converges quadratically

function _newton_raphson_powmeancone_pos(
    dim::Int,
    p::AbstractVector{T},
    r::T,
    α::AbstractVector{T}
) where {T}

    # init point x0 = 0
    x0 = zero(T)

    # function for f(x) = 0
    function f0(x)
        f0 = -logsafe(one(T) + 1/(r+x));
        @inbounds for i = 1:dim
            f0 += α[i]*logsafe(((1+α[i])*x/α[i]+r)/p[i])
        end

        return f0
    end

    # first derivative
    function f1(x)
        f1 = one(T)/((r+x)*(r+x+1));
        @inbounds for i = 1:dim
            f1 += α[i]/(x + α[i]*r/(1+α[i]))
        end

        return f1
    end
    
    return _newton_raphson_onesided(x0,f0,f1)
end

function _newton_raphson_powmeancone_nonpos(
    dim::Int,
    p::AbstractVector{T},
    r::T,
    α::AbstractVector{T}
) where {T}

    # init point x0 = 0
    x0 = zero(T);
    @inbounds for i = 1:dim
        ti = α[i]*p[i]
        x0 += α[i]*logsafe(ti/(1+α[i]))
    end
    x0 = exp(x0)



    # function for f(x) = 0
    function f0(x)
        f0 = zero(T);
        t = (r + sqrt(r*r + 4*x*x))/2;
        @inbounds for i = 1:dim
            f0 += α[i]*logsafe(x/(α[i]*p[i]) + t/p[i])
        end

        return f0
    end

    # first derivative
    function f1(x)
        f1 = zero(T);
        t0 = sqrt(r*r + 4*x*x);
        t1 = 2*inv(t0);
        t2 = (t0 + r)/2;
        @inbounds for i = 1:dim
            f1 += α[i]*((1 + α[i]*x*t1)/(x + α[i]*t2))
        end

        return f1
    end
    
    return _newton_raphson_onesided(x0,f0,f1)
end

# update gradient and Hessian at dual z = (u,w)
function _update_dual_grad_H(
    K::PowerMeanCone{T},
    z::AbstractVector{T}
) where {T}
    
    α = K.α
    p = K.data.p
    q = K.data.q        
    d1 = K.data.d1

    dim1 = K.data.d

    # ϕ = ∏_{i ∈ dim1}(ui/αi)^(αi), ζ = φ + w
    ϕ = one(T)
    @inbounds for i = 1:dim1
        ϕ *= (z[i]/α[i])^(α[i])
    end
    ζ = ϕ + z[end]
    @assert ζ > zero(T)

    # compute the gradient at z
    grad = K.data.grad
    τ = K.data.q           # τ shares memory with K.q
    @inbounds for i = 1:dim1
        τ[i] = α[i]/(z[i]*ζ)
        grad[i] = -τ[i]*ϕ - (1-α[i])/z[i]
    end
    grad[end] = - inv(ζ) - inv(z[end])

    # compute Hessian information at z 
    p0 = ϕ
    p1 = one(T)
    q0 = sqrt(ζ*ϕ)

    # compute the diagonal d1,d2
    @inbounds for i = 1:dim1
        d1[i] = -grad[i]/z[i]
    end   
    K.data.d2 = 1/(z[end]^2) + 1

    # compute p, q, r where τ shares memory with q
    p[1:dim1] .= p0*τ
    p[end] = p1/ζ

    q .*= q0      #τ is abandoned
    K.data.r[1] = one(T)

    K.data.phi = ϕ 
    K.data.ζ = ζ
    # println("dual ζ is: ", ζ)
end
