# ----------------------------------------------------
# Generalized Power Cone
# ----------------------------------------------------

#dimensions of the subcomponents
dim1(K::DualGenPowerCone{T}) where {T} = length(K.α)
dim2(K::DualGenPowerCone{T}) where {T} = K.dim2

# degree of the cone is the dim of power vector + 1
dim(K::DualGenPowerCone{T}) where {T} = dim1(K) + dim2(K)
degree(K::DualGenPowerCone{T}) where {T} = dim1(K) + 1
numel(K::DualGenPowerCone{T}) where {T} = dim(K)

function is_sparse_expandable(::DualGenPowerCone{T}) where{T}
    # we do not curently have a way of representing
    # this cone in non-expanded form
    return true
end

is_symmetric(::DualGenPowerCone{T}) where {T} = false
allows_primal_dual_scaling(::DualGenPowerCone{T}) where {T} = false

function shift_to_cone!(
    K::DualGenPowerCone{T},
    z::AbstractVector{T}
) where{T}

    # We should never end up shifting to this cone, since 
    # asymmetric problems should always use unit initialization
    error("This function should never be reached.");
    # 
end

function unit_initialization!(
    K::DualGenPowerCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T}
 ) where{T}
 
    # init u[i] = √(1+αi), i ∈ [dim1(K)]
    @inbounds for i = 1:dim1(K)
        s[i] = sqrt(one(T)+K.α[i])
    end
    # init w = 0
    s[dim1(K)+1:end] .= zero(T)
 
     #@. z = s
     @inbounds for i = 1:dim(K)
         z[i] = s[i]
     end
 
    return nothing
 end

function set_identity_scaling!(
    K::DualGenPowerCone{T},
) where {T}

    # We should never use identity scaling because 
    # we never want to allow symmetric initialization
    error("This function should never be reached.");
end

function update_scaling!(
    K::DualGenPowerCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    # update both gradient and Hessian for function f*(z) at the point z
    update_dual_grad_H(K,z)
    K.data.μ = μ

    # K.z .= z
    @inbounds for i in eachindex(z)
        K.data.z[i] = z[i]
    end

    # monitor_residuals(K,z,s)

    return is_scaling_success = true
end

function Hs_is_diagonal(
    K::DualGenPowerCone{T}
) where{T}
    return true
end

# return μH*(z) for generalized power cone
function get_Hs!(
    K::DualGenPowerCone{T},
    Hsblock::AbstractVector{T}
) where {T}

    #NB: we are returning here the diagonal D = [d1; d2] block from the
    #sparse representation of W^TW, but not the
    #extra 3 entries at the bottom right of the block.
    #The ConicVector for s and z (and its views) don't
    #know anything about the 3 extra sparsifying entries
    dim1 = Clarabel.dim1(K)
    data = K.data
    
    @. Hsblock[1:dim1]     = data.μ*data.d1
    @. Hsblock[dim1+1:end] = data.μ*data.d2

end

# compute the product y = Hs*x = μH(z)x
function mul_Hs!(
    K::DualGenPowerCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    workz::AbstractVector{T}
) where {T}

    # Hs = μ*(D + pp' -qq' -rr')

    data = K.data

    rng1 = 1:dim1(K)
    rng2 = (dim1(K)+1):dim(K)

    coef_p = dot(data.p,x)
    @views coef_q = dot(data.q,x[rng1])
    @views coef_r = dot(data.r,x[rng2])
    
    @. y[rng1] = data.d1*x[rng1] - coef_q*K.data.q
    @. y[rng2] = data.d2*x[rng2] - coef_r*K.data.r

    @. y += coef_p*data.p
    @. y *= data.μ

end

function affine_ds!(
    K::DualGenPowerCone{T},
    ds::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

    # @. x = y
    @inbounds for i = 1:dim(K)
        ds[i] = s[i]
    end
end

function combined_ds_shift!(
    K::DualGenPowerCone{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}
    
    # #3rd order correction requires input variables z
    # and an allocated vector for the correction η
    η = K.data.work_pb

    if (all(step_z .== zero(T)))
        η .= zero(T)
    else
        higher_correction_mosek!(K,η,step_s,step_z)  
    end  
    # println("Higher order norm: ", norm(η,Inf))
    @inbounds for i = 1:Clarabel.dim(K)
        shift[i] = K.data.grad[i]*σμ + η[i]
    end

    return nothing
end

function Δs_from_Δz_offset!(
    K::DualGenPowerCone{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    @inbounds for i = 1:dim(K)
        out[i] = ds[i]
    end

    return nothing
end

#return maximum allowable step length while remaining in the generalized power cone
function step_length(
    K::DualGenPowerCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     settings::Settings{T},
     αmax::T,
) where {T}

    step = settings.linesearch_backtrack_step
    αmin = settings.min_terminate_step_length
    work = K.data.work

    is_prim_feasible_fcn = s -> is_primal_feasible(K,s)
    is_dual_feasible_fcn = s -> is_dual_feasible(K,s)

    αz = backtrack_search(K, dz, z, αmax, αmin, step, is_dual_feasible_fcn,work)
    αs = backtrack_search(K, ds, s, αmax, αmin, step, is_prim_feasible_fcn,work)

    return (αz,αs)
end

function compute_barrier(
    K::DualGenPowerCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T
) where {T}

    barrier = zero(T)
    work = K.data.work
    # tmp = zero(T)

    #dual barrier
    @inbounds for i = 1:dim(K)
        work[i] = s[i] + α*ds[i]
    end
    barrier += barrier_primal(K, work)

    #primal barrier
    @inbounds for i = 1:dim(K)
        work[i] = z[i] + α*dz[i]
        # tmp += (s[i] + α*ds[i])*(z[i] + α*dz[i])
    end
    barrier += barrier_dual(K, work)

    # μi = tmp/degree(K)
    # tmp = degree(K)*logsafe(μi) + barrier
    # println("μi is ", μi, "  barrier is ", tmp)
    # @assert(tmp > -1e-5)

    return barrier
end

function check_neighbourhood(
    K::DualGenPowerCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T},  
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    μ::T,
    thr::T
) where {T}   

    # #Hypatia neighbourhood
    # work = K.data.work
    # @. work = z+α*dz
    # @. K.data.z = work

    # #Update Hessian and gradient information
    # update_dual_grad_H(K,work)
    # work2 = K.data.work_pb
    # @. work2 = s + α*ds + μ*K.data.grad
    
    # mul_Hinv!(K,work,work2)
    # centrality = sqrt(dot(work,work2))

    # #Outside of the neighbourhood
    # if centrality > μ
    #     return false
    # end

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
#  internal operations for generalized power cones
#
# Primal generalized power cone: ∏_{i ∈ [d1]}s[i]^{α[i]} ≥ ||s[d1+1:end]||, s[1:d1] ≥ 0
# Dual generalized power cone: ∏_{i ∈ [d1]}(z[i]/α[i])^{α[i]} ≥ ||z[d1+1:end]||, z[1:d1] ≥ 0
# We use the dual barrier function: 
# f*(z) = -log((∏_{i ∈ [d1]}(z[i]/α[i])^{2*α[i]} - ||z[d1+1:end]||^2) - ∑_{i ∈ [d1]} (1-α[i])*log(z[i]):
# Evaluates the gradient of the dual generalized power cone ∇f*(z) at z, 
# and stores the result at g


# Returns true if z is dual feasible
function is_dual_feasible(
    K::DualGenPowerCone{T},
    s::AbstractVector{T},
) where {T}

    dim1 = Clarabel.dim1(K)
    α = K.α

    if (all(s[1:dim1] .> zero(T)))
        @views res = mapreduce((i,j)->2*i*logsafe(j),+,α,s[1:dim1])
        res = exp(res) - sumsq(@view s[dim1+1:end])
        if res > zero(T)
            return true
        end
    end

    return false
end

# Returns true if s is primal feasible
function is_primal_feasible(
    K::DualGenPowerCone{T},
    z::AbstractVector{T},
) where {T}

    dim1 = Clarabel.dim1(K)
    α = K.α

    if (all(z[1:dim1] .> zero(T)))
        res = zero(T)
        @inbounds for i = 1:dim1
            res += 2*α[i]*logsafe(z[i]/α[i])
        end
        res = exp(res) - sumsq(@view z[dim1+1:end])
        # res = mapreduce((i,j)->2*i*logsafe(j/i),+,α,(@view z[1:dim1]))
        # println("primal residual is: ", res)
        if res > zero(T)
            return true
        end
    end
    
    return false
end

@inline function barrier_primal(
    K::DualGenPowerCone{T},
    s::AbstractVector{T}, 
) where {T}

    # Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    # NB: ⟨s,g(s)⟩ = -(dim1(K)+1) = - ν

    # can't use "work" here because it was already
    # used to construct the argument s in some cases
    g = K.data.work_pb

    gradient_primal!(K,g,s)      
    g .= -g                 #-g(s)
    # println("Inner product is ", dot(g,s))
    # @assert(is_dual_feasible(K,g))
    return -barrier_dual(K,g) #- degree(K)      #YC: previously a bug, force it back to the correct one
end 


@inline function barrier_dual(
    K::DualGenPowerCone{T},
    z::AbstractVector{T}, 
) where {T}

    # Dual barrier
    α = K.α

    res = mapreduce((i,j)->2*i*logsafe(j),+,α,(@view z[1:dim1(K)]))
    res = exp(res) - sumsq(@view z[dim1(K)+1:end])
    barrier = -logsafe(res) 
    @inbounds for i = 1:dim1(K)
        barrier -= (one(T)-α[i])*logsafe(z[i])
    end

    return barrier

end

#YC: monitor the primal and dual residuals
function monitor_residuals(
    K::DualGenPowerCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T}
) where {T}
    
    α = K.α

    #Dual residual
    phi = mapreduce((i,j)->2*i*logsafe(j),+,α,(@view z[1:dim1(K)]))
    phi = exp(phi)
    norm2w = sumsq(@view z[dim1(K)+1:end])
    ζd = phi - norm2w

    #Primal residual
    phi = mapreduce((i,j)->2*i*logsafe(j/i),+,α,(@view s[1:dim1(K)]))
    phi = exp(phi)
    norm2w = sumsq(@view s[dim1(K)+1:end])
    ζp = phi - norm2w

    println("primal ζp is: ", ζp, "  dual ζd is: ", ζd)
end

# update gradient and Hessian at dual z = (u,w)
function update_dual_grad_H(
    K::DualGenPowerCone{T},
    z::AbstractVector{T}
) where {T}
    
    α = K.α
    data = K.data
    p = data.p
    q = data.q
    r = data.r 
    d1 = data.d1

    # ϕ = ∏_{i ∈ dim1}(ui)^(2*αi), ζ = ϕ - ||w||^2
    phi = mapreduce((i,j)->2*i*logsafe(j),+,α,(@view z[1:dim1(K)]))
    phi = exp(phi)
    norm2w = sumsq(@view z[dim1(K)+1:end])
    ζ = phi - norm2w

    @assert ζ > zero(T)

    # compute the gradient at z
    grad = data.grad
    τ = q           # τ shares memory with q

    @inbounds for i = 1:dim1(K)
        τ[i] = 2*α[i]/z[i]
        grad[i] = -τ[i]*phi/ζ - (1-α[i])/z[i]
    end
    @inbounds for i = (dim1(K)+1):dim(K)
        grad[i] = 2*z[i]/ζ
    end

    # compute Hessian information at z 
    p0 = sqrt(phi*(phi+norm2w)/2)
    p1 = -2*phi/p0
    q0 = sqrt(ζ*phi/2)
    r1 = 2*sqrt(ζ/(phi+norm2w))

    #YC: p0^2-q0^2 = phi*norm2w donesn't hold for the initial value 
    # when the dimension of n becomes quite large
    # It seems that the dual scaling is not a good choice since we need to have the additional term ∏(1/α[i])^(2*α[i]),
    # which will be increasing when n tends to infinity

    # compute the diagonal d1,d2
    @inbounds for i = 1:dim1(K)
        d1[i] = τ[i]*phi/(ζ*z[i]) + (1-α[i])/(z[i]*z[i])
    end   
    data.d2 = 2/ζ

    # compute p, q, r where τ shares memory with q
    p[1:dim1(K)] .= p0*τ/ζ
    @views p[(dim1(K)+1):end] .= p1*z[(dim1(K)+1):end]/ζ

    q .*= q0/ζ      #τ is abandoned
    @views r .= r1*z[(dim1(K)+1):end]/ζ

    #Make the copy for higher-order correction
    K.data.phi = phi
    K.data.ζ = ζ
    K.data.w2 = norm2w

end

# Compute the primal gradient of f(s) at s
# solve it by the Newton-Raphson method
function gradient_primal!(
    K::DualGenPowerCone{T},
    g::AbstractVector{T},
    s::AbstractVector{T},
) where {T}

    α = K.α

    # unscaled phi
    phi = mapreduce((i,j)->2*i*logsafe(j),+,α,(@view s[1:dim1(K)]))
    phi = exp(phi)

    # obtain g1 from the Newton-Raphson method
    p = @view s[1:dim1(K)]
    r = @view s[dim1(K)+1:end]
    gp = @view g[1:dim1(K)]
    gr = @view g[dim1(K)+1:end]
    norm_r = norm(r)

    if norm_r > eps(T)
        g1 = _newton_raphson_dualgenpowcone(norm_r,phi,α,dim1(K))
        @. gr = g1*r/norm_r
        @. gp = -(α*(one(T) + g1*norm_r)+one(T))/p
    else
        @. gr = zero(T)
        @. gp = -(1+α)/p
    end

    return nothing
end

function gradient_dual!(
    K::DualGenPowerCone{T},
    grad::AbstractVector{T},
    z::AbstractVector{T}
) where {T}
    
    α = K.α

    # ϕ = ∏_{i ∈ dim1}(ui)^(2*αi), ζ = ϕ - ||w||^2
    phi = mapreduce((i,j)->2*i*logsafe(j),+,α,(@view z[1:dim1(K)]))
    phi = exp(phi)
    norm2w = sumsq(@view z[dim1(K)+1:end])
    ζ = phi - norm2w
    @assert ζ > zero(T)

    # compute the gradient at z
    @inbounds for i = 1:dim1(K)
        grad[i] = -2*α[i]*phi/(ζ*z[i]) - (1-α[i])/z[i]
    end
    @inbounds for i = (dim1(K)+1):dim(K)
        grad[i] = 2*z[i]/ζ
    end

end
# ----------------------------------------------
#  internal operations for generalized power cones

function _newton_raphson_dualgenpowcone(
    norm_r::T,
    phi::T,
    α::AbstractVector{T},
    d1::Int
) where {T}

    # init point x0: f(x0) > 0
    d11 = d1*d1
    x0 = -inv(norm_r) + d1*(norm_r + sqrt(phi*((d1/norm_r)^2*phi + d11 - 1)))/(phi*d11 - norm_r*norm_r)

    # # additional shift due to the choice of dual barrier
    # t0 = - 2*α*logsafe(α) - 2*(1-α)*logsafe(1-α)   

    # function for f(x) = 0
    function f0(x)
        f0 = - logsafe(phi) - logsafe(2*x/norm_r + x*x) - 2*logsafe(2*x/norm_r);
        @inbounds for i in eachindex(α)
            f0 += 2*α[i]*logsafe(2*α[i]*x*x + 2*x*(1+α[i])/norm_r)
        end

        return f0
    end

    # first derivative
    function f1(x)
        f1 = -2*(x + inv(norm_r))/(x*(x + 2/norm_r));
        @inbounds for i in eachindex(α)
            f1 += 2*α[i]/(x + (1+inv(α[i]))/norm_r)
        end

        return f1
    end
    
    return _newton_raphson_onesided(x0,f0,f1)
end



####################################
# 3rd-order correction
####################################
#H^{-1}*x 
function mul_Hinv!(
    K::DualGenPowerCone{T},
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


function higher_correction_mosek!(
    K::DualGenPowerCone{T},
    η::AbstractVector{T},
    ds::AbstractVector{T},
    dz::AbstractVector{T}
) where {T}

    #data.work = H^{-1}*ds
    mul_Hinv!(K,K.data.work,ds)
    # tensor product
    α = K.α
    dim1 = Clarabel.dim1(K)
    dim2 = Clarabel.dim2(K)

    du = @view K.data.work[1:dim1]
    dw = @view K.data.work[dim1+1:end]
    tu = @view dz[1:dim1]
    tw = @view dz[dim1+1:end]

    z = K.data.z
    u = @view z[1:dim1]
    w = @view z[dim1+1:end]

    #workspace
    τ = @view ds[1:dim1]      #YC: ds is no longer used later
    @. τ = 2*α/u
    normd = @view K.data.work_pp[1:dim1]
    @. normd = du/u

    #constants
    ϕ = K.data.phi
    ζ = K.data.ζ
    ϕdζ = ϕ/ζ
    τdu = dot(τ,du)
    wdw = dot(w,dw)
    τtu = dot(τ,tu)
    wtw = dot(w,tw)
    dtw = dot(dw,tw)
    c0 = zero(T)
    @inbounds for i in 1:dim1
        c0 += τ[i]*normd[i]*tu[i]
    end

    #search direction
    diru = @view  η[1:dim1]
    dirw = @view  η[dim1+1:end]

    #constant 
    cu1 = ϕdζ*(2*ϕdζ-1)*((1-ϕdζ)*τdu*τtu + 2/ζ*(wdw*τtu + τdu*wtw)) - 2*ϕdζ/ζ*(4*wdw*wtw/ζ + dtw) + ϕdζ*(1-ϕdζ)*c0
    cud = 2*ϕdζ*wtw/ζ + ϕdζ*(1-ϕdζ)*τtu
    cut = 2*ϕdζ*wdw/ζ + ϕdζ*(1-ϕdζ)*τdu

    @inbounds for i in 1:dim1
        diru[i] = cu1 + (cud*du[i] + cut*tu[i])/u[i]
    end
    @. diru  *= τ
    @inbounds for i in 1:dim1
        diru[i] -= 2*du[i]*tu[i]*((1-α[i])/u[i]+τ[i]*ϕdζ)/(u[i]*u[i])
    end
    
    cw1 = 2*ϕ*((2ϕdζ-1)*τdu*τtu + c0 - 4/ζ*(wdw*τtu+τdu*wtw)) + (16*wdw*wtw/ζ + 4*dtw)
    cwd = 4*wtw - 2*ϕ*τtu
    cwt = 4*wdw - 2*ϕ*τdu
    @inbounds for i in 1:dim2
        dirw[i] = (cw1*w[i] + (cwd*dw[i]+cwt*tw[i]))/(ζ*ζ) 
    end

    @. η /= -2
    
    return nothing
end