# -------------------------------------
# Enum and dict for user interface
# -------------------------------------
"""
    SupportedCone
An abstract type use by the Clarabel API used when passing cone specifications to the solver [`setup!`](@ref).
The currently supported concrete types are:

* `ZeroConeT`       : The zero cone.  Used to define equalities.
* `NonnegativeConeT`: The nonnegative orthant.
* `SecondOrderConeT`: The second order / Lorentz / ice-cream cone.
* `PSDTriangleConeT`: The positive semidefinite cone (triangular format).
* `ExponentialConeT`:         The exponential cone (in R^3)
* `PowerConeT`      : The power cone with power α (in R^3)
* `GenPowerConeT`   : The generalized power cone with power α

"""
abstract type SupportedCone <: MOI.AbstractVectorSet end

struct ZeroConeT <: SupportedCone
    dim::DefaultInt
end

struct NonnegativeConeT <: SupportedCone
    dim::DefaultInt
end

struct SecondOrderConeT <: SupportedCone
    dim::DefaultInt
end

struct PowerConeT <: SupportedCone
    #dim = 3 always
    α::DefaultFloat
end

struct GenPowerConeT <: SupportedCone
    α::Vector{DefaultFloat}
    dim2::DefaultInt
    function GenPowerConeT(α::Vector{DefaultFloat}, dim2::DefaultInt)

        @assert all(α .> zero(DefaultFloat))
        @assert isapprox(sum(α),one(DefaultFloat), atol=eps()*length(α)/2)

        new(copy(α), dim2)
    end
end
dim1(cone::GenPowerConeT) = length(cone.α)
dim2(cone::GenPowerConeT) = cone.dim2

# enable use of this type as a MOI constraint type
MOI.dimension(cone::GenPowerConeT) = (length(cone.α) + cone.dim2)

struct ExponentialConeT <: SupportedCone
    #no fields, #dim = 3 always
end
struct PSDTriangleConeT <: SupportedCone
    dim::DefaultInt
end


# this reports the number of slack variables that
# will be generated by this cone.  Equivalent to
# `numel` for the internal cone representation

function nvars(cone:: SupportedCone)

    if isa(cone, PSDTriangleConeT)
        triangular_number(cone.dim)
    elseif isa(cone, ExponentialConeT)
        3
    elseif isa(cone, PowerConeT)
        3
    elseif isa(cone, GenPowerConeT)
        dim1(cone) + dim2(cone)
    else
        cone.dim
    end
end


# we use the SupportedCone as a user facing marker
# for the constraint types, and then map them through
# make_cone to get the internal cone representations.
function make_cone(T::Type, coneT)

    typeT = typeof(coneT)
    if typeT == ExponentialConeT
        cone = ConeDict[typeT]{T}()
    elseif typeT == PowerConeT
        cone = ConeDict[typeT]{T}(T(coneT.α))
    elseif typeT == GenPowerConeT
        cone = ConeDict[typeof(coneT)]{T}(T.(coneT.α),coneT.dim2)
    else
        cone = ConeDict[typeT]{T}(coneT.dim)
    end

end