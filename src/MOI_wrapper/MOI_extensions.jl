using MathOptInterface 
using ..Clarabel
const DefaultInt = Clarabel.DefaultInt

# This structure is parallel to the our API type, but 
# it is subtyped from MOI.AbstractVectorSet rather than 
# our own user API SupportedCone type.   

#NB: This "GenPowerCone" will end up within Clarabel.MOI.  
# It is not the same type as the one our top level API.

struct GenPowerCone{T<:Real} <: MathOptInterface.AbstractVectorSet
    α::Vector{T}
    dim2::DefaultInt
    function GenPowerCone(α::AbstractVector{T}, dim2::DefaultInt) where{T}
        @assert all(α .> zero(T))
        @assert isapprox(sum(α),one(T), atol=eps()*length(α)/2)
        new{T}(collect(α), dim2)
    end
end 

GenPowerCone(args...) = GenPowerCone{Float64}(args...)

# enable use of this type as a MOI constraint type
MathOptInterface.dimension(cone::GenPowerCone) = length(cone.α) + cone.dim2
Base.copy(cone::GenPowerCone{Float64}) = deepcopy(cone)

struct DualGenPowerCone{T<:Real} <: MathOptInterface.AbstractVectorSet
    α::Vector{T}
    dim2::Int
    function DualGenPowerCone(α::AbstractVector{T}, dim2::Int) where{T}
        @assert all(α .> zero(T))
        @assert isapprox(sum(α),one(T), atol=eps()*length(α)/2)
        new{T}(collect(α), dim2)
    end
end 

DualGenPowerCone(args...) = DualGenPowerCone{Float64}(args...)

# enable use of this type as a MOI constraint type
MathOptInterface.dimension(cone::DualGenPowerCone) = length(cone.α) + cone.dim2
Base.copy(cone::DualGenPowerCone{Float64}) = deepcopy(cone)

struct PowerMeanCone{T<:Real} <: MathOptInterface.AbstractVectorSet
    α::Vector{T}
    function PowerMeanCone(α::AbstractVector{T}) where{T}
        @assert all(α .> zero(T))
        @assert isapprox(sum(α),one(T), atol=eps()*length(α)/2)
        new{T}(collect(α))
    end
end

PowerMeanCone(args...) = PowerMeanCone{Float64}(args...)
MathOptInterface.dimension(cone::PowerMeanCone) = (length(cone.α)+1)
Base.copy(cone::PowerMeanCone{Float64}) = deepcopy(cone)

struct DualPowerMeanCone{T<:Real} <: MathOptInterface.AbstractVectorSet
    α::Vector{T}
    function DualPowerMeanCone(α::AbstractVector{T}) where{T}
        @assert all(α .> zero(T))
        @assert isapprox(sum(α),one(T), atol=eps()*length(α)/2)
        new{T}(collect(α))
    end
end

DualPowerMeanCone(args...) = DualPowerMeanCone{Float64}(args...)
MathOptInterface.dimension(cone::DualPowerMeanCone) = (length(cone.α)+1)
Base.copy(cone::DualPowerMeanCone{Float64}) = deepcopy(cone)

struct EntropyCone{T<:Real} <: MathOptInterface.AbstractVectorSet
    dim::Int
end

EntropyCone(args...) = EntropyCone{Float64}(args...)
MathOptInterface.dimension(cone::EntropyCone) = (cone.dim)
Base.copy(cone::EntropyCone{Float64}) = deepcopy(cone)

struct DualEntropyCone{T<:Real} <: MathOptInterface.AbstractVectorSet
    dim::Int
end

DualEntropyCone(args...) = DualEntropyCone{Float64}(args...)
MathOptInterface.dimension(cone::DualEntropyCone) = (cone.dim)
Base.copy(cone::DualEntropyCone{Float64}) = deepcopy(cone)