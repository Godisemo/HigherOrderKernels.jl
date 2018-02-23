module HigherOrderKernels

export GaussianKernel,
       PolynomialKernel,
       UniformKernel,
       EpanechnikovKernel,
       BiweightKernel,
       TriweightKernel,
       bandwidth,
       kpdf,
       order

using GaussQuadrature

abstract type AbstractKernel{ν} end
struct GaussianKernel{ν} <: AbstractKernel{ν} end
struct PolynomialKernel{s,ν} <: AbstractKernel{ν} end
const UniformKernel = PolynomialKernel{0}
const EpanechnikovKernel = PolynomialKernel{1}
const BiweightKernel = PolynomialKernel{2}
const TriweightKernel = PolynomialKernel{3}

order(::Type{T}) where {ν,T<:AbstractKernel{ν}} = ν

function roughness(k::Type{T}) where {s,ν,T<:PolynomialKernel{s,ν}}
  N = 2s+2(div(ν,2)-1)+1
  x, w = legendre(N)
  R = sum(w[i] * kernel(k, x[i])^2 for i=1:N)
end

function firstnzmoment(k::Type{T}) where {s,ν,T<:PolynomialKernel{s,ν}}
  N = s+2*div(ν,2)
  x, w = legendre(N)
  κ = sum(w[i] * x[i]^ν * kernel(k, x[i]) for i=1:N)
end

function roughness(k::Type{T}) where {ν,T<:GaussianKernel{ν}}
  N = 2(div(ν,2)-1)+1
  x, w = hermite(N)
  R = sum(w[i] * exp(x[i]^2) * kernel(k, x[i])^2 for i=1:N)
end

function firstnzmoment(k::Type{T}) where {ν,T<:GaussianKernel{ν}}
  N = 2*div(ν,2)
  x, w = hermite(N)
  κ = sum(w[i] * (sqrt(2) * x[i])^ν * exp(x[i]^2) * kernel(k, sqrt(2) * x[i]) * sqrt(2) for i=1:N)
end

const bandwidth_constant_lookup = Dict{Type{T} where T<:AbstractKernel,Float64}()

function bandwidth_constant(k::Type{T}) where {ν,T<:AbstractKernel{ν}}
  if haskey(bandwidth_constant_lookup, k)
    return bandwidth_constant_lookup[k]
  end
  R = roughness(k)
  κ = firstnzmoment(k)
  C = Float64(2*((sqrt(pi)*factorial(big(ν))^3*R)/(2*ν*factorial(big(2ν))*κ^2))^(1/(2ν+1)))
  bandwidth_constant_lookup[k] = C
end

bandwidth(k::Type{T}, data) where {ν,T<:AbstractKernel{ν}} =
  bandwidth_constant(k) * std(data) * size(data, 1)^(-1/(2ν+1))

pochhammer(x, n) = gamma(x + n) / gamma(x)

# efficient evaluation of polynomial using Horner's rule
function polyval_expr(p::AbstractArray{T,1}, x::S) where {T,S}
  lenp = length(p)
  if lenp == 0
    return zero(T) * x
  else
    y = convert(T, p[end])
    for i = lenp-1:-1:1
      y = :($(p[i]) + $x*$y)
    end
    return y
  end
end

@generated function kernel(::Type{PolynomialKernel{s,ν}}, u) where {s,ν}
  r = div(ν, 2)
  M_constant = pochhammer(big(1/2), s+1) / factorial(big(s))
  M_expr = (s == 0) ? :(abs(u) <= 1) :
           (s == 1) ? :((1-u2)*(abs(u) <= 1)) :
                      :((1-u2)^$s*(abs(u) <= 1))
  if r == 1
    expr = :($(Float64(M_constant)) * $M_expr)
  else
    B_constant = pochhammer(big(3/2+s), r-1) / pochhammer(big(s+1), r-1) * pochhammer(big(3/2), r-1)
    B_coefficients = Float64.(B_constant * M_constant * [(-1)^k * pochhammer(big(1/2+s+r), k) / (factorial(big(k)) * factorial(big(r-1-k)) * pochhammer(big(3/2), k)) for k=0:r-1])
    expr = polyval_expr(B_coefficients, :u2)
    expr = :($expr * $M_expr)
  end
  quote
    $(Expr(:meta, :inline))
    u2 = u^2
    $expr
  end
end

@generated function kernel(::Type{GaussianKernel{ν}}, u) where {ν}
  r = div(ν, 2)
  ϕ_expr = :(exp(-u2/2)/sqrt(2*pi))
  if r == 1
    expr = ϕ_expr
  else
    Q_coefficients = Float64.([(-1)^i * factorial(big(2r))/(big(2)^(2r-i-1)*factorial(big(r))*factorial(big(2i+1))*factorial(big(r-i-1))) for i=0:r-1])
    expr = polyval_expr(Q_coefficients, :u2)
    expr = :($expr * $ϕ_expr)
  end
  quote
    $(Expr(:meta, :inline))
    u2 = u^2
    $expr
  end
end

kpdf(k, x, data) = kpdf(k, x, data, bandwidth(k, data))

function kpdf(::Type{T}, x, data, h) where {T<:AbstractKernel}
  sum = 0.0
  @simd for y in data
    sum += kernel(T, (y - x) / h)
  end
  sum /= length(data) * h
  return sum
end

end # module
