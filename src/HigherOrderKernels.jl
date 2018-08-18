module HigherOrderKernels

export GaussianKernel,
       PolynomialKernel,
       UniformKernel,
       EpanechnikovKernel,
       BiweightKernel,
       TriweightKernel,
       density_bandwidth,
       distribution_bandwidth,
       kpdf,
       kcdf,
       order,
       poly_order

using GaussQuadrature,
      Polynomials

abstract type AbstractKernel{ν} end
struct GaussianKernel{ν} <: AbstractKernel{ν} end
struct PolynomialKernel{s,ν} <: AbstractKernel{ν} end
const UniformKernel = PolynomialKernel{0}
const EpanechnikovKernel = PolynomialKernel{1}
const BiweightKernel = PolynomialKernel{2}
const TriweightKernel = PolynomialKernel{3}

poly_order(k::Type{T}) where {s,ν,T<:PolynomialKernel{s,ν}} =
  degree(_u_poly(k))

order(::Type{T}) where {ν,T<:AbstractKernel{ν}} = ν

function roughness(k::Type{T}) where {s,ν,T<:PolynomialKernel{s,ν}}
  # N = 2s+2(div(ν,2)-1)+1
  # x, w = legendre(N)
  # R = sum(w[i] * density_kernel(k, x[i])^2 for i=1:N)
  Float64(polyint(_u_poly(k)^2, -1, 1))
end

function distribution_roughness(k::Type{T}) where {s,ν,T<:PolynomialKernel{s,ν}}
  I = polyint(_u_poly(k))
  K = I - I(-1)
  Float64(1 - polyint(K^2, -1, 1))
end

function firstnzmoment(k::Type{T}) where {s,ν,T<:PolynomialKernel{s,ν}}
  # N = s+2*div(ν,2)
  # x, w = legendre(N)
  # κ = sum(w[i] * x[i]^ν * density_kernel(k, x[i]) for i=1:N)
  polyint(Poly([zeros(ν); 1]) * _u_poly(k), -1, 1)
end

function roughness(k::Type{T}) where {ν,T<:GaussianKernel{ν}}
  N = 2(div(ν,2)-1)+1
  x, w = hermite(N)
  R = sum(w[i] * exp(x[i]^2) * density_kernel(k, x[i])^2 for i=1:N)
end

function firstnzmoment(k::Type{T}) where {ν,T<:GaussianKernel{ν}}
  N = 2*div(ν,2)
  x, w = hermite(N)
  κ = sum(w[i] * (sqrt(2) * x[i])^ν * exp(x[i]^2) * density_kernel(k, sqrt(2) * x[i]) * sqrt(2) for i=1:N)
end

function _std(x)
  iqr = diff(quantile(x, [0.25, 0.75]))[1] / 1.3489795003921636 # diff(quantile(Normal, [0.25, 0.75]))
  σ = std(x)
  min(iqr, σ)
end

const density_bandwidth_constant_lookup = Dict{Type{T} where T<:AbstractKernel,Float64}()

function density_bandwidth_constant(k::Type{T}) where {ν,T<:AbstractKernel{ν}}
  if haskey(density_bandwidth_constant_lookup, k)
    return density_bandwidth_constant_lookup[k]
  end
  R = roughness(k)
  κ = firstnzmoment(k)
  C = Float64(2*((sqrt(pi)*factorial(big(ν))^3*R)/(2*ν*factorial(big(2ν))*κ^2))^(1/(2ν+1)))
  density_bandwidth_constant_lookup[k] = C
end

density_bandwidth(k::Type{T}, data) where {ν,T<:AbstractKernel{ν}} =
  density_bandwidth_constant(k) * _std(data) * size(data, 1)^(-1/(2ν+1))

const distribution_bandwidth_constant_lookup = Dict{Type{T} where T<:AbstractKernel,Float64}()

function distribution_bandwidth_constant(k::Type{T}) where {ν,T<:AbstractKernel{ν}}
  if haskey(distribution_bandwidth_constant_lookup, k)
    return distribution_bandwidth_constant_lookup[k]
  end
  R = distribution_roughness(k)
  κ = firstnzmoment(k)
  C = Float64(2*((sqrt(pi)*factorial(big(ν))^3*R*(2ν-1))/(factorial(big(2ν))*κ^2*ν))^(1/(2ν-1)))
  distribution_bandwidth_constant_lookup[k] = C
end

distribution_bandwidth(k::Type{T}, data) where {ν,T<:AbstractKernel{ν}} =
  distribution_bandwidth_constant(k) * _std(data) * size(data, 1)^(-1/(2ν-1))

pochhammer(x, n) = gamma(x + n) / gamma(x)

# efficient evaluation of polynomial using Horner's rule
function polyval_expr(p::AbstractArray{T,1}, x::S) where {T,S}
  lenp = length(p)
  if lenp == 0
    return :($(zero(T)) * x)
  else
    y = convert(T, p[end])
    for i = lenp-1:-1:1
      y = :(muladd($x, $y, $(p[i])))
    end
    return y
  end
end

function _u2_poly(::Type{PolynomialKernel{s,ν}}) where {s,ν}
  r = div(ν, 2)
  c = pochhammer(big(1/2), s+1) * pochhammer(big(3/2+s), r-1) * pochhammer(big(3/2), r-1) / factorial(big(s+r-1))
  Poly([c * (-1)^k * pochhammer(big(1/2+s+r), k) / (factorial(big(k)) * factorial(big(r-1-k)) * pochhammer(big(3/2), k)) for k=0:r-1], :u2)
end

function _u_poly(k::Type{PolynomialKernel{s,ν}}) where {s,ν}
  p0 = Poly([1, -1], :u2)^s
  p1 = _u2_poly(k)
  p = p0 * p1
  u2 = Poly([0, 0, 1])
  p(u2)
end

@generated function density_kernel(k::Type{PolynomialKernel{s,ν}}, u) where {s,ν}
  e0 = (s == 0) ? :(abs(u) <= 1) :
       (s == 1) ? :((1-u2)*(abs(u) <= 1)) :
                  :((1-u2)^$s*(abs(u) <= 1))
  p1 = _u2_poly(PolynomialKernel{s,ν})
  e1 = polyval_expr(Float64.(coeffs(p1)), :u2)
  quote
    $(Expr(:meta, :inline))
    u2 = u^2
    $e1 * $e0
  end
end

@generated function distribution_kernel(k::Type{PolynomialKernel{s,ν}}, u) where {s,ν}
  p = _u_poly(PolynomialKernel{s,ν})
  P = polyint(p)
  P -= P(-1)
  e = polyval_expr(Float64.(coeffs(P)), :u)
  quote
    $(Expr(:meta, :inline))
    $e * (abs(u) <= 1) + (u > 1)
  end
end

@generated function density_kernel(::Type{GaussianKernel{ν}}, u) where {ν}
  r = div(ν, 2)
  ϕ = :(exp(-0.5u2))
  p = Poly(Float64.([(-1)^i / sqrt(2*pi) * factorial(big(2r))/(big(2)^(2r-i-1)*factorial(big(r))*factorial(big(2i+1))*factorial(big(r-i-1))) for i=0:r-1]))
  e = polyval_expr(coeffs(p), :u2)
  quote
    $(Expr(:meta, :inline))
    u2 = u^2
    $e * $ϕ
  end
end

kpdf(k, x, data) = kpdf(k, x, data, density_bandwidth(k, data))

function kpdf(::Type{T}, x, data, h) where {T<:AbstractKernel}
  sum = 0.0
  @fastmath @simd for y in data
    sum += density_kernel(T, (y - x) / h)
  end
  sum /= length(data) * h
  return sum
end

function kpdf(::Type{T}, x, data, h, lb, ub) where {T<:AbstractKernel}
  sum = 0.0
  if !(lb <= x <= ub)
    return sum
  end
  @simd for y in data
    sum += density_kernel(T, (y - x) / h)
  end
  if isfinite(lb)
    @simd for y in data
      sum += density_kernel(T, (y - (2lb - x)) / h)
    end
  end
  if isfinite(ub)
    @simd for y in data
      sum += density_kernel(T, (y - (2ub - x)) / h)
    end
  end
  sum /= length(data) * h
  return sum
end

kcdf(k, x, data) = kpdf(k, x, data, distribution_bandwidth(k, data))

function kcdf(::Type{T}, x, data, h) where {T<:AbstractKernel}
  sum = 0.0
  @fastmath @simd for y in data
    sum += distribution_kernel(T, (x - y) / h)
  end
  sum /= length(data)
  return sum
end

end # module
