using HigherOrderKernels
using GaussQuadrature
using Base.Test

import HigherOrderKernels: kernel, bandwidth_constant

for s = 0:20, ν = 2:2:20
    N = s + div(ν, 2)
    x, w = legendre(N)
    @test kernel.(PolynomialKernel{s,ν}, x)'*w ≈ 1
end

@test round(bandwidth_constant(PolynomialKernel{1,2}), 2) == 2.34
@test round(bandwidth_constant(PolynomialKernel{1,4}), 2) == 3.03
@test round(bandwidth_constant(PolynomialKernel{1,6}), 2) == 3.53
@test round(bandwidth_constant(PolynomialKernel{2,2}), 2) == 2.78
@test round(bandwidth_constant(PolynomialKernel{2,4}), 2) == 3.39
@test round(bandwidth_constant(PolynomialKernel{2,6}), 2) == 3.84
@test round(bandwidth_constant(PolynomialKernel{3,2}), 2) == 3.15
@test round(bandwidth_constant(PolynomialKernel{3,4}), 2) == 3.72
@test round(bandwidth_constant(PolynomialKernel{3,6}), 2) == 4.13
