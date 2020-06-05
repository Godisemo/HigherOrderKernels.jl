using HigherOrderKernels
using GaussQuadrature
using Test

import HigherOrderKernels: density_kernel, density_bandwidth_constant, roughness, firstnzmoment

for s = 0:20, ν = 2:2:20
    N = s + div(ν, 2)
    x, w = legendre(N)
    @test density_kernel.(PolynomialKernel{s,ν}, x)'*w ≈ 1
end

for ν = 2:2:20
    N = div(ν, 2)
    k = x -> density_kernel(GaussianKernel{ν}, x)
    x, w = hermite(N)
    @test sum(w[i]*k(sqrt(2)*x[i])*exp(x[i]^2) for i=1:N)*sqrt(2) ≈ 1
end

# table 1
@test roughness(     UniformKernel{2}) ≈ 1/2
@test roughness(EpanechnikovKernel{2}) ≈ 3/5
@test roughness(    BiweightKernel{2}) ≈ 5/7
@test roughness(   TriweightKernel{2}) ≈ 350/429
@test roughness(    GaussianKernel{2}) ≈ 1/(2sqrt(pi))
@test firstnzmoment(     UniformKernel{2}) ≈ 1/3
@test firstnzmoment(EpanechnikovKernel{2}) ≈ 1/5
@test firstnzmoment(    BiweightKernel{2}) ≈ 1/7
@test firstnzmoment(   TriweightKernel{2}) ≈ 1/9
@test firstnzmoment(    GaussianKernel{2}) ≈ 1

# table 2
@test roughness(EpanechnikovKernel{4}) ≈ 5/4
@test roughness(    BiweightKernel{4}) ≈ 805/572
@test roughness(   TriweightKernel{4}) ≈ 3780/2431
@test roughness(    GaussianKernel{4}) ≈ 27/(32sqrt(pi))
@test firstnzmoment(EpanechnikovKernel{4}) ≈ -1/21
@test firstnzmoment(    BiweightKernel{4}) ≈ -1/33
@test firstnzmoment(   TriweightKernel{4}) ≈ -3/143
@test firstnzmoment(    GaussianKernel{4}) ≈ -3

# table 3
@test roughness(EpanechnikovKernel{6}) ≈ 1575/832
@test roughness(    BiweightKernel{6}) ≈ 29295/14144
@test roughness(   TriweightKernel{6}) ≈ 301455/134368
@test roughness(    GaussianKernel{6}) ≈ 2265/(2048sqrt(pi))
@test firstnzmoment(EpanechnikovKernel{6}) ≈ 5/429
@test firstnzmoment(    BiweightKernel{6}) ≈ 1/143
@test firstnzmoment(   TriweightKernel{6}) ≈ 1/221
@test firstnzmoment(    GaussianKernel{6}) ≈ 15

# table 4
@test round(density_bandwidth_constant(PolynomialKernel{1,2}), digits=2) == 2.34
@test round(density_bandwidth_constant(PolynomialKernel{1,4}), digits=2) == 3.03
@test round(density_bandwidth_constant(PolynomialKernel{1,6}), digits=2) == 3.53
@test round(density_bandwidth_constant(PolynomialKernel{2,2}), digits=2) == 2.78
@test round(density_bandwidth_constant(PolynomialKernel{2,4}), digits=2) == 3.39
@test round(density_bandwidth_constant(PolynomialKernel{2,6}), digits=2) == 3.84
@test round(density_bandwidth_constant(PolynomialKernel{3,2}), digits=2) == 3.15
@test round(density_bandwidth_constant(PolynomialKernel{3,4}), digits=2) == 3.72
@test round(density_bandwidth_constant(PolynomialKernel{3,6}), digits=2) == 4.13
@test round(density_bandwidth_constant(GaussianKernel{2}), digits=2) == 1.06
@test round(density_bandwidth_constant(GaussianKernel{4}), digits=2) == 1.08
@test round(density_bandwidth_constant(GaussianKernel{6}), digits=2) == 1.08
