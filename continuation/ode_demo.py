import numpy as np
from numpy import linalg as npla
import scipy.sparse.linalg as spla
import scipy as sp
from iterative import *
from linear_operators import *
from sys import exit
from utils import *
np.random.seed(6)

np.set_printoptions(
    linewidth=200,
    legacy='1.25',
    precision=3)

nt = 256
T = nt
zbar = -0.00 + 0.8j
dz = 0.16j
theta = 0.5
alpha = 1e-4
zperiods = 2

maxiter = 12
atol = 1e-10
rtol = 1e-10

omega = 1.0
omega_x = 1.0
omega_v = 0.5
# omega_v = "optimal"

update = 'multiplicative'

dtype = complex

ones = np.ones(nt, dtype=dtype)

t = np.linspace(T/nt, T, nt, dtype=dtype)

zprofile = np.sin(zperiods*2*np.pi*t/T) if (zperiods > 0) else ones
z = zbar + dz*zprofile
phi = theta_method(theta, z)

phi_bar = theta_method(theta, np.mean(z))
phi_mean = geomean(phi)

phihat = phi_mean

eta, vtheta, dphi = eta_expected(phi, phihat, nt, alpha)
print(f"{eta = :.3e} | {vtheta = :.3e} | {dphi = :.3e}")

A = ConstantCoefficientOperator(phihat, nt, inverse=False)
P = CirculantOperator(phihat, nt, alpha, inverse=False)
J = VariableCoefficientOperator(phi, nt, inverse=False)
Q = PeriodicOperator(phi, nt, alpha, inverse=False)

Ainv = ConstantCoefficientOperator(phihat, nt, inverse=True)
Pinv = CirculantOperator(phihat, nt, alpha, inverse=True)
Jinv = VariableCoefficientOperator(phi, nt, inverse=True)
Qinv = PeriodicOperator(phi, nt, alpha, inverse=True)

print(f"{np.min(np.abs(phi)) = :.3e}")
print(f"{np.max(np.abs(phi)) = :.3e}")
print(f"{np.abs(phi_bar)  = :.3e} | {phi_bar  = :.3e}")
print(f"{np.abs(phi_mean) = :.3e} | {phi_mean = :.3e}")

# rhs and solution
x0 = random_array(nt, dtype=dtype)
b = random_array(nt, dtype=dtype)
x0 /= npla.norm(x0)
b /= npla.norm(b)

residuals = []

def callback(xk, r, pr, res, k):
    residuals.append(res)
    print(f"it = {k:>2d} | {res = :.4e}")

AA = J
MM = Pinv
print()
print("Richardson iterations:")
richardson(AA, b, x0=x0, M=MM,
           omega=omega,
           rtol=rtol, atol=atol,
           maxiter=maxiter,
           callback=callback)

rate, r2 = contraction_rate(residuals)
print(f"Convergence rate = {rate:.4e} | R^2 = {r2:.3e}")
print()

# calculate fundamental solution of J from I
v0 = np.zeros(nt, dtype=dtype)
v0[0] = 1
v = Jinv(v0)
v /= phi_mean**np.arange(nt)

F = FloquetTransformOperator(phi, phihat, nt, v, alpha=alpha, inverse=False)
Finv = FloquetTransformOperator(phi, phihat, nt, v, alpha=alpha, inverse=True)

Imat = np.eye(nt)
Fmat = F(Imat)
Qmat = Q(Imat)
Jmat = J(Imat)
dFQ = Fmat - Qmat
dFJ = Fmat - Jmat
print("Exact fundamental solution:")
print(f"max(F-Q) = {np.max(np.abs(dFQ)):.3e}")
print(f"max(F-J) = {np.max(np.abs(dFJ)):.3e}")
print()

Finv.v[:] = 1
# floquet_continuation(J, Finv, b, x0=x0,
#                      maxiter=maxiter, atol=atol,
#                      omega_x=omega_x, omega_v=omega_v,
#                      update=update)
floquet_continuation_shifted_inverse(J, Finv, b, x0=x0,
                     maxiter=maxiter, atol=atol,
                     shift_fudge=1e-8,
                     omega_x=omega_x, omega_v=omega_v,
                     update=update)

F.update_v(v=Finv.v)
Fmat = F(Imat)
dFQ = Fmat - Qmat
dFJ = Fmat - Jmat
print()
print("Approximate fundamental solution:")
print(f"max(F-Q) = {np.max(np.abs(dFQ)):.3e}")
print(f"max(F-J) = {np.max(np.abs(dFJ)):.3e}")
