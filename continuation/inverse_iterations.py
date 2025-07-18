import numpy as np
from numpy import linalg as npla
import scipy.sparse.linalg as spla
import scipy as sp
from iterative import *
from iterative import get_omega
from linear_operators import *
from sys import exit
from utils import *
np.random.seed(6)

np.set_printoptions(
    linewidth=200,
    legacy='1.25',
    precision=3)

nt = 128
T = nt
zbar = -0.00 + 0.8j
dz = 0.20j
theta = 0.5
alpha = 1e-4
zperiods = 1

inner_monitor = False
max_outer_iter = 6
max_inner_iter = 2
atol = 0
rtol = 1e-10
shift_fudge = 1e-8
mu = 0e-0
mu_min = 0e-0
# mu = 0
# mu_min = 0

omega = 0.95
# omega = "optimal"

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

print(f"{np.min(np.abs(phi)) = :.3e} | {np.max(np.abs(phi)) = :.3e}")
print(f"{np.abs(phi_bar)  = :.3e}    | {phi_bar  = :.3e}")
print(f"{np.abs(phi_mean) = :.3e}    | {phi_mean = :.3e}")

# rhs and solution
x0 = random_array(nt, dtype=dtype)
b = random_array(nt, dtype=dtype)
x0 /= npla.norm(x0)
b /= npla.norm(b)

residuals = []

def richardson_callback(xk, r, pr, res, k):
    residuals.append(res)
    if inner_monitor:
        print(f"    it = {k:>2d} | {res = :.4e}")

gmres_its = 0
def gmres_callback(pr_norm):
    global gmres_its
    res = pr_norm
    residuals.append(res)
    if inner_monitor:
        print(f"    it = {gmres_its:>2d} | {res = :.4e}")
    gmres_its += 1

# calculate fundamental solution of J from I
v = np.ones(nt, dtype=dtype)
v /= np.linalg.norm(v)

F = FloquetTransformOperator(phi, phihat, nt, v, alpha=alpha, inverse=False)
Finv = FloquetTransformOperator(phi, phihat, nt, v, alpha=alpha, inverse=True)

# inverse iterations
v = Finv.v.copy()

shift = 1.0 - Finv.phibar

Qs = PeriodicOperator(Finv.phis, Finv.n, shift=shift)
Qs_inv = FloquetTransformOperator(Finv.phis, Finv.phibar, Finv.n, Finv.v,
                                  shift=shift-shift_fudge, inverse=True)

resQ = []
b = np.zeros_like(v)

def update_rho():
    # need to use the rayleigh quotient for convergence
    Qs.update_shift(0.)
    Qsv = Qs(v)
    rho = np.vdot(v, Qsv)/np.vdot(v, v)

    Qs.update_shift(rho - mu)
    Qs_inv.update_shift(rho - (shift_fudge - mu))
    return rho

print()
for k in range(max_outer_iter):

    rho = update_rho()
    Qs.update_shift(rho)
    rQ = Qs(v)
    resQ.append(np.linalg.norm(rQ)/np.linalg.norm(v))
    print(f"it {k:>2d} | {rho = :.2e} | rQ = {resQ[-1]:.3e}")

    b[:] = (1 + mu)*v
    Qs.update_shift(rho - mu)
    # vals = richardson(Qs, b, M=Qs_inv, x0=v,
    #                   omega=omega,
    #                   rtol=rtol, atol=atol,
    #                   maxiter=max_inner_iter,
    #                   callback=richardson_callback)
    # v, _, iters, res = vals
    gmres_its -= gmres_its
    v, iters = spla.gmres(Qs, b, M=Qs_inv,# x0=v,
                         rtol=rtol, atol=atol,
                         restart=max_inner_iter,
                         maxiter=1,
                         callback=gmres_callback,
                         callback_type='pr_norm')
    print(f"{gmres_its = }")
    Qs.update_shift(rho)
    print()

    v /= np.linalg.norm(v)
    Qs_inv.update_v(v=v)

    mu = max(mu_min, 0.5*mu)

    if resQ[-1] < atol:
        rho = update_rho()
        Qs.update_shift(rho)
        rQ = Qs(v)
        resQ.append(np.linalg.norm(rQ/np.linalg.norm(v)))
        print(f"it {k+1:>2d} | {rho = :.2e} | rQ = {resQ[-1]:.3e}")
        break
else:
    rho = update_rho()
    Qs.update_shift(rho)
    rQ = Qs(v)
    resQ.append(np.linalg.norm(rQ/np.linalg.norm(v)))
    print(f"it {k+1:>2d} | {rho = :.2e} | rQ = {resQ[-1]:.3e}")

rateQ, r2Q = contraction_rate(resQ)
print()
print(f"{[f'{r:.3e}' for r in resQ]}")
print(f"Convergence rate (Q) = {rateQ:.4e} | R^2 = {r2Q:.3e}")

Imat = np.eye(nt)
F.update_v(v=Qs_inv.v)
Fmat = F(Imat)
Qmat = Q(Imat)
Jmat = J(Imat)
dFQ = Fmat - Qmat
dFJ = Fmat - Jmat
print()
print("Approximate fundamental solution:")
print(f"max(F-Q) = {np.max(np.abs(dFQ)):.3e}")
print(f"max(F-J) = {np.max(np.abs(dFJ)):.3e}")
