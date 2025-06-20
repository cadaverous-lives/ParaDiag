import numpy as np
from scipy.sparse.linalg._isolve.utils import make_system
from scipy.sparse.linalg._isolve.iterative import _get_atol_rtol
from utils import contraction_rate

__all__ = ["richardson", "floquet_continuation"]


def optimal_omega(A, M, z):
    # z = M(r)
    y = M(A(z))
    rdot = np.dot(z, y)
    abr = np.dot(y, y)
    omega = rdot/abr 
    print(f"omega_opt = {omega:.2e}")
    return omega

def get_omega(omega, A, M, z):
    if omega == 'optimal':
        return optimal_omega(A, M, z)
    else:
        return omega


def richardson(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None,
               M=None, callback=None, omega=1.):
    """Use Richardson iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        ``A`` must represent a hermitian, positive definite matrix.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, ndarray, LinearOperator}
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk, r, pr, res, k), where xk is the current solution
        vector, r is the current residual vector ``b-Ax``, pr is the
        preconditioned residual vector, res is the norm of r, and ``k`` is
        the iteration index.
    omega : weighting factor on solution update.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
    k : integer
        Number of iterations
    res : float
        Final residual norm
    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    bnrm2 = np.linalg.norm(b)

    atol, _ = _get_atol_rtol('cg', bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0

    if maxiter is None:
        maxiter = len(b)*10

    matvec = A.matvec
    psolve = M.matvec

    for iteration in range(maxiter):

        r = b - matvec(x)
        r2 = np.linalg.norm(r)

        if r2 < atol:  # Are we done?
            return postprocess(x), 0, iteration, r2

        pr = psolve(r)
        x += pr*get_omega(omega, A, M, pr)

        if callback:
            callback(x, r, pr, r2, iteration)

    else:
        return postprocess(x), maxiter, iteration, r2


def floquet_continuation(J, Qinv, b, x0=None,
                         omega_x=1, omega_v=1,
                         update='multiplicative',
                         maxiter=10, atol=1e-5):
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    if not update in ('multiplicative', 'additive'):
        raise ValueError("update type must be 'multiplicative' or 'additive'")

    v = Qinv.v.copy()*Qinv.phi_n
    v0 = np.zeros_like(Qinv.v)
    v0[0] = Qinv.v[0]

    resJ = []
    resQ = []

    for iteration in range(maxiter):

        rQ = v0 - J(v)
        dv = Qinv(rQ)
        resQ.append(np.linalg.norm(rQ))

        if update == 'multiplicative':
            v += dv*get_omega(omega_v, J, Qinv, dv)
            v[0] = v0[0]
            Qinv.update_v(vhat=v)

        rJ = b - J(x)
        dx = Qinv(rJ)
        resJ.append(np.linalg.norm(rJ))

        x += dx*get_omega(omega_x, J, Qinv, dx)

        if update == 'additive':
            v += dv*get_omega(omega_v, J, Qinv, dv)
            v[0] = v0[0]
            Qinv.update_v(vhat=v)

        vnorm = np.linalg.norm(Qinv.v)
        xnorm = np.linalg.norm(x)

        print(f"it {iteration:>2d} | rJ = {resJ[-1]:.3e} | rQ = {resQ[-1]:.3e}")# | {xnorm = :.2e} | {vnorm = :.2e}")

        if resJ[-1] < atol:
            resQ.append(np.linalg.norm(v0 - J(v)))
            resJ.append(np.linalg.norm(b - J(x)))
            print(f"it {iteration+1:>2d} | rJ = {resJ[-1]:.3e} | rQ = {resQ[-1]:.3e}")# | {xnorm = :.2e} | {vnorm = :.2e}")
            break

    rateJ, r2J = contraction_rate(resJ)
    rateQ, r2Q = contraction_rate(resQ)
    print()
    print(f"Convergence rate (J) = {rateJ:.4e} | R^2 = {r2J:.3e}")
    print(f"Convergence rate (Q) = {rateQ:.4e} | R^2 = {r2Q:.3e}")
