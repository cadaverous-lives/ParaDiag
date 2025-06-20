import numpy as np
from scipy.sparse import diags_array
from scipy.linalg import matmul_toeplitz, solve_toeplitz
from scipy.fft import fft, ifft
import scipy.sparse.linalg as spla

__all__ = [
    "ConstantCoefficientOperator",
    "CirculantOperator",
    "VariableCoefficientOperator",
    "PeriodicOperator",
    "AllAtOnceOperator",
    "FloquetTransformOperator",
]


class ConstantCoefficientOperator(spla.LinearOperator):
    def __init__(self, phi, n, inverse=False):
        self.dtype = phi.dtype
        self.shape = tuple((n, n))

        self.col = np.zeros(n, dtype=self.dtype)
        self.row = np.zeros(n, dtype=self.dtype)

        self.row[0] = 1
        self.col[0] = 1
        self.col[1] = -phi

        self.mat = tuple((self.col, self.row))

        self._op = solve_toeplitz if inverse else matmul_toeplitz
        self.inverse = inverse

    def _matvec(self, v):
        return self._op(self.mat, v)


class CirculantOperator(spla.LinearOperator):
    def __init__(self, phi, n, alpha=1., inverse=False):
        if np.iscomplexobj(alpha):
            self.dtype = complex
        else:
            self.dtype = np.dtype(type(phi))
        self.alpha = alpha
        self.phi = phi
        col = np.zeros(n, dtype=self.dtype)
        col[:2] = [1, -phi]
        self.shape = tuple((n, n))

        # fft weighting
        self.gamma = alpha**(np.arange(n)/n)

        # eigenvalues
        self.eigvals = fft(col*self.gamma, norm='backward')

        self._scale = 1/self.eigvals if inverse else self.eigvals
        self.inverse = inverse

    def _to_eigvecs(self, v):
        return fft(v*self.gamma, norm='ortho')

    def _from_eigvecs(self, v):
        return ifft(v, norm='ortho')/self.gamma

    def _matvec(self, v):
        return self._from_eigvecs(self._to_eigvecs(v.flatten())*self._scale)


class VariableCoefficientOperator(spla.LinearOperator):
    def __init__(self, phis, n, inverse=False):

        # we're not periodic so possibly ignore last propagator
        self.phis = phis if len(phis) == n-1 else phis[:-1]

        self.inverse = inverse

        mat = diags_array(
            [np.ones(n), -self.phis],
            offsets=[0, -1], shape=[n, n])

        if inverse:
            self.mat = mat.tocsr()
        else:
            self.mat = mat

        self.dtype = self.mat.dtype
        self.shape = self.mat.shape

    def _matvec(self, v):
        if self.inverse:
            return spla.spsolve_triangular(
                self.mat, v, lower=True,
                unit_diagonal=True)
        else:
            return self.mat @ v


class PeriodicOperator(spla.LinearOperator):
    def __init__(self, phis, n, alpha=1., inverse=False):
        if np.iscomplexobj(alpha):
            self.dtype = complex
        else:
            self.dtype = phis.dtype
        self.shape = tuple((n, n))

        self.phis = phis
        self.alpha = alpha
        self.inverse = inverse

        # include the periodic element
        self.mat = diags_array(
            [np.ones(n, dtype=self.dtype),
             -self.phis[:-1],
             -alpha*self.phis[-1]],
            offsets=[0, -1, n-1], shape=[n, n])

    def _matvec(self, v):
        if self.inverse:
            return spla.spsolve(self.mat, v)
        else:
            return self.mat @ v


def AllAtOnceOperator(phi, n, alpha=None, inverse=False):
    if isinstance(phi, np.ndarray):
        if alpha is None or alpha == 0:
            return VariableCoefficientOperator(phi, n, inverse=inverse)
        else:
            return PeriodicOperator(phi, n, alpha=alpha, inverse=inverse)
    else:
        if alpha is None or alpha == 0:
            return ConstantCoefficientOperator(phi, n, inverse=inverse)
        else:
            return CirculantOperator(phi, n, alpha=alpha, inverse=inverse)


class FloquetTransformOperator(spla.LinearOperator):
    def __init__(self, phibar, n, v, alpha=1, inverse=False):

        self.n = n
        self.v = v
        self.phibar = phibar

        # scaling for fundamental solution
        self.phi_n = phibar**np.arange(n)

        self.circulant_mat = CirculantOperator(
            phibar, n, alpha=alpha, inverse=inverse)

        self.dtype = self.circulant_mat.dtype
        self.shape = self.circulant_mat.shape

    def update_v(self, v=None, vhat=None):
        if v is None and vhat is None:
            raise ValueError("Can only provide one of transform (v) or fundamental solution (vhat), not both")
        if v is not None:
            self.v[:] = v
        else:
            self.v[:] = vhat/self.phi_n

    def _to_circulant(self, y):
        return y/self.v

    def _from_circulant(self, y):
        return y*self.v

    def _matvec(self, y):
        return (
            self._from_circulant(
                self.circulant_mat.matvec(
                    self._to_circulant(y.flatten()))))
