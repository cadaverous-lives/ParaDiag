import numpy as np
from scipy.stats import linregress

def theta_method(theta, z):
    return (1 + (1-theta)*z)/(1 - theta*z)

def geomean(x):
    return np.exp(np.mean(np.log(x)))

def random_array(shape, dtype, seed=None):
    if seed:
        np.random.seed(seed)
    x = np.empty(shape, dtype=dtype)
    if dtype is complex:
        x[:].real = np.random.random_sample(x.shape)
        x[:].imag = np.random.random_sample(x.shape)
    else:
        x[:] = np.random.random_sample(x.shape)
    return x

def vartheta(phi, n, tol1=1e-8):
    phi = abs(phi)
    if np.isclose(phi, 1, rtol=tol1, atol=tol1):
        return n
    else:
        eta = (1 - phi**n)/(1 - phi)
        return min(n, eta)
        
def eta_expected(phi, phi_hat, n, alpha):
    dphi = np.max(np.abs(phi - phi_hat))
    phihat = np.max(np.abs(phi_hat))
    vtheta = vartheta(phihat, n)
    eta = (vtheta*dphi + alpha)/(1 - alpha)
    return eta, vtheta, dphi

def contraction_rate(res):
    if len(res) == 1:
        return 1, 1
    its = np.arange(len(res))
    res /= res[0]
    logc, _, rval, _, _ = linregress(its, np.log(res))
    return np.exp(logc), rval*rval
