import numpy as np
from scipy import linalg


def simplex_equiangular_tight_frame(k, d):
    assert k <= d + 1, 
    A = np.random.randn(k, d)
    U, _ = linalg.polar(A) 
    M = np.sqrt(k / (k - 1)) * (np.eye(k) - np.ones(k) / k) @ U  
    return M

k = 100
d = 128

M = simplex_equiangular_tight_frame(k, d)
np.save('{}_{}_target.npy'.format(k, d),M)
