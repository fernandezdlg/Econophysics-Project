#==============================================================================
# Was working on here, but found a library in python
#==============================================================================

import numpy as np



time = np.linspace(0, len(data))

def autocovariance(hurst, k):
    return 0.5 * (data(k - 1) ** (2 * hurst) -
                  2 * data(k) ** (2 * hurst) +
                  data(k + 1) ** (2 * hurst))


C = np.matrix(np.zeros([n, n]))
for i in range(n):
    for j in range(i + 1):
        C[i, j] = autocovariance(hurst, i - j)
    
# Cholesky decomposition
M = np.linalg.cholesky(C)

def fgn():
    # Fractional Gaussian Noise
    scale = (1.0 * range(data) / len(data)) ** hurst
    gn = np.random.normal(0.0, 1.0, n)

    if hurst == 0.5:
        return gn * scale
    else:
        return M * scale

