import numpy as np
import scipy.sparse.linalg as LA
class Reduction(object):
    """Callable modal reduction object.
    Example usage:
    xreduction = Reduction(x), x shape [n_features, n_samples]
    xmean, ux, at, energy_content = xreduction.PCA(n_components=3)
    """
    def __init__(self, X):
        self._X = X

    def PCA(self, n_components=None):
        """
        Inputs:
        n_components: integer, number of principal components
        Returns:
        xmean: mean of data
        ux: principal components
        at: principal components coefficients
        energy_content: energy content percentage in the principal components
        """
        x = np.copy(self._X)
        nStock = len(x)
        nTime = len(x)
        if n_components is None or n_components > min(nStock, nTime):
            n_components = min(3, nStock, nTime)
        xmean = np.mean(x, axis = 1)
        for i in range(nTime):
            x[:,i] = x[:,i] - xmean
        U, s, Vt = LA(x, k = n_components)
        # use scipy to compute only the leading singular vectors
        energy_content = np.sum(s[:]**2)/np.sum(s**2)
        ux = U[:,:n_components]
        at = ux.T.dot(x)
        return xmean, ux, at, energy_content

    def ICA(self):
        return 0
