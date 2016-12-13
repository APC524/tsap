import numpy as np

class Reduction(object):
    """Callable modal reduction object.
    Example usage:
    xreduction = Reduction(x), x shape [n_features, n_samples]
    ux, at, energy_content = xreduction.PCA(n_components=3)
    """
    def __init__(self, X):
        self._X = X
        
    def PCA(self, n_components=None):
        """
        Principal component analysis of data in matrix
        Inputs:
        n_components: integer, number of principal components
        Returns:
        ux: principal components
        at: principal components coefficients
        energy_content: energy content percentage in the principal components
        """
        x = np.copy(self._X)
        nStock = len(x)
        nTime = len(x)
        if n_components is None or n_components > min(nStock, nTime):
            n_components = min(3, nStock, nTime)
        U, s, Vt = np.linalg.svd(x)
        energy_content = np.sum(s[:n_components]**2)/np.sum(s**2)
        ux = U[:,:n_components]
        at = ux.T.dot(x)
        return ux, at, energy_content
        
    def ICA(self, n_components=None):
        """
        Independent component analysis(ICA) of data in matrix
        """
        return 0
        
    def DMD(self, n_components=None):
        """
        Dynamic mode decomposition(DMD) of time series data x(k), find square 
        matrix A such that x(k+1) = Ax(k). Find eigendecomposition of A, and 
        corresponding DMD modes, and DMD eigenvalues.
        
        """
        x = np.copy(self._X)
        nStock = len(x)
        nTime = len(x)
        if n_components is None or n_components > min(nStock, nTime):
            n_components = min(3, nStock, nTime)
        U, s, Vt = np.linalg.svd(x)
        energy_content = np.sum(s[:n_components]**2)/np.sum(s**2)
        Ur = U[:,:n_components]
        A = Ur.T.dot(x[:,1:]).dot(np.linalg.pinv(Ur.T.dot(x[:,:-1])))
        evals, evecs = np.linalg.pinv(A)
        modes = Ur.dot(evecs)
        return evals, modes, energy_content

