import numpy as np

class Reduction(object):
    """Callable modal reduction object.
    Example usage:
    xreduction = Reduction(X), X shape [n_features, n_samples], make sure X is 
    zero-mean
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
        if(np.linalg.norm(np.mean(self._X,axis=1))>1e-5):
            print "Make sure columns of X is zero-mean"
        nStock = len(self._X[:,0])
        nTime = len(self._X[0,:])
        if n_components is None or n_components > min(nStock, nTime):
            n_components = min(3, nStock, nTime)
        U, s, Vt = np.linalg.svd(self._X)
        energy_content = np.sum(s[:n_components]**2)/np.sum(s**2)
        ux = U[:,:n_components]
        at = ux.T.dot(self._X)
        return ux, at, energy_content
        
    def ICA(self):
        """
        Independent component analysis(ICA) of data in matrix X
        """
        if(np.norm(np.mean(self._X,axis=1))>1e-5):
            print "Make sure columns of X is zero-mean"
        nStock = len(self._X[:,0])
        nTime = len(self._X[0,:])
        return 0
        
    def DMD(self, n_components=None):
        """
        Dynamic mode decomposition(DMD) of time series data x(k), find square 
        matrix A such that x(k+1) = Ax(k). Find eigendecomposition of A, and 
        corresponding DMD modes, and DMD eigenvalues.
        
        """
        nStock = len(self._X[:,0])
        nTime = len(self._X[0,:])
        if n_components is None or n_components > min(nStock, nTime):
            n_components = min(3, nStock, nTime)
        X, Y = self._X[:,:-1], self._X[:,1:]
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        energy_content = np.sum(s[:n_components]**2)/np.sum(s**2)
        Ur = U[:,:n_components]
        Vr = Vt.T[:,:n_components]
        Sr = np.diag(s[:n_components])
        A = Ur.T.dot(Y).dot(Vr).dot(np.linalg.pinv(Sr))
        evals, evecs = np.linalg.eig(A)
        modes = Ur.dot(evecs)
        return evals, modes, energy_content
