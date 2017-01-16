import numpy as np

class Reduction(object):
    """Callable modal reduction object.
    Example usage:
    xreduction = Reduction(X), X shape [n_features, n_samples], make sure X is 
    zero-mean
    xmean, ux, at, energy_content = xreduction.PCA(n_components=3)
    """
    def __init__(self, X):
        self._X = X

    def PCA(self, n_components=None):
        """
        Principal component analysis (PCA) of data in matrix
        Inputs:
        n_components: integer, number of principal components
        Returns:
        ux: principal components
        at: principal components coefficients
        energy_content: energy content percentage in the principal components
        """
        if np.linalg.norm(np.mean(self._X,axis=1))>1e-5:
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
        
    def ICA(self, n_components, gfunc='logcosh', tol=1e-4, max_iter=200):
        """
        Independent component analysis(ICA) of data in matrix X
        Inputs:
        n_components: integer, number of independent components
        gfunc: string, 'logcosh' or 'exp', default 'logcosh', Non-gaussian function
        tol: float, tolerance of iteration, default 1e-4
        max_iter: integer, maximum iteration steps, default 200
        Returns:
        Ex: array, mean of data
        T: array [n_features, n_features], whitening matrix, st, xtilde = Tx
        A: array [n_features, n_components], mixing matrix, st, xtilde = As
        W: array [n_components, n_features], orthogonal rows, unmixing matrix, st, W = inv(A), s = W*xtilde
        S: array, [n_components, n_samples], source data, st, S = W*Xtilde
        """
        # preprocessing, centering
        def _centering(X):
            x = X.copy()
            Ex = np.mean(x,axis=1)
            for i in range(len(self._X[0,:])):
                x[:,i] -= Ex
            return x, Ex
        # whitening
        def _whitening(X):
            x = X.copy()
            Cov = x.dot(x.T)/len(x[0,:])
            d, E = np.linalg.eigh(Cov)
            T = E.dot(np.diag(1./np.sqrt(d))).dot(E.T)
            xtilde = T.dot(x)
            return xtilde, T
        # Non-gaussian function
        def _logcosh(x, a1=1):
            gx = np.tanh(a1*x)
            dgdx = a1*(1 - np.tanh(a1*x)**2)
            return gx, dgdx
        def _exp(x):
            exp = np.exp(-(x ** 2) / 2)
            gx = x * exp
            dgdx = (1 - x ** 2) * exp
            return gx, dgdx
        # decorrelate weights w
        def _Gram_Schmidt_decorrelate(w, W, j):
            w -= np.dot(np.dot(w, W[:j].T), W[:j])
            return w
        
        # Independent component analysis(ICA) of data in matrix X
        X = self._X.copy()
        n_features = len(X[:,0])
        # preprocessing
        X, Ex = _centering(X)
        X, T = _whitening(X)    
        # Non-gaussian function
        if gfunc == 'logcosh':
            g = _logcosh
        elif gfunc == 'exp':
            g = _exp
        # mixing matrix [orthogonal rows], st, W = inv(A), x = As, s = inv(A)x = W*x
        W = np.random.normal(0,1,(n_components, n_features))
        for j in range(n_components):
            w = W[j,:]
            w /= np.sqrt((w ** 2).sum())
            count_iter, delta = 0, 1
            while count_iter < max_iter and delta > tol:
                gx, dgdx = g(np.dot(w.T,X))
                w1 = (X * gx).mean(axis=1) - dgdx.mean() * w
                _Gram_Schmidt_decorrelate(w1, W, j)
                w1 /= np.sqrt((w1 ** 2).sum())
                delta = np.abs(np.abs((w1 * w).sum()) - 1)
                w = w1
                count_iter += 1
            W[j,:] = w
        # mixing matrix, A = inv(W) = W.T
        A = W.T
        # source matrix, s = W*xtilde
        S = W.dot(X)
        return Ex, T, A, W, S
        
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
