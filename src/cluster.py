
import numpy as np

class Cluster(object):
    def __init__(self, X):
        """Return a new object to cluster data based on selected clutering
        algorithm.
        Example usage: clusterObj = Clustering(X)
        X:     numpy array, shape (n_samples, n_features)"""
        self._X = X
        self._nsample = X.shape[0]
        self._nfeature = X.shape[1]


    ###################################################################


    def assign_label(self, Centers):
        """ Assign labels to the data points
            Input:
                self
                Centers:  numpy array, shape (n_clusters, n_features) the centers of each cluster
            Output:
                clusters: the index of data points in each class
                labels: the label of each class

        """
        numClusters = Centers.shape[0]
        clusters = {}
        labels = []

        for sample_idx in range(self._nsample):
            x = self._X[sample_idx, :]
            dist = []

            label_x = min( [(i, np.linalg.norm( x-Centers[i,:] ) ) for i in range(numClusters)], key = lambda t:t[1])[0]
            try:
                clusters[label_x].append(sample_idx)
            except KeyError:
                clusters[label_x] = [sample_idx]

            labels.append(label_x)

        return clusters, labels


    def kMeans(self, nClusters, maxIter=300):
        """
        K-means clustering algorithm.

        Function usage: kMeans(nClusters, maxIter, nInit)

        Inputs:
        nClusters : int
                    The number of clusters to form as well as the number of
                    centroids to generate.
        maxIter : int, optional, default 300
                    Maximum number of iterations of the k-means algorithm to run.

        Returns:
        centroid :  float ndarray with shape (k, n_features)
                    Centroids found at the last iteration of k-means.
        label : integer ndarray with shape (n_samples,)
                    label[i] is the code or index of the centroid the i’th
                    observation is closest to.
        inertia : float
                    The final value of the inertia criterion (sum of squared
                    distances to the closest centroid for all observations
                    in the training set)."""


        # Initialize K-means algorithm by randomly sampling k points
        idx = np.random.choice(range(self._nfeature), size= nClusters, replace=False)
        centroid = self._X[idx, :]

        # fix centrod, get the label of each data point
        old_clusters, old_labels  = assign_label(self, Centers = centroid )


        # set flag of convergence
        flag_converge = False
        nite = 0 # iteration counter

        while not flag_converge:
            nite = nite + 1
            if nite > maxIter:
                raise RuntimeError('Exceeds maximum number of iterations')
            # obtain new estimate of clusters
            for i in range(nClusters):
                class_index = cluster[i]
                centroid[i,:] = np.mean(self._X[class_index], axis = 0)
                new_clusters, new_labels = assign_label(self, Centers = centroid)
                if old_labels == new_labels:
                    flag_converge = True
                old_labels = new_labels
        clusters = new_clusters
        labels = new_labels
        return centroid, labels, clusters



        ###################################################################


        import scipy.cluster.hierarchy as hierarchy
        def H_clustering(self, nClusters, cluster_metric = 'euclidian', linkage_method = 'ward'):
            # construct hierarchical clustering matrix
            Z = hierarchy.linkage(self._X, metric=cluster_metric, method = linkage_method)
            # obtain labels
            labels = fcluster(Z, nClusters, criterion='maxclust')
            clusters = {}
            centroid = np.zeros(nClusters, self._nfeature)
            for i in range(nClusters):
                class_index  = np.where( labels == i)[0]
                clusters[i].append(class_index)
                centroid[i,:] = np.mean(self._X[class_index, :], axis = 0)
            return centroid, labels, clusters



        ###################################################################
        # Gaussian mixture clustering using EM algorithm
        def Gaussian_mixture(self, nClusters):
            # Initialize EM algorithm by randomly sampling k points as centers
            idx = np.random.choice(range(self._nfeature), size= nClusters, replace=False)
            centroid = self._X[idx, :] # initial mean vectors
            Sigma = np.zeros(k, self._nfeature, self._nfeature)
            cov_init = np.cov(self._X.T) # nfeature by nfeature matrix
            for i in range(nClusters):
                Sigma[i] = cov_init

        ###################################################################
        from scipy.spatial.distance import pdist, squareform
        import scipy.linalg as Linalg
        def Spectral(self, nClusters, cluster_metric = 'euclidian', sigma = 0.05 ):
            """ Spectral Clustering
                cluster_metric is the metric used to compute the affinity matrix

                sigma is the standard deviation used in the Gaussian kernel

            """

            # compute the affinity matrix
            dist_mat = squareform( pdist (self._X, metric = cluster_metric)/sigma )
            aff_mat = np.fill_diagonal( np.exp( - dist_mat), 0 )

            # construct D^{-1/2} by taking the square root of the sum of column of A
            D_mat = np.diag( 1 / np.sqrt(np.sum(A, axis = 0)) )

            # graph Laplacian, an n by n matrix
            L = np.dot( np.dot(D_mat, aff_mat), D_mat )

            # Now that we have the graph Laplacian, spectral clustering does eigen decomposition on L and obtain the first k eigenvectors
            _ , X_embed =  Linalg.eigh(L, eigvals = (self._nsample - nClusters, self._nsample-1))

            # X_embed can be viewd as the embedding of data

            # normalize the rows of X_embed to unit norm
            row_norm = np.linalg.norm( X_embed, axis = 1).reshape( self._nsample, 1)
            Y = np.divide(X_embed, row_norm)  # n by k matrix, feed to K means
            -, labels, clusters = kMeans(Y, nClustres)
            return labels, clusters, X_embed
