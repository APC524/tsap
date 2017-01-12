
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import scipy.linalg as Linalg
from scipy.stats import multivariate_normal as mvn


class Cluster(object):
    def __init__(self, X):
        """Return a new object to cluster data based on selected clutering
        algorithm.
        Example usage: clusterObj = Cluster(X)
        X:     numpy array, shape (n_samples, n_features)"""
        self._X = X
        self._nsample = X.shape[0]
        self._nfeature = X.shape[1]


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
                    label[i] is the code or index of the centroid the i-th
                    observation is closest to.
        clusters : identity of the data point in the cluster
        """

        # Initialize K-means algorithm by randomly sampling k points
        idx = np.random.choice(range(self._nsample), size= nClusters, replace=False)
        centroid = self._X[idx, :]

        # fix centrod, get the label of each data point
        old_clusters, old_labels  = self.assign_label( Centers = centroid )


        # set flag of convergence
        flag_converge = False
        nite = 0 # iteration counter

        while not flag_converge:
            nite = nite + 1
            if nite > maxIter:
                raise RuntimeError('Exceeds maximum number of iterations')
            # obtain new estimate of clusters
            for i in range(nClusters):
                class_index = old_clusters[i]
                centroid[i,:] = np.mean(self._X[class_index], axis = 0)
                new_clusters, new_labels = self.assign_label( Centers = centroid)
                if old_labels == new_labels:
                    flag_converge = True
                old_labels = new_labels
                old_clusters = new_clusters

        clusters = new_clusters
        labels = new_labels
        return centroid, labels, clusters




    def H_clustering(self, nClusters):
        """
        Performe hierarchical clustering
        """

        # construct hierarchical clustering matrix
        Z = linkage(self._X, metric='euclidean', method = 'ward')
        # obtain labels
        labels = fcluster(Z, nClusters, criterion='maxclust')
        clusters = {}
        centroid = np.zeros( (nClusters, self._nfeature) )
        for i in range(nClusters):
            class_index  = np.where( labels == i)[0]
            clusters[i]= class_index
            centroid[i,:] = np.mean(self._X[class_index, :], axis = 0)
        return centroid, labels, clusters




    # Gaussian mixture clustering using EM algorithm
    def Gaussian_mixture(self, nClusters, max_iter = 300):
        # Initialize EM algorithm by randomly sampling k points as centers
        idx = np.random.choice(range(self._nsample), size= nClusters, replace=False)
        centroid = self._X[idx, :] # initial mean vectors

        # initialize the covariance matrices for each gaussians
        Sigma= [np.eye(self._nfeature)] * nClusters

        # initialize the probabilities/weights for each gaussians
        w = [1./nClusters] * nClusters


        # responsibility matrix is initialized to all zeros
        # we have responsibility for each of n points for eack of k gaussians
        R = np.zeros((self._nsample, nClusters))

        ### log_likelihoods
        log_likelihoods = []

        P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-self._X.shape[1]/2.) \
                * np.exp(-.5 * np.einsum('ij, ij -> i',\
                        self._X  - mu, np.dot(np.linalg.inv(s) , (self._X - mu).T).T ) )

        # Iterate till max_iters iterations
        while len(log_likelihoods) < max_iter:

            # E - Step

            ## Vectorized implementation of e-step equation to calculate the
            ## membership for each of k -gaussians
            for k in range(nClusters):
                R[:, k] = w[k] * P(centroid[k], Sigma[k])

            ### Likelihood computation
            log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))

            log_likelihoods.append(log_likelihood)

            ## Normalize so that the responsibility matrix is row stochastic
            R = (R.T / np.sum(R, axis = 1)).T

            ## The number of datapoints belonging to each gaussian
            N_ks = np.sum(R, axis = 0)


            # M Step
            ## calculate the new mean and covariance for each gaussian by
            ## utilizing the new responsibilities
            for k in range(nClusters):

                ## means
                centroid[k] = 1. / N_ks[k] * np.sum(R[:, k] * self._X.T, axis = 1).T
                x_mu = np.matrix(self._X  - centroid[k])

                ## covariances
                Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))

                ## and finally the probabilities
                w[k] = 1. / self._nsample * N_ks[k]
            # check for onvergence
            if len(log_likelihoods) < 2 : continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < 1e-6: break

        clusters, labels  = self.assign_label( Centers = centroid )


        return clusters, labels, centroid, Sigma, w


    def Spectral(self, nClusters = 5, cluster_metric = 'euclidean', sigma = 0.05 ):
        """ Spectral Clustering
            cluster_metric is the metric used to compute the affinity matrix

            sigma is the standard deviation used in the Gaussian kernel

        """

        num_cls = nClusters
        # compute the affinity matrix
        aff_mat = squareform( pdist (self._X, metric = cluster_metric)/sigma )
        np.fill_diagonal(  aff_mat, 1)
        aff_mat = 1 / aff_mat
        np.fill_diagonal(  aff_mat, 0)

        # construct D^{-1/2} by taking the square root of the sum of column of A
        #print(np.sqrt( np.sum(aff_mat, axis = 0)))
        D_mat = np.diag( 1 / np.sqrt( np.sum(aff_mat, axis = 0)) )

        # graph Laplacian, an n by n matrix
        L = np.dot( np.dot(D_mat, aff_mat), D_mat )

        # Now that we have the graph Laplacian, spectral clustering does eigen decomposition on L and obtain the first k eigenvectors
        _ , X_embed =  Linalg.eigh(L, eigvals = (self._nsample - nClusters, self._nsample-1))

        # X_embed can be viewd as the embedding of data

        # normalize the rows of X_embed to unit norm
        row_norm = np.linalg.norm( X_embed, axis = 1).reshape( self._nsample, 1)
        Y = np.divide(X_embed, row_norm)  # n by k matrix, feed to K means
        model1 = Cluster(Y)
        _, labels, clusters = model1.kMeans(nClusters = num_cls)
        return labels, clusters, X_embed
