
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


    def kMeans(self, nClusters, maxIter=300, nInit=10):
        """
        K-means clustering algorithm.

        Function usage: kMeans(nClusters, maxIter, nInit)

        Inputs:
        nClusters : int
                    The number of clusters to form as well as the number of
                    centroids to generate.
        maxIter : int, optional, default 300
                    Maximum number of iterations of the k-means algorithm to run.
        nInit : int, optional, default: 10
                    Number of time the k-means algorithm will be run with
                    different centroid seeds. The final results will be the
                    best output of n_init consecutive runs in terms of inertia.

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
        idx = nrd.choice(range(self._nfeature), size= nClusters, replace=False)
        centroid = self._X[idx, :]

        # fix centrod, get the label of each data point
        old_clusters, old_labels  = assign_label(self, Centers = centroid )


        # set flag of convergence
        flag_converge = False

        while not flag_converge:
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
