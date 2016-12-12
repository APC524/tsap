# -*- coding: utf-8 -*-
from sklearn import cluster
from sklearn.mixture import GaussianMixture

class Clustering(object):
    def __init__(self, X):
        """Return a new object to cluster data based on selected clutering 
        algorithm.
        Example usage: clusterObj = Clustering(X)
        X:       data array, shape (n_samples, n_features)"""
        self._X = X
        
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
        centroid : float ndarray with shape (k, n_features)
                    Centroids found at the last iteration of k-means.
        label : integer ndarray with shape (n_samples,)
                    label[i] is the code or index of the centroid the i’th 
                    observation is closest to.
        inertia : float
                    The final value of the inertia criterion (sum of squared 
                    distances to the closest centroid for all observations 
                    in the training set)."""
        return cluster.k_means(self._X, n_clusters=nClusters, max_iter=maxIter, n_init=nInit)
        
    def descan(self, EPS, minSamples):
        """
        Perform DBSCAN clustering from vector array or distance matrix.
        
        Function usage: descan(EPS, minSamples)
        
        Inputs:
        eps : float, optional
                The maximum distance between two samples for them to be 
                considered as in the same neighborhood.
        min_samples : int, optional
                The number of samples (or total weight) in a neighborhood 
                for a point to be considered as a core point. 
                This includes the point itself.
                    
        Returns:
        core_samples : array [n_core_samples]
                        Indices of core samples.
        labels : array [n_samples]
                        Cluster labels for each point. Noisy samples are 
                        given the label -1."""
        return cluster.descan(self._X, eps=EPS,min_samples=minSamples)
        
    def meanShift(self, band_width):
        """
        Perform mean shift clustering of data using a flat kernel.
        
        Function usage: meanShift()
        
        Inputs:
        band_width : float, optional
                    Kernel bandwidth. If bandwidth is not given, it is determined
                    using a heuristic based on the median of all pairwise distances.
                    
        Returns:
        cluster_centers : array, shape=[n_clusters, n_features]
                            Coordinates of cluster centers.
        labels : array, shape=[n_samples]
                    Cluster labels for each point."""
        return cluster.mean_shift(self._X, bandwidth=band_width)
        
    def gaussianMixture(self, nComponents=1, maxIter=100, TOL=1e-3):
        """
        The GaussianMixture object implements the expectation-maximization (EM) 
        algorithm for fitting mixture-of-Gaussian models.
        
        Function usage:
            
        Inputs:
        nComponents : int, defaults to 1.
                        The number of mixture components.
                        
        maxIter : int, defaults to 100.
                    The number of EM iterations to perform.

        TOL : float, defaults to 1e-3.
                The convergence threshold. EM iterations will stop when the 
                lower bound average gain is below this threshold
        """
        gmObj = GaussianMixture(n_components=nComponents, max_iter=maxIter, tol=TOL)
        gmObj.fit(self._X)
        return gmObj.predict(self._X)