import numpy as np
import math
from basemodel import base

class AR(base):    

    def __init__(self, lag, phi, sigma, intercept):
        """lag, phi, sigma, intercept is the parameter of AR"""
        self._lag = lag
        self.params = {}
        self.params['phi'] = phi 
        self.params['sigma'] = sigma
        self.params['intercept'] = intercept

    def loss(self, X):
        """X is dataset, right now X is a row vector"""
        """phi is a column vector, and we need to make it into matrix form"""

        input_dim = X.shape[1]    
        lag = self._lag    
        phi = self.params['phi']
        sigma = self.params['sigma']
        intercept = self.params['intercept']

        
        
        loglikelihood = 0
        grad_phi = np.zeros((lag,1))
        grad_intercept = 0
        grad_sigma = 0

        for i in range(input_dim - lag):
            """np.fliplr can only flip matrix form vector"""
            temp = intercept + np.dot(np.matrix(X[0,i:(i+lag+1)]), np.transpose(np.fliplr(np.hstack(([[-1.0]],phi)))))
            loglikelihood -= temp**2
            grad_phi -= temp * (np.fliplr(np.matrix(X[0,i:(i+lag)])))
            grad_intercept -= temp
            grad_sigma += temp**2

        loglikelihood = loglikelihood / (2 * sigma**2)
        loglikelihood += (lag - input_dim) / 2 * math.log(sigma**2)
        grad_phi = (grad_phi / (sigma**2)).T
        grad_intercept = grad_intercept / (sigma**2)
        grad_sigma = grad_sigma / (sigma**3)
        grad_sigma += (lag - input_dim) / (sigma)

        grads = {} 
        """grad_phi is a column vector"""
        grads['phi'] = grad_phi   
        grads['intercept'] = grad_intercept 
        grads['sigma'] = grad_sigma


        return loglikelihood, grads


