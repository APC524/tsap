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
        num_data = X.shape[0]
        input_dim = X.shape[1]    
        lag = self._lag    
        phi = self.params['phi']
        sigma = self.params['sigma']
        intercept = self.params['intercept']
        
        loglikelihood = 0.0
        grad_phi = np.zeros((lag,1))
        grad_intercept = 0.0
        grad_sigma = 0.0

        for i in range(input_dim - lag):
            """np.fliplr can only flip matrix form vector"""
            temp = intercept + np.dot(X[0,i:(i+lag+1)], np.vstack((np.flipud(phi),-1.0)))
            loglikelihood -= temp**2
            grad_phi -= float(temp) * (np.fliplr(np.matrix(X[0,i:(i+lag)]))).T
            grad_intercept -= temp
            grad_sigma += temp**2

        loglikelihood = loglikelihood / (2 * sigma**2)
        loglikelihood += (lag - input_dim) / 2 * math.log(sigma**2)
        grad_phi = grad_phi / (sigma**2)
        grad_intercept = grad_intercept / (sigma**2)
        grad_sigma = grad_sigma / (sigma**3)
        grad_sigma += (lag - input_dim) / (sigma)

        grads = {} 
        """grad_phi is a column vector"""
        grads['phi'] = grad_phi   
        grads['intercept'] = grad_intercept 
        grads['sigma'] = grad_sigma


        return loglikelihood, grads

    def predict(self, X, nstep):
        lag = self._lag    
        phi = self.params['phi']
        sigma = self.params['sigma']
        intercept = self.params['intercept']
        input_dim = X.shape[1]


        pred_state = np.zeros((1,nstep))
        train = np.hstack((X[0,(input_dim-lag):input_dim], pred_state))
        for i in range(nstep):
            pred_state(0,i) = np.dot(train[(input_dim+i-lag):(input_dim+i)],phi) + intercept


class MA(base):
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

        loglikelihood=-input_dim/2*math.log(2*math.pi*sigma**2)

        autocov = np.zeros((lag+1,1))
        autocov[0]=sigma**2+np.dot(phi,phi)*sigma**2[0,0]
        for i in range(lag):
            autocov[i+1]=np.dot(phi[0:lag-i-2],phi[i+1:lag-1])*sigma**2[0,0]-phi[i]*sigma**2

        covmat=np.zeros((input_dim,input_dim))
        for i in range(input_dim):
            for j in range(i+1):
                if abs(i-j)<=lag:
                    covmat[i,j]=autocov[abs(i-j)]
                    covmat[j,i]=autocov[abs(i-j)]
        
        loglikelihood -= 0.5*math.log(abs(np.linalg.det(covmat)))+float(1)/2/sigma/sigma*np.matmul(np.matmul(np.transpose(X),inv(autocov)),X)[0,0]





        return loglikelihood


