import numpy as np
import math
from basemodel import base


"""class AR implements the AR model which has __init__ , loss and predict"""
"""__init__: initialize the model with lag phi sigma and intercept"""
"""loss: calculate the loglikelihood and get its gradient with respect to phi, sigma and intercept"""
"""predict: does the prediction. Given the sample, it predicts future prices """
class AR(base):    

    def __init__(self, lag, phi, sigma, intercept):
        """lag, phi, sigma, intercept is the parameter of AR"""
        """lag is time lag, phi is the coefficient with dimension lag*1, sigma is the common volatility, intercept is just intercept"""
        self._lag = lag
        self.params = {}
        self.params['phi'] = phi 
        self.params['sigma'] = sigma
        self.params['intercept'] = intercept

    def loss(self, X):
        """X is dataset, right now X is a row vector"""
        """phi is a column vector"""

        """the number of samples, usually it's about how many stocks we have """
        num_data = X.shape[0]
        """the length of time"""
        input_dim = X.shape[1] 

        """parameters"""   
        lag = self._lag    
        phi = self.params['phi']
        sigma = self.params['sigma']
        intercept = self.params['intercept']
        
        """initialization"""
        loglikelihood = 0.0
        grad_phi = np.zeros((lag,1))
        grad_intercept = 0.0
        grad_sigma = 0.0

        """get the expression of loglikelihood and its gradient with respect to phi, sigma and intercept"""
        for i in range(input_dim - lag):
            """np.flipud can flip the vector"""
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

        """grad_phi is a column vector"""
        grads = {} 
        grads['phi'] = grad_phi   
        grads['intercept'] = grad_intercept 
        grads['sigma'] = grad_sigma

        return loglikelihood, grads

    """predict: does the prediction. Given the sample, it predicts future prices. nstep: how many future prices you wanna predict """
    def predict(self, X, nstep):
        """X is a row vector"""
        lag = self._lag    
        phi = self.params['phi']
        sigma = self.params['sigma']
        intercept = self.params['intercept']
        input_dim = X.shape[1]

        """pred_state stores the predicted prices, it is a row vector """
        pred_state = np.zeros((1,nstep))
        train = np.hstack((X[0,(input_dim-lag):input_dim], pred_state))
        for i in range(nstep):
            pred_state(0,i) = np.dot(train[(input_dim+i-lag):(input_dim+i)],phi) + intercept

        return pred_state


"""max_drawdown is not in any class, it is a function in this file"""
"""Sometimes, we need to control the risk, so we calculate the largest distance that the price can drop down. This function will return a negative 
number whose absolute value is the largest distance between the peak and the trough"""
def max_drawdown(X):
    """the length of time"""
    l = X.shape[1]
    diff = np.zeros((l - 1,1))

    """get the difference of price series"""
    for i in range(l-1):
        diff[i] = X[i+1] - X[i]

    """dynamic programming"""
    temp_sum = 0
    min_num = 0
    for i in range(l-1):
        if temp_sum > 0:
            temp_sum = 0
        if temp_sum < min_num:
            min_num = temp_sum
    return min_num


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

        
        """initialization"""
        loglikelihood = 0
        grad_phi = np.zeros((lag,1))
        grad_intercept = 0
        grad_sigma = 0

        loglikelihood=-input_dim/2*math.log(2*math.pi*sigma**2)

        """Derive autocorrelation for likelihood function"""
        autocov = np.zeros((lag+1,1))
        autocov[0]=sigma**2+np.dot(phi,phi)*sigma**2[0,0]
        for i in range(lag):
            autocov[i+1]=np.dot(phi[0:lag-i-2],phi[i+1:lag-1])*sigma**2[0,0]-phi[i]*sigma**2

        """Derive the covariance matrix for likelihood function"""
        covmat=np.zeros((input_dim,input_dim))
        for i in range(input_dim):
            for j in range(i+1):
                if abs(i-j)<=lag:
                    covmat[i,j]=autocov[abs(i-j)]
                    covmat[j,i]=autocov[abs(i-j)]
        
        loglikelihood -= 0.5*math.log(abs(np.linalg.det(covmat)))+float(1)/2/sigma/sigma*np.matmul(np.matmul(np.transpose(X),inv(autocov)),X)[0,0]





        return loglikelihood


        
    """predict: does the prediction. Given the sample, it predicts future prices. nstep: how many future prices you wanna predict """
    def predict(self, X, nstep):
        """X is a row vector"""
        lag = self._lag    
        phi = self.params['phi']
        sigma = self.params['sigma']
        intercept = self.params['intercept']
        input_dim = X.shape[1]

        """pred_state stores the predicted prices, it is a row vector """
        pred_state = np.zeros((1,nstep))
        train = np.hstack((X[0,(input_dim-lag):input_dim], pred_state))
        for i in range(nstep):
            pred_state(0,i) = np.dot(train[(input_dim+i-lag):(input_dim+i)],phi) + intercept

        return pred_state


