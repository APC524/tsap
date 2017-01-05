import numpy as np
import math

def get_return(X, option = 0):
	"""get_return: return the daily change of given time series, usually it's about price series of stocks
           Input: 
                 X: the input time series, each row is about one stock. For one stock, X is a row vector. Each column corresponds to the price at a time step
                 option: set the type of the return. The default is option = 0, in which it gives discrete return. If option is not 0, then it will calculate the log return
           Output:
                  rt: the daily change of given time series"""

    """the number of samples, usually it's about how many stocks we have """
    num_data = X.shape[0]
    """the length of time"""
    input_dim = X.shape[1] 

    if option == 0:
        rt = (X[:,1:input_dim] - X[:,0:input_dim-1]) / X[:,0:input_dim-1]
    else:
        """to get log return"""
        rt = math.log(X[:,1:input_dim]/ X[:,0:input_dim-1])
    
    return rt 