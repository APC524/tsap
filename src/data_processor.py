import numpy as np
import math


def get_return(X, option = 0):
	"""get_return: return the daily change of given time series, usually it's about price series of stocks
       Input: 
             X: the input time series, each row is about one stock. For one stock, X is a row vector. Each column corresponds to the price at a time step
             option: set the type of the return. The default is option = 0, in which it gives discrete return. If option is not 0, then it will calculate the log return
       Output:
              rt: the daily change of given time series"""

	num_data = X.shape[0]
	"""the length of time"""
	input_dim = X.shape[1]

	if option == 0:
		rt = (X[:,1:input_dim] - X[:,0:input_dim-1]) / X[:,0:input_dim-1]
	else:
		"""to get log return"""
		rt = math.log(X[:,1:input_dim] / X[:,0:input_dim-1])

	return rt

###################################################################

def max_drawdown(X):
	"""max_drawdown: return the largest distance that the price can drop down. This function will return a negative 
                    number whose absolute value is the largest distance between the peak and the trough (in terms of percentage)
       Input:
             X: the time series, in our project it is price series. It is a numpy array, and a row vector
       Output:
              ratio: the maximum drawdown, which is a number"""
	
	

	"""price can not be smaller than 0"""
	l = X.shape[1]
	for i in range(l):
		if float(X[0,i]) < 0.0:
			print "Price must be positive !"
			exit(1)


	"""get the difference series"""
	diff = np.zeros((l-1,1))

	for i in range(l-1):
		diff[i] = X[0,i+1] - X[0,i]

	"""initialization"""
	temp_sum = 0.0
	min_num = 0.0
	max_num = 0.0

	min_index = 0
	max_index = 0

	"""get the peak and the trough"""
	for i in range(l-1):
		"""convert diff[i] to a number. This is necessary, otherwise max_num will copy temp_sum"""
		temp_sum += float(diff[i])

		if temp_sum < min_num:
			min_num = temp_sum
			min_index = i + 1

		if temp_sum > max_num:
			max_num = temp_sum
			max_index = i + 1

	ratio = (X[0,min_index] - X[0,max_index]) / X[0,max_index]

	return ratio

###################################################################


def get_indicator(X):
	"""get_indices: return the indicator for peak and trough. The peak and trough has the largest distance. The trough must be before the peak
       Input:
             X: the time series, in our project it is price series. It is a numpy array, and a row vector
       Output:
              indicator: a numpy array with numbers: 0, 1, -1, which is a row vector. 0 denotes nothing, 1 denote peak, -1 denote trough.
                         there's unique 1 and -1. However, if the peak is before the trough, then all the indicator is 0"""
	
	

	"""price can not be smaller than 0"""
	l = X.shape[1]
	for i in range(l):
		if float(X[0,i]) < 0.0:
			print "Price must be positive !"
			exit(1)


	"""get the difference series"""
	diff = np.zeros((l-1,1))

	for i in range(l-1):
		diff[i] = X[0,i+1] - X[0,i]

	"""initialization"""
	temp_sum = 0.0
	min_num = 0.0
	max_num = 0.0

	min_index = 0
	max_index = 0

	"""get the peak and the trough"""
	for i in range(l-1):
		"""convert diff[i] to a number. This is necessary, otherwise max_num will copy temp_sum"""
		temp_sum += float(diff[i])

		if temp_sum < min_num:
			min_num = temp_sum
			min_index = i + 1

		if temp_sum > max_num:
			max_num = temp_sum
			max_index = i + 1

	"""define indicator"""
	indicator = np.zeros((1,l))

	"""trough is before the peak"""
	if min_index < max_index:
		indicator[0,min_index] = 1.0
		indicator[0,max_index] = -1.0

	return indicator

###################################################################
