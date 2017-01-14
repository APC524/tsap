
import model 
import data_processor as dp
import math
import numpy as np

def signal_generation(X, window = 2):
	"""signal_generation: return the signal, when to buy, when to sell
		Input: 
			  X: time seires of prices, it's a numpy array and a row vector
			  window: the length of time period during which there's at most one trade
		Output:
			   signal: a numpy array and a row vector. If it is 0, nothing to be done. If it is 1,
			          then buy. If it is -1, then sell. In addition, buying is before selling.
	"""
	
	"""get the integer number of the window"""
	w = int(window)

	"""how long is the time series"""
	l = X.shape[1]

	if w < 2:
		print "The time between each trade is too short !"
		exit(0)

	"""At most, how many trades we can make"""
	number = int(math.floor(l / w))

	"""initialization of signal"""
	signal = np.zeros((1,l))
	
	"""record buy and sell signal into signal"""
	for i in range(number):
		"""have to convert X[0,i*window:(i+1)*window] into numpy array so that it can be input in dp.get_indicator"""
		Y = np.array([X[0,i*window:(i+1)*window]])
		Z = dp.get_indicator(Y)
		signal[0,i*window:(i+1)*window] = Z[0,:]

	return signal

###################################################################


def profit_loss(X, signal, money = 1.0):
	"""profit_loss: return the profit at any time with given future price and signal
		Input:
			  X: numpy array and a row vector. price series, this is future price
			  signal: numpy array and a row vector. this is future trading signal
			  money: number. this is the initial investment
		Output:
		       m: numpy array and a row vector. This records the money that we have at any future time"""


	"""price can not be smaller than 0"""
	l = X.shape[1]
	for i in range(l):
		if float(X[0,i]) < 0.0:
			print "Price must be positive !"
			exit(0)

	"""initialize the profit vector"""
	m = np.array([np.repeat(0.0,l)])
	m[0,0] = money

	"""position records current state. It takes 1, if we hold a stock, otherwise it takes 0"""
	position = signal[0,0]

	for i in range(l):
		
		"""calculate the profit at any time"""
		if i > 0:
			if position == 0.0:
				m[0,i] = m[0,i-1]
			else:
				m[0,i] = m[0,i-1] * X[0,i] / X[0,i-1]

		"""record the position status"""
		if signal[0,i] == 1.0:
			position = 1.0

		if signal[0,i] == -1.0:
			position = 0.0

	return m


def trade(X, Y, M, nstep, window, money = 1.0):
	"""trade: return the profit series. First we are given a fitted model, then we need to use this model 
	to forecast future returns based on historical returns, which is obtained by transfering historical prices 
	to historical returns. After that, we again transfer the returns to prices and then generate trading signal 
	based on predicted price. At last we calculate the profit based on the real price series and the trading signal """

	"""
		Input: 
				X: numpy array and row vector. Historical prices.
				Y: numpy array and row vector. Future prices right after X
				M: the fitted model from model.py
				nstep: an integer. How many time steps you want to predict
				window: an integer. How long we should trade
				money: a number. Initial wealth
		Output:
				profit: numpy array and row vector. profit series indicating how much money you have"""

	"""The forecast should not be longer than the future prices data"""			
	if nstep > Y.shape[1]:
		print "The prediction should not be more than the test data !"
		exit(0)

	"""The trading window should not be longer than predicted time steps"""
	if window > nstep:
		print "The period should not be longer than the predicted prices !"
		exit(1)

	"""get historical returns based on historical prices"""
	rt_X = dp.get_return(X)

	"""predict future returns based on model M"""
	pred_rt = M.predict(rt_X, nstep)

	"""transfer the future returns to future prices"""
	pred_price = dp.get_price(X[0,X.shape[1]-1], pred_rt)

	"""generate trading signals based on future prices"""
	signal = signal_generation(pred_price, window)

	"""get the profti time series"""
	profit = profit_loss(np.array([Y[0,0:nstep]]), signal, money)

	return profit

def rolltrade(X, M, l, nstep, window, money = 1.0):
	"""rolltrade: return the profit series. We repeat the following procedure. First we are given a fitted model, then we need to use this model 
	to forecast future returns based on historical returns, which is obtained by transfering historical prices 
	to historical returns. After that, we again transfer the returns to prices and then generate trading signal 
	based on predicted price. At last we calculate the profit based on the real price series and the trading signal """

	"""
		Input: 
				X: numpy array and row vector. Real prices.
				M: the fitted model from model.py
				l: the length of trading data
				nstep: an integer. Each time, how many time steps you want to predict
				window: an integer. How long we should trade
				money: a number. Initial wealth
		Output:
				profit: numpy array and row vector. profit series indicating how much money you have
				signal: numpy array and row vector. trading signal, 0 no trade, 1 buy signal, -1 sell signal
				out_pred_price: numpy array and row vector. prediced prices series """


	"""The trading window should not be longer than predicted time steps"""
	if window > nstep:
		print "The period should not be longer than the predicted prices !"
		exit(1)

	"""This is how many data we want to use to train the model"""
	train_length = l

	"""This is how many future steps we want to predict"""
	pred_length = nstep

	"""the length of data"""
	data_length = X.shape[1]

	"""how many predictions you make for the trading"""
	number_of_pred = int(math.floor((data_length - train_length) / pred_length))

	"""initilization"""
	signal = np.zeros((1,data_length))
	profit = money*np.ones((1,data_length))
	out_pred_price = np.zeros((1,data_length))
	out_pred_price[0,0:train_length] = X[0,0:train_length]

	flag = 0

	for i in range(number_of_pred):

		"""get the training data"""
		train = np.array([X[0,i*pred_length:(train_length+i*pred_length)]])

		"""get the future price"""
		Y = np.array([X[0,(train_length+i*pred_length):(train_length+(i+1)*pred_length)]])

		"""get historical returns based on historical prices"""
		rt = dp.get_return(train)

		"""predict future returns based on model M"""
		pred_rt = M.predict(rt, pred_length)

		"""transfer the future returns to future prices"""
		pred_price = dp.get_price(train[0,train.shape[1]-1], pred_rt)

		"""generate trading signals based on future prices"""
		temp_signal = signal_generation(pred_price, window)

		"""get the profti time series"""
		temp_profit = profit_loss(Y, temp_signal, money)

		"""get the terminal money after one trading period"""
		money = temp_profit[0,-1]

		"""record the profit and signal and predicted prices"""
		signal[0,(train_length+i*pred_length):(train_length+(i+1)*pred_length)] = temp_signal[0,:]
		profit[0,(train_length+i*pred_length):(train_length+(i+1)*pred_length)] = temp_profit[0,:]
		out_pred_price[0,(train_length+i*pred_length):(train_length+(i+1)*pred_length)] = pred_price[0,:]

		flag = i

	profit[0,train_length+(flag+1)*pred_length:] = money

	return profit, signal, out_pred_price



