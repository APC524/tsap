import unittest
import math
import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../")
import tsap.data_processor as dp
from tsap import trading
from tsap import model



class TestTrading(unittest.TestCase):

    def testSignalGeneration1(self):
        """test signal_generation with upper trend"""
        x = np.array([[1.0, 3.0, 5.0, 7.0]])
        y = trading.signal_generation(x)
        z = np.array([[1, -1, 1, -1]])
        np.testing.assert_array_almost_equal(y,z)

    def testSignalGeneration2(self):
        """test signal_generation with lower trend"""
        x = np.array([[10.0, 8.0, 6.0, 4.0, 2.0]])
        y = trading.signal_generation(x)
        z = np.array([[0, 0, 0, 0, 0]])
        np.testing.assert_array_almost_equal(y,z)
    
    def testSignalGeneration3(self):
        """test signal_generation without trend"""
        x = x = np.array([[2.0, 8.5, 6.0, 3.0, 10.0, 1.5, 23.0, 7.5, 30.6]])
        y = trading.signal_generation(x, 3)
        z = np.array([[1, -1, 0, 0, 0, 0, 0, 1, -1]])
        np.testing.assert_array_almost_equal(y,z)

    def testSignalGeneration4(self):
        """test signal_generation with bigger window"""
        x = x = np.array([[2.0, 8.5, 6.0, 3.0, 10.0, 1.5, 23.0, 7.5, 30.6, 4.5]])
        y = trading.signal_generation(x, 4)
        z = np.array([[1, -1, 0, 0, 0, 1, -1, 0, 0, 0]])
        np.testing.assert_array_almost_equal(y,z)

    def testSignalGeneration5(self):
        """test signal_generation with bigger holding period"""
        x = x = np.array([[2.0, 4.5, 6.0, 3.0, 1.0, 1.5, 2.3, 7.5, 30.6, 4.5]])
        y = trading.signal_generation(x, 4)
        z = np.array([[1, 0, -1, 0, 1, 0, 0, -1, 0, 0]])
        np.testing.assert_array_almost_equal(y,z)

    ###################################################################

    def testProfitLoss1(self):
        """test profit_loss with upper trend, this is immediate buy"""
        x = np.array([[1.0, 3.0, 5.0, 7.0]])
        y = np.array([[1, -1, 1, -1]])
        z = trading.profit_loss(x,y)
        w = np.array([[1.0, 3.0, 3.0, 4.2]])
        np.testing.assert_array_almost_equal(z,w)

    def testProfitLoss2(self):
        """test profit_loss with upper trend, this is immediate buy"""
        x = np.array([[1.0, 3.0, 5.0, 7.0, 9.0, 10.0]])
        y = np.zeros((1,6))
        z = trading.profit_loss(x,y,2.0)
        w = np.array([np.repeat(2.0,6)])
        np.testing.assert_array_almost_equal(z,w)

    def testProfitLoss3(self):
        """test profit_loss with longer holding period"""
        x = np.array([[1.0, 2.0, 5.0, 7.0, 9.0, 10.0, 16.0]])
        y = np.array([[0, 1, 0, 0, 0, 0, -1]])
        z = trading.profit_loss(x,y)
        w = np.array([[1.0, 1.0, 2.5, 3.5, 4.5, 5.0, 8.0]])
        np.testing.assert_array_almost_equal(z,w)

    def testProfitLoss4(self):
        """test profit_loss with longer holding period, multiple trades and more money"""
        x = np.array([[1.0, 3.0, 5.0, 7.0, 9.0, 10.0, 16.0, 4.0, 5.0, 5.0, 2.0, 1.0]])
        y = np.array([[0, 1, 0, 0, 0, 0, -1, 1, -1, 1, 0, -1]])
        z = trading.profit_loss(x,y,3.0)
        w = np.array([[3.0, 3.0, 5.0, 7.0, 9.0, 10.0, 16.0, 16.0, 20.0, 20.0, 8.0, 4.0]])
        np.testing.assert_array_almost_equal(z,w)

    ###################################################################

    def testTrade1(self):
        """test trade with a very simple model"""
        AR_model = model.AR(lag=1, phi=np.array([[1.0]]), sigma=0.1, intercept=0.0)
        """historical prices"""
        x = np.array([[1.0, 3.0, 5.0, 7.0, 1.0, 2.0]])
        """future prices"""
        y = np.array([[1.0, 2.6, 1.7, 3.0]])
        n = 4
        window = 3 
        """trading profits"""
        z = trading.trade(x, y, AR_model, n, window)
        w = np.array([[1.0, 2.6, 1.7, 1.7]])
        np.testing.assert_array_almost_equal(z,w)


if __name__ == "__main__":
    unittest.main()
