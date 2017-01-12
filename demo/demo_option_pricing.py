import sys
if "../" not in sys.path:
    sys.path.append("../src/")

import matplotlib.pyplot as plt
import numpy as np
from option_pricing import OptionPricing


def main():

    # read GOOG data
    goog = np.genfromtxt("../data/GOOG.csv", delimiter=",")
    
    # set Black-Scholes equation parameters
    sigma = np.std((goog[1:] - goog[:-1]) / goog[:-1])
    r = 0.01
    T = 180
    K = 800
    Smax = K*2
    
    # test optionPricing class
    option_price = OptionPricing(sigma=sigma, r=r, T=T, K=K, Smax=Smax)
    nS = 100
    nt = 180
    option_price.solve_black_scholes(nS=nS, nt=nt)
    
    # define (S,t) grids
    S = np.linspace(0, Smax, nS + 1)
    t = np.linspace(0, T, nt + 1)
    S, t = np.meshgrid(S, t)
    levels = np.arange(0, Smax, float(Smax) / nS)
    
    # visualize V(S,t)
    plt.figure()
    # plt.rc('font', family='serif')
    plt.contourf(
        t, S, option_price.V, levels,
        cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    plt.colorbar()
    plt.xlabel("Stock price", fontsize=18)   
    plt.ylabel("Time", fontsize=18)
    plt.xlim([0,Smax])
    plt.ylim([0,T])
    plt.title("option pricing V(S,t)", fontsize=12)
    plt.show()

if __name__ == '__main__':
        main()
