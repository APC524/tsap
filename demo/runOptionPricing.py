import sys
if "../" not in sys.path:
    sys.path.append("../src/")

import matplotlib.pyplot as plt
import numpy as np
from option_pricing import OptionPricing


def main():

    # read GOOG data
    GOOG = np.genfromtxt('../data/GOOG.csv', delimiter=',')
    
    # set Black-Scholes equation parameters
    sigma = 0.1
    r = 0.01
    T = 1
    K = 1
    Smax = K*5
    
    # test optionPricing class
    option_price = OptionPricing(sigma=sigma, r=r, T=T, K=K, Smax=Smax)
    dS = 1e-1
    dt = 1e-2
    option_price.solve_black_scholes(dS, dt)
    
    # define (S,t) grids
    S = np.linspace(0, Smax, int(Smax / dS) + 1)
    t = np.linspace(0, T, int(T / dt) + 1)
    S, t = np.meshgrid(S, t)
    levels = np.arange(0, Smax, dS)
    
    # visualize V(S,t)
    plt.figure()
    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.contourf(S, t, V.T, levels, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    plt.colorbar()
    plt.xlabel('S/stock price', fontsize=18)   
    plt.ylabel('t/year', fontsize=18)
    plt.xlim([0,Smax])
    plt.ylim([0,T])
    plt.title('option pricing V(S,t)', fontsize=18)
    plt.show()

if __name__ == '__main__':
        main()
