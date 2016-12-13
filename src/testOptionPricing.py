import numpy as np
import matplotlib.pyplot as plt
import optionPricing

# read SP500 data
SP500 = np.genfromtxt('../data/SP500array.csv', delimiter=',')
SP500 = SP500.T

# set Black-Scholes equation parameters
sigma = 0.1
r = 0.01
T = 1
K = 1
Sboundary=10*K
Vboundary=9*K

# test optionPricing class
optionPriceobj = optionPricing.OptionPricing(sigma,r,T,K,Sboundary,Vboundary)
dS = 0.1
dt = 1e-2
V = optionPriceobj.BlackScholesEqn(dS,dt)

# define (S,t) grids
nS, nt = int(Sboundary/dS), int(T/dt)
S = np.arange(0, Sboundary, dS)
t = np.arange(0, T, dt)
S, t = np.meshgrid(S, t)
levels = np.arange(0,Sboundary,0.1)

# visualize V(S,t) as contourf
plt.figure()
plt.contourf(S, t, V.T, levels, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
plt.colorbar()
plt.xlabel('S/stock price', fontsize=20)   
plt.ylabel('t/year', fontsize=20)
plt.title('option pricing V(S,t)', fontsize=20)
plt.show()