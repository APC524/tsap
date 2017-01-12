import numpy as np
import matplotlib.pyplot as plt
import sys
if "../" not in sys.path:
  sys.path.append("../src/")
import optionPricing

# read SP500 data
SP500 = np.genfromtxt('../data/GOOG.csv', delimiter=',')
SP500 = SP500.T

# set Black-Scholes equation parameters
sigma = 0.1
r = 0.01
T = 1
K = 1
Smax=5*K
Vmax=max(Smax-K,0)

# test optionPricing class
optionPriceobj = optionPricing.OptionPricing(sigma,r,T,K,Smax,Vmax)
dS = 1e-1
dt = 1e-2
V = optionPriceobj.BlackScholesEqn(dS,dt)

# define (S,t) grids
nS, nt = int(Smax/dS)+1, int(T/dt)+1
S = np.linspace(0, Smax, nS)
t = np.linspace(0, T, nt)
S, t = np.meshgrid(S, t)
levels = np.arange(0,Smax,dS)

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
