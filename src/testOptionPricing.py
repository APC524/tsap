import numpy as np
import matplotlib.pyplot as plt
import optionPricing
from mpl_toolkits.mplot3d import Axes3D

# read SP500 data
SP500 = np.genfromtxt('../data/SP500array.csv', delimiter=',')
SP500 = SP500.T

# take first stock, Abbott Laboratories
x = SP500[0,:]
sigma = np.std(x)
r = 0.01
T = 1
K = np.mean(x)
Sboundary=10*K
Vboundary=9*K

# test optionPricing class
optionPriceobj = optionPricing.OptionPricing(sigma,r,T,K,Sboundary,Vboundary)
dS = 0.01
dt = 1./365
V = optionPriceobj.BlackScholesEqn(dS,dt)

# define (S,t) grids
nS, nt = int(Sboundary/dS), int(T/dt)
S = np.arange(0, Sboundary, nS)
t = np.arange(0, T, nt)
S, t = np.meshgrid(S, t)

# visualize V(S,t)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(S, t, V, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)                       
plt.show()