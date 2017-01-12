import numpy as np


class OptionPricing(object):
    """
    Callable option pricing object.
    Example usage:
    optionPriceobj = optionPricing(sigma,r,T,K), declare a class object
    V = optionPriceobj.BlackScholesEqn(dS,dt), compute V array in shape [nS,nt]
    Vst = optionPriceobj.optionPrice(V,S,t), compute V(S,t) given V array
    """

    def __init__(self, sigma=0.1, r=0.01, T=1, K=1, Smax=None):
        self.sigma = sigma
        self.r = r
        self.T = T
        self.K = K
        if Smax is None:
            self.Smax = self.K*5
        else:
            self.Smax = Smax
        self._Vmax = max(self.Smax - self.K, 0)
        
    def solve_black_scholes(self, nS, nt):
        """
        V(S, t) satisfies Black-Scholes equation
        dVdt + (1/2)*sigma^2*S^2*d2VdS2 + r*S*dVdS - r*V = 0
        
        Inputs:
        nS: int, size of grids in S dimension
        nt: int, size of grids in t dimension
        """
        dS = float(self.Smax) / nS
        dt = float(self.T) / nt
        # test stability condition
        assert dt < 1. / (self.sigma**2 * nS + self.r/2), (
            'Make sure dt < 1/(sigma**2 * nS + r/2)')
        self.V = np.zeros((nS + 1, nt + 1))

        # terminal condition at t = T (j = nt)
        for i in range(nS + 1):
            self.V[i,-1] = max(dS * i - self.K, 0)
        # fixed boundary condition at S = 0 (i = 0), S = Smax (i = nS)
        for j in range(nt + 1):
            self.V[0,j] = 0
            self.V[-1,j] = self._Vmax
        
        # backward time integration j = nt, nt - 1, ..., 1
        for j in range(nt, 0, -1):
            # update inner points i = 1, 2, 3, ..., nS - 1
            for i in range(1, nS):
                # no gradient boundary condition
                S = dS * i
                dVdS = (self.V[i+1,j] - self.V[i-1,j]) / (dS*2)
                d2VdS2 = (self.V[i+1,j] - self.V[i,j]*2 + self.V[i-1,j]) / dS**2
                spatialDeritive = (-0.5 * self.sigma**2 * S**2 * d2VdS2 - 
                    self.r * S * dVdS + self.r * self.V[i,j])
                self.V[i,j-1] = self.V[i,j] - spatialDeritive * dt
        
    def get_option_price(self, S, t):
        """
        Compute V at V(S,t) by simple interpolation
        Inputs:
        V: array in shape [nS, nt]
        S: stock price
        t: time
        Returns:
        VSt: option price V(S,t)
        """
        nS, nt = np.shape(V)
        nS, nt = nS - 1, nt - 1
        return self.V[S * nS // self.Smax, t * nt // self.T]
