import numpy as np


class OptionPricing(object):
    """
    Callable option pricing object.
    Example usage:
    optionPriceobj = optionPricing(sigma,r,T,K), declare a class object
    V = optionPriceobj.BlackScholesEqn(dS,dt), compute V array in shape [nS,nt]
    Vst = optionPriceobj.optionPrice(V,S,t), compute V(S,t) given V array
    """

    def __init__(self, sigma=0.1, r=0.01, T=1, K=1, Smax=None, Vmax=None):
        self._sigma = sigma
        self._r = r
        self._T = T
        self._K = K
        if Smax is None or Vmax is None:
            self._Smax = 5*self._K
            self._Vmax = max(self._Smax-self._K, 0)
        else:
            self._Smax = Smax
            self._Vmax = Vmax
        
    def solve_black_scholes(self, dS, dt):
        """
        V(S, t) satisfies Black-Scholes equation
        dVdt + (1/2)*sigma^2*S^2*d2VdS2 + r*S*dVdS - r*V = 0
        
        Inputs:
        dS: float, size of grids in S dimension
        dt: float, size of grids in t dimension
        """
        nS = int(self._Smax / dS) + 1
        nT = int(self._T / dt) + 1
        # test stability condition
        assert dt < 1. / (self._sigma**2 * (nS - 1) + self._r/2), (
            'Make sure dt < 1/(sigma**2 (nS-1) + r/2)')
        self.V = np.zeros((nS,nt))
        # terminal condition at t = T, j = nt-1
        for i in range(nS):
            self.V[i,-1] = max(i * dS - self._K, 0)
        # fixed boundary condition at S=0(i=0), S=Smax(i=nS-1)
        for j in range(nt):
            self.V[0,j] = 0
            self.V[-1,j] = self._Vmax
        # backward time integration j = nt-1, nt-1, ..., 1
        for j in range(nt-1, 0, -1):
            # update inner points i = 1, 2, 3, ..., nS - 2
            for i in range(1, nS - 1):
                # no gradient boundary condition
                d2VdS2 = (self.V[i+1,j] - self.V[i,j]*2 + self.V[i-1,j]) / dS**2
                dVdS = (self.V[i+1,j] - self.V[i-1,j]) / (dS*2)
                S = i * dS
                spatialDeritive = (-0.5 * self._sigma**2 * S**2 * d2VdS2 - 
                    self._r * S * dVdS + self._r * self.V[i,j])
                self.V[i,j-1] = self.V[i,j] - dt * spatialDeritive
        
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
        nS, nT = np.shape(V)
        return self.V[int(S * nS / self._Smax), int(t * nt / self._T)]
