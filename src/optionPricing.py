import numpy as np

class OptionPricing(object):
    """Callable option pricing object.
    Example usage:
    optionPriceobj = optionPricing(sigma,r,T,K)
    V = BlackScholesEqn(dS,dt), V shape [nS, nt]
    """
    def __init__(self, sigma=0.1, r=0.01, T=1, K=1, Sboundary=None, Vboundary=None):
        self._sigma = sigma
        self._r = r
        self._T = T
        self._K = K
        if Sboundary is None or Vboundary is None:
            self._Sboundary = 10*self._K
            self._Vboundary = max(self._Sboundary-self._K,0)
        else:
            self._Sboundary = Sboundary
            self._Vboundary = Vboundary
        
    def BlackScholesEqn(self, dS, dt):
        """
        V(S, t) satisfies Black-Scholes equation
        dVdt + (1/2)*sigma^2*S^2*d2VdS2 + r*S*dVdS - r*V = 0
        
        Inputs:
        dS: float, size of grids in S dimension
        dt: float, size of grids in t dimension
        
        Returns:
        V: array, shape [nS, nt]
        """
        nS, nt = int(self._Sboundary/dS), int(self._T/dt)
        # test stability condition
        if(dt>1./(self._sigma**2*(nS-1)+self._r/2)):
            print 'Make sure dt < 1/(sigma**2(nS-1)+r/2)'
            return 0
        V = np.zeros((nS,nt))
        # terminal condition at t = T, j = nt-1
        for i in range(nS):
            V[i,-1] = max(i*dS,0)
        # fixed boundary condition at S=0(i=0), S=Sboundary(i=nS-1)
        for j in range(nt):
            V[0,j] = 0
            V[-1,j] =self._Vboundary
        # backward time integration j = nt-1, nt-1, ..., 1
        for j in range(nt-1,0,-1):
            # update inner points i = 1, 2, 3, ..., nS - 2
            for i in range(1,nS-1):
                # no gradient boundary condition
                d2VdS2 = (V[i+1,j] - 2*V[i,j] + V[i-1,j])/dS**2
                dVdS = (V[i+1,j] - V[i-1,j])/(2*dS)
                S = i*dS
                spatialDeritive = -0.5*self._sigma**2*S**2*d2VdS2 - self._r*S*dVdS + self._r*V[i,j]
                V[i,j-1] = V[i,j] - dt*spatialDeritive
                #print 'j = %d, i = %d, V[i,j] = %f' % (j, i,V[i,j])
        return V
        
    def optionPrice(self,V,S,t):
        nS = V[:,0]
        nt = V[0,:]
        iS = int(nS*S/self._Sboundary)
        jt = int(nt*t/self._T)
        return V[iS,jt]

        
    
        

