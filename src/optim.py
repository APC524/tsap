import numpy as np


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """

    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
      average of the gradients.
    """

    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))
    
    next_w = None
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
    config['velocity'] = v

    return next_w, config


def bfgs(x, dx, config):
    """
    Performs BFGS method.

    config format:
    - model: Model function.
    - modelgrad: The gradient of model function.
    - f: Function value at x.
    - H: BFGS matrix.
    - alpha: Step size.
    """

    def golden_search(objfcn, a, x, b, fx, epsl=1e-6):
        d = np.linalg.norm(a - b)
        if d <= epsl:
            return (a + b) / 2

        TAU = 2 / (np.sqrt(5) - 1)
        feval = int(np.ceil(np.log(d / epsl) / np.log(TAU))) 
        tag = True
        
        for itr in range(feval):
            y = a + b - x
            fy = f(y)
            if tag:
                if fx >= fy:
                    b = y
                    tag = False
                else:
                    a, x, fx = x, y, fy
            else:
                if fx <= fy:
                    a = y
                    tag = True
                else:
                    b, x, fx = x, y, fy

        return x
    
    def line_search(
        objfcn, x0, d, fx0=None, alpha0=1e-3, epsl=1e-6, max_step=100):
        if not fx0:
            fx0 = objfcn(x0)

        TAU = 2 / (np.sqrt(5) - 1)
        d = alpha0 / np.linalg.norm(d) * d
        x1 = x0 + d
        fx1 = objfcn(x1)

        if fx1 >= fx0:
            d = -d / TAU
            x2 = x0 + d
            fx2 = objfcn(x2)
            if fx2 >= fx0:
                return golden_search(objfcn, x2, x0, x1, fx0, epsl)
            else:
                x1, fx1 = x2, fx2

        for itr in range(5):
            d = d * TAU
            x2 = x1 + d
            fx2 = objfcn(x2)
            if fx2 >= fx1:
                return golden_search(objfcn, x0, x1, x2, fx1, epsl)
            x0, x1, fx1 = x1, x2, fx2

        for itr in range(5, max_step):
            x2 = x1 + d / TAU
            fx2 = objfcn(x2)
            if fx2 >= fx1:
                return golden_search(objfcn, x2, x1, x0, fx1, epsl)
            x0, x1, fx1 = x1, x2, fx2

            x2 = x1 + d
            fx2 = f(x2)
            if fx2 >= fx1:
                return golden_search(objfcn, x0, x1, x2, fx1, epsl)
            x0, x1, fx1 = x1, x2, fx2

    def update_H(H, s, y):
        u = np.vdot(y, s)
        v = np.dot(H, y)
        H += (np.dot(s, np.transpose(s)) * (1 + np.vdot(v, y) / u) / u
                - (np.dot(s, v) + np.dot(v, s)) / u)
        return H

    objfcn = config['model']
    objfcngrad = config['modelgrad']
    config.setdefault('f', objfcn(x))
    config.setdefault('H', np.eye(x.size))
    config.setdefault('alpha', 1e-3)
    f = config['f']
    H = config['H']
    alpha = config['alpha']

    d = -np.linalg.solve(H, dx)
    next_x = line_search(
        objfcn=objfcn, x0=x, d=d, fx0=f, alpha0=alpha, epsl=alpha*0.01)
    next_x = x + next_alpha * d
    next_f = objfcn(next_x)
    next_H = update_H(H, next_x - x, next_dx - dx)
    next_alpha = np.linalg.norm(next_x - x)

    config['f'] = next_f
    config['H'] = next_H
    config['alpha'] = next_alpha
    return next_x, config
