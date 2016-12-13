import numpy as np
from scipy.optimize import line_search


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
    - modelgrad: Model function gradient.
    - f: Function value at x.
    - dx: Function gradient at x.
    - H: BFGS matrix.
    - alpha: Step size.
    """
    def correction(H, s, y):
        u = np.vdot(y, s)
        v = np.dot(H, y)
        H += (np.dot(s, np.transpose(s)) * (1 + np.vdot(v, y) / u) / u
                - (np.dot(s, v) + np.dot(v, s)) / u)
        return H

    objfcn = config['model']
    objfcngrad = config['modelgrad']
    config.setdefault('f', objfcn(x))
    config.setdefault('dx', objfcngrad(x))
    config.setdefault('H', np.eye(x.size))
    config.setdefault('alpha', 1e-3)
    f = config['f']
    H = config['H']
    alpha = config['alpha']

    d = -np.linalg.solve(H, dx)
    next_alpha = line_search(f=objfcn, myfprime=objfcngrad, xk=x, pk=d, gfk=dx,
            old_fval=f)
    next_x = x + next_alpha * d
    next_f, next_dx = objfcn(next_x)
    next_H = correction(H, next_x - x, next_dx - dx)

    config['f'] = next_f
    config['dx'] = next_dx
    config['H'] = next_H
    config['alpha'] = next_alpha
    return next_x, config
