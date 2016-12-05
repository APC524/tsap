import numpy as np

import optim

class Solver(object):
 

  def __init__(self, model, data, **kwargs):
  
    self.model = model
    self.X = data
    
    
    # Unpack keyword arguments
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.batch_size = kwargs.pop('batch_size', 100)
    self.num_epochs = kwargs.pop('num_epochs', 10)

    self.print_every = kwargs.pop('print_every', 10)
    self.verbose = kwargs.pop('verbose', True)

    # Throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
      extra = ', '.join('"%s"' % k for k in kwargs.keys())
      raise ValueError('Unrecognized arguments %s' % extra)

    # Make sure the update rule exists, then replace the string
    # name with the actual function
    if not hasattr(optim, self.update_rule):
      raise ValueError('Invalid update_rule "%s"' % self.update_rule)
    self.update_rule = getattr(optim, self.update_rule)

    self._reset()


  def _reset(self):

    # Set up some variables for book-keeping
    self.epoch = 0
    self.loss_history = []

    # Make a deep copy of the optim_config for each parameter
    self.optim_configs = {}
    for p in self.model.params:
      d = {k: v for k, v in self.optim_config.iteritems()}
      self.optim_configs[p] = d


  def _step(self):

    # Make a minibatch of training data
    num_train = self.X.shape[0]
    # batch_mask = np.random.choice(num_train, self.batch_size)
    # X_batch = self.X[batch_mask]

    # Compute loss and gradient
    # loss, grads = self.model.loss(X_batch)
    loss, grads = self.model.loss(self.X)
    print loss, grads
    self.loss_history.append(loss)

    # Perform a parameter update
    for p, w in self.model.params.iteritems():
      dw = grads[p]
      print dw
      config = self.optim_configs[p]
      
      next_w, next_config = self.update_rule(w, dw, config)
      self.model.params[p] = next_w
      self.optim_configs[p] = next_config


  def train(self):

    num_train = self.X.shape[0]
    iterations_per_epoch = max(num_train / self.batch_size, 1)
    num_iterations = self.num_epochs * iterations_per_epoch
    print num_iterations
    for t in xrange(num_iterations):
      self._step()

