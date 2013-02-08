#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# testmodel.py
#
# (c) 2013 Dominik Wabersich <dominik.wabersich [aet] gmail.com>
# GPL 3.0+ or (cc) by-sa (http://creativecommons.org/licenses/by-sa/3.0/)
#
# created 2013-02-06
# last mod 2013-02-07 16:20 DW
#

# Import relevant modules
import pymc
import numpy as np

# example from the PyMC docs
def mymodel():
  # Some data
  n = 5*np.ones(4,dtype=int)
  x = np.array([-.86,-.3,-.05,.73])

  # Priors on unknown parameters
  alpha = pymc.Normal('alpha',mu=0,tau=.01)
  beta = pymc.Normal('beta',mu=0,tau=.01)

  # Arbitrary deterministic function of parameters
  @pymc.deterministic
  def theta(a=alpha, b=beta):
      """theta = logit^{-1}(a+b)"""
      return pymc.invlogit(a+b*x)

  # Binomial likelihood for data
  d = pymc.Binomial('d', n=n, p=theta, value=np.array([0.,1.,3.,5.]),\
                    observed=True)
  return locals()

# Easy dice example (analytically solvable)
def dice(data=None):
  #data
  if data == None:
    x = [pymc.rbernoulli(1.0/6.0) for i in range(0,100)]
  else:
    x = data

  prob = pymc.Uniform('prob', lower=0, upper=1)

  d = pymc.Bernoulli('d', p=prob, value=x, observed=True)

  return locals()

if __name__ == '__main__':
  from SliceSampler import Slicer
  M = pymc.MCMC(dice(), db='pickle')
  M.use_step_method(Slicer, M.prob, w=.1, n_tune=500)
  M.sample(iter=10000, burn=0, thin=1, tune_interval=1)
