#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# slice_sampler.py
#
# (c) 2013 Dominik Wabersich <dominik.wabersich [aet] gmail.com>
# GPL 3.0+ or (cc) by-sa (http://creativecommons.org/licenses/by-sa/3.0/)
#
# created 2013-02-05
# last mod 2013-02-07 18:30 DW
#

from pymc.StepMethods import StepMethod
from pymc.utils import float_dtypes
from pymc.Node import ZeroProbability
from pymc import runiform, rexponential
from numpy import floor, exp, abs, infty

class Slicer(StepMethod):
  """ 
  Slice Sampler Step Method
  """

  def __init__(self, stochastic, w=1, m=20, n_tune=0, verbose=-1, tally=False):
    """ 
    Slice sampler class initialization
    """
    #a Initialize superclass
    StepMethod.__init__(self, [stochastic], tally=tally)

    # id string
    self._id = "Slicer"

    # Set public attributes
    self.stochastic = stochastic
    if verbose > -1:
      self.verbose = verbose
    else:
      self.verbose = stochastic.verbose

    self.w_tune = []
    self.n_tune = n_tune
    self._tuning_info = ["w_tune", "n_tune"]
    self.w = w
    self.m = m

  @staticmethod
  def competence(s):
    """ 
    The competence function for Slice
    """
    if s.dtype in float_dtypes:
      return 1
    else:
      return 0

  def step(self):
    """ 
    Slice step method
    """
    #y = runiform(0,1) * exp(self.loglike)
    y = self.loglike - rexponential(1)

    # Stepping out procedure
    L = self.stochastic.value - self.w*runiform(0,1)
    R = L + self.w
    J = floor(self.m*runiform(0,1))
    K = (self.m-1)-J
    while(J>0 and y<self.fll(L)):
      L = L - self.w
      J = J - 1
    while(K>0 and y<self.fll(R)):
      R = R + self.w
      K = K - 1
    self.stochastic.value = runiform(L,R)
    try:
      #y_new = exp(self.loglike)
      y_new = self.loglike
    except ZeroProbability:
      #print("ZeroProbability Warning")
      #y_new = 0.0
      y_new = -infty
    i = 0
    while(y_new<y):
      i = i+1
      #print str(y) + "value" + str(self.stochastic.value)
      #print str(L) + " left | right " + str(R)
      if not(y_new == -infty):
        if ((L+(R-L)/2.0) > self.stochastic.value):
          L = self.stochastic.value
        else:
          R = self.stochastic.value
      self.stochastic.revert()
      # For some reason, this algorithm runs into bullshit sometimes
      if (L == R or i == 50):
        #print("Something went wrong with the Slice sampler")
        self.step()
        break
      self.stochastic.value = runiform(L,R)
      try:
        #y_new = exp(self.loglike)
        y_new = self.loglike
      except ZeroProbability:
        #print("ZeroProbability Warning")
        #y_new = 0.0
        y_new = -infty
      print self.stochastic.value
      

  def fll(self, value):
    """
    fll(value) returns loglike of value
    """
    self.stochastic.value = value
    try:
      ll = self.loglike
    except ZeroProbability:
      #ll = 0.0
      ll = -infty
    self.stochastic.revert()
    #return exp(ll)
    return ll

  def tune(self, verbose=-1):
    """
    Slice tune method
    """
    if (len(self.w_tune) >= self.n_tune):
      return False
    else:
      self.w_tune.append(abs(self.stochastic.last_value - self.stochastic.value))
      self.w = 2 * (sum(self.w_tune)/len(self.w_tune))
      return True
