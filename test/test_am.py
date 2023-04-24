#!/usr/bin/env python3

from neml import interpolate
from neml.math import tensors, rotations, matrix
from neml.cp import crystallography, slipharden, sliprules, addmaf

from common import differentiate
from nicediff import *

import unittest
import numpy as np
import numpy.linalg as la
import numpy.random as ra

class CommonSlipHardening():
  def test_d_hist_d_stress(self):
    d = np.array(self.model.d_hist_d_s(self.S, self.Q, self.H, self.L, self.T, self.sliprule, self.fixed))
    nd = diff_history_symmetric(lambda s: self.model.hist(s, self.Q, self.H, self.L, self.T,
      self.sliprule, self.fixed), self.S)

    self.assertTrue(np.allclose(nd.reshape(d.shape), d))

  def test_d_hist_d_hist(self):
    d = np.array(self.model.d_hist_d_h(self.S, self.Q, self.H, self.L, self.T, self.sliprule, 
      self.fixed))
    nd = diff_history_history(lambda h: self.model.hist(self.S, self.Q, h, self.L, self.T,
      self.sliprule, self.fixed), self.H)
    
    nd = nd.reshape(d.shape)

    self.assertTrue(np.allclose(nd.reshape(d.shape), d, rtol = 1.0e-4))

  def test_d_hist_to_tau_d_hist(self):
    for g in range(self.L.ngroup):
      for i in range(self.L.nslip(g)):
        nd = diff_history_scalar(lambda h: self.model.hist_to_tau(g, i, h, self.L, self.T, self.fixed), self.H)
        d = self.model.d_hist_to_tau(g, i, self.H, self.L, self.T, self.fixed)
        self.assertTrue(np.allclose(np.array(nd), np.array(d)))
    
class TestAMModel(unittest.TestCase, CommonSlipHardening):
  def setUp(self):
    self.L = crystallography.CubicLattice(1.0)
    self.L.add_slip_system([1,1,0],[1,1,1])
    
    self.Q = rotations.Orientation(35.0,17.0,14.0, angle_type = "degrees")
    self.S = tensors.Symmetric(np.array([
      [100.0,-25.0,10.0],
      [-25.0,-17.0,15.0],
      [10.0,  15.0,35.0]]))
    
    self.nslip = self.L.ntotal
    self.H = history.History()

    for i in range(25):
        if i == 0:
            self.H.add_scalar("wall"+str(i))
            self.H.set_scalar("wall"+str(i), 550e-9)
        elif (i<=12 and i>0):
            self.H.add_scalar("wslip"+str(i))
            self.H.set_scalar("wslip"+str(i), 1.0e11)
        else:
            self.H.add_scalar("islip"+str(i))
            self.H.set_scalar("islip"+str(i), 1.0e5)


    self.M_v = 200.0e3
    self.M = np.ones((12,)) * self.M_v
    
    self.T = 300.0
    
    self.kw1_v = 1.13e9
    self.kw2_v = 50.0
    
    self.kw1 = np.ones((12,)) * self.kw1_v
    self.kw2 = np.ones((12,)) * self.kw2_v
    
    self.ki1_v = 1.13e8
    self.ki2_v = 50.0

    self.ki1 = np.ones((12,)) * self.ki1_v
    self.ki2 = np.ones((12,)) * self.ki2_v

    self.model = addmaf.AMModel(self.M, self.kw1, self.kw2, self.ki1, self.ki2)
    
    
    self.g0 = 1.0
    self.n = 3.0
    self.sliprule = sliprules.PowerLawSlipRule(self.model, self.g0, self.n)

    self.fixed = history.History()

  def test_hist_to_tau(self):
    for g in range(self.L.ngroup):
      for i in range(self.L.nslip(g)):
        model = self.model.hist_to_tau(g, i, self.H, self.L, self.T,
            self.fixed)
        should = self.static
        self.assertAlmostEqual(model, should)

  def test_definition(self):
    hrate = self.model.hist(self.S, self.Q, self.H, self.L, self.T, self.sliprule,
        self.fixed)
    exact = np.array([])
    self.assertTrue(np.allclose(hrate, exact))

