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


class CommonSlipHardening:
    def test_d_hist_d_stress(self):
        d = np.array(
            self.model.d_hist_d_s(
                self.S, self.Q, self.H, self.L, self.T, self.sliprule, self.fixed
            )
        )
        nd = diff_history_symmetric(
            lambda s: self.model.hist(
                s, self.Q, self.H, self.L, self.T, self.sliprule, self.fixed
            ),
            self.S,
        )

        self.assertTrue(np.allclose(nd.reshape(d.shape), d))

    
    def test_d_hist_d_hist(self):
        d = np.array(
            self.model.d_hist_d_h(
                self.S, self.Q, self.H, self.L, self.T, self.sliprule, self.fixed
            )
        )
        nd = diff_history_history(
            lambda h: self.model.hist(
                self.S, self.Q, h, self.L, self.T, self.sliprule, self.fixed
            ),
            self.H,
        )

        nd = nd.reshape(d.shape)
        
        print("nd is:", nd.reshape(25, 25))
        print("d is:", d.reshape(25, 25))

        self.assertTrue(np.allclose(nd.reshape(d.shape), d, rtol=1.0e-4))
    
    """
    def test_d_hist_to_tau_d_hist(self):
        iq = 0
        for g in range(self.L.ngroup):
            for i in range(self.L.nslip(g)):
                nd = diff_history_scalar(
                    lambda h: self.model.hist_to_tau(
                        g, i, h, self.L, self.T, self.fixed
                    ),
                    self.H,
                )
                d = self.model.d_hist_to_tau(g, i, self.H, self.L, self.T, self.fixed)

                print("neml sf is:", self.model.fmod(self.H))
                # print("dsfdd is:", self.model.dfsigdd(self.H))
                # print("f is:", self.wfrac(iq, g, i))
                # print("dsfdd act is:", self.dsfdd())
                print("sf is: ", self.fmod())


                d_act = np.zeros((25,))

                d_act[0] = (
                    self.alphai
                    * self.M[iq]
                    * self.b
                    * np.sqrt(np.array(self.H)[iq + 12 + 1])
                    * (
                        - self.dfdd(iq, g, i)
                        + self.dfdd(iq, g, i) * self.fmod()
                        + self.wfrac(iq, g, i) * self.dsfdd()
                    )
                ) + self.alphaw * self.M[iq] * self.b * np.sqrt(
                    np.array(self.H)[iq + 1]
                ) * (
                    self.dfdd(iq, g, i)
                    - (
                        self.dfdd(iq, g, i) * self.fmod()
                        + self.wfrac(iq, g, i) * self.dsfdd()
                    )
                )

                d_act[iq + 1] = (
                    0.5
                    * self.alphaw
                    * self.M[iq]
                    * self.b
                    * 1
                    / np.sqrt(np.array(self.H)[iq + 1])
                    * self.wfrac(iq, g, i)
                    * (1 - self.fmod())
                )

                d_act[iq + 12 + 1] = (
                    0.5
                    * self.alphai
                    * self.M[iq]
                    * self.b
                    * 1
                    / np.sqrt(np.array(self.H)[iq + 12 + 1])
                    * (1 - self.wfrac(iq, g, i) * (1 - self.fmod()))
                )

                print("nd is: ", np.array(nd))
                print("d is: ", np.array(d))
                # print("d_act is: ", d_act)
                self.assertTrue(np.allclose(np.array(nd), np.array(d)))
    """

class TestAMModel(unittest.TestCase, CommonSlipHardening):
    def setUp(self):
        self.L = crystallography.CubicLattice(1.0)
        self.L.add_slip_system([1, 1, 0], [1, 1, 1])

        self.Q = rotations.Orientation(35.0, 17.0, 14.0, angle_type="degrees")
        self.S = tensors.Symmetric(
            np.array([[100.0, -25.0, 10.0], [-25.0, -17.0, 15.0], [10.0, 15.0, 35.0]])
        )

        self.nslip = self.L.ntotal
        self.H = history.History()

        self.uf = 1.0e9

        for i in range(25):
            if i == 0:
                self.H.add_scalar("wall" + str(i))
                self.H.set_scalar("wall" + str(i), 5.5e-7 * self.uf)
            elif i <= 12 and i > 0:
                self.H.add_scalar("wslip" + str(i))
                self.H.set_scalar("wslip" + str(i), 5.0e12 / (self.uf**2))
            else:
                self.H.add_scalar("islip" + str(i))
                self.H.set_scalar("islip" + str(i), 1.0e6 / (self.uf**2))

        self.M_v = 200.0e3
        self.M = np.ones((12,)) * self.M_v
        self.b = 0.256e-9 * self.uf

        self.T = 300.0
        self.kb = 1.380649e-23
        self.kw1_v = 1.13e9 / self.uf
        self.kw2_v = 50.0

        self.kw1 = np.ones((12,)) * self.kw1_v
        self.kw2 = np.ones((12,)) * self.kw2_v

        self.ki1_v = 1.13e8 / self.uf
        self.ki2_v = 50.0

        self.ki1 = np.ones((12,)) * self.ki1_v
        self.ki2 = np.ones((12,)) * self.ki2_v

        self.c = 0.0
        self.dc = 1.0e-7 * self.uf

        self.alphai = 0.25
        self.alphaw = 0.95

        self.omega = 1.687e-4 / (self.uf**2)
        self.k0 = 1.0e-6 / self.uf
        self.Qv = 1.0e4
        self.R = 8.3145
        self.lamda = 1.0
        self.Tr = 298.0
        self.ftr = 0.1

        self.model = addmaf.AMModel(
            self.M,
            self.kw1,
            self.kw2,
            self.ki1,
            self.ki2,
            b=self.b,
            dc=self.dc,
            c=self.c,
            k0=self.k0,
            omega=self.omega,
            alpha_w=self.alphaw,
            alpha_i=self.alphai,
            ftr=self.ftr,
            # iniwvalue=5.0e-6,
            # iniivalue=1.0e-12,
            # inibvalue=5.5e2,
        )

        self.g0 = 1.0
        self.n = 3.0
        self.sliprule = sliprules.PowerLawSlipRule(self.model, self.g0, self.n)

        self.fixed = history.History()

    def wfrac(self, iq, g, i):
        return (
            self.M[iq]
            * self.omega
            * self.b**3
            / (self.kb * self.T)
            * 1
            / np.array(self.H)[0]
        )

    def fmod(self):
        return 1 / (1 + np.exp(-self.c * (np.array(self.H)[0] / self.dc - 1)))

    def dsfdd(self):
        return (
            self.c
            / self.dc
            * np.exp(self.c - self.c / self.dc * np.array(self.H)[0])
            / (np.exp(self.c - self.c / self.dc * np.array(self.H)[0]) + 1) ** 2
        )

    def dfdd(self, iq, g, i):
        return (
            -self.M[iq]
            * self.omega
            * self.b**3
            / (self.kb * self.T)
            * 1
            / (np.array(self.H)[0]) ** 2
        )

    def test_hist_to_tau(self):
        direct = [
            self.model.hist_to_tau(g, i, self.H, self.L, self.T, self.fixed)
            for g in range(self.L.ngroup)
            for i in range(self.L.nslip(g))
        ]
        direct += [direct[0]]
        direct += [direct[-1] for i in range(12)]

        # Then implement what it should be in python and compare
        check = np.zeros((25,))

        iq = 0
        for g in range(self.L.ngroup):
            for i in range(self.L.nslip(g)):
                check[iq + 1] = self.alphai * self.M[iq] * self.b * np.sqrt(
                    np.array(self.H)[iq + 12 + 1]
                ) * (
                    1 - self.wfrac(iq, g, i) * (1 - self.fmod())
                ) + self.alphaw * self.M[
                    iq
                ] * self.b * np.sqrt(
                    np.array(self.H)[iq + 1]
                ) * self.wfrac(
                    iq, g, i
                ) * (
                    1 - self.fmod()
                )
                iq += 1
        check[13:] = check[1:13]
        check[0] = check[-1]

        self.assertTrue(np.allclose(direct, check, rtol=1.0e-4))

    def srates(self, g, i):
        return self.sliprule.slip(
            g, i, self.S, self.Q, self.H, self.L, self.T, self.fixed
        )

    def test_definition(self):
        direct = self.model.hist(
            self.S, self.Q, self.H, self.L, self.T, self.sliprule, self.fixed
        )

        act = np.zeros((25,))
        act[0] = (
            self.k0
            * np.exp(-self.Qv / (self.R * self.T))
            * np.array(self.H)[0]
            * np.exp(-np.array(self.H)[0] / self.dc)
        )

        iq = 0
        for g in range(self.L.ngroup):
            for i in range(self.L.nslip(g)):
                act[iq + 1] = (
                    self.kw1[iq] * np.sqrt(np.array(self.H)[iq + 1])
                    - self.kw2[iq] * np.array(self.H)[iq + 1]
                ) * np.abs(self.srates(g, i)) - 2 / self.lamda * (
                    np.array(self.H)[iq + 1]
                ) ** 1.5 * (
                    act[0] * self.T / (self.T + self.Tr) * self.ftr
                )

                act[iq + 1 + 12] = (
                    self.ki1[iq] * np.sqrt(np.array(self.H)[iq + 12 + 1])
                    - self.ki2[iq] * np.array(self.H)[iq + 12 + 1]
                ) * np.abs(self.srates(g, i)) + 2 / self.lamda * (
                    np.array(self.H)[iq + 12 + 1]
                ) ** 1.5 * (
                    act[0] * self.T / (self.T + self.Tr) * self.ftr
                )
                iq += 1
        # print(np.array(direct))
        # print(act)
        self.assertTrue(np.allclose(direct, act, atol=1.0e-4))
