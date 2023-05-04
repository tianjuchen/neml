#!/usr/bin/env python3

import sys

sys.path.append("../..")

import numpy as np

from neml.cp import (
    crystallography,
    slipharden,
    sliprules,
    inelasticity,
    kinematics,
    singlecrystal,
    polycrystal,
    addmaf,
)
from neml.math import rotations, tensors, nemlmath
from neml import elasticity, drivers

import matplotlib.pyplot as plt

if __name__ == "__main__":
    sdir = np.array([0, 0, 1.0, 0, 0, 0])
    erate = 1.0e-4
    steps = 500
    emax = 0.25

    uf = 1.0e9

    E = 200.0e3
    nu = 0.3

    mu = E / (2 * (1 + nu))
    alpha = 0.3
    M = np.ones((12,)) * mu
    b = 0.256e-9 * uf
    T = 300.0
    kb = 1.380649e-23 * (uf**3)
    kw1_v = 1.13e9 / uf
    kw2_v = 50.0
    kw1 = np.ones((12,)) * kw1_v
    kw2 = np.ones((12,)) * kw2_v
    ki1_v = 1.13e8 / uf
    ki2_v = 50.0
    ki1 = np.ones((12,)) * ki1_v
    ki2 = np.ones((12,)) * ki2_v
    c = 100.0
    dc = 1.0e-5 * uf
    alphai = 0.25
    alphaw = 0.95
    omega = 1.687e-4 * 2.5 * uf
    k0 = 1.0e-6
    Qv = 1.0e4
    R = 8.3145
    lamda = 1.0
    Tr = 298.0
    ftr = 0.1

    strengthmodel = addmaf.AMModel(M, kw1, kw2, ki1, ki2, ftr=ftr, initsigma=75.0)

    g0 = 1.0
    n = 20.0

    N = 100

    nthreads = 4

    slipmodel = sliprules.PowerLawSlipRule(strengthmodel, g0, n)
    imodel = inelasticity.AsaroInelasticity(slipmodel)
    emodel = elasticity.IsotropicLinearElasticModel(E, "youngs", nu, "poissons")
    kmodel = kinematics.StandardKinematicModel(emodel, imodel)

    lattice = crystallography.CubicLattice(1.0)
    lattice.add_slip_system([1, 1, 0], [1, 1, 1])

    model = singlecrystal.SingleCrystalModel(kmodel, lattice, verbose=True)

    orientations = rotations.random_orientations(N)

    dt = emax / erate / steps

    # pmodel = polycrystal.TaylorModel(model, orientations, nthreads=nthreads)

    res = drivers.uniaxial_test(model, erate, emax=emax, nsteps=steps)

    strains = res["strain"]
    stresses = res["stress"]

    plt.figure()

    plt.plot(strains, stresses, lw=4)

    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    plt.close()
