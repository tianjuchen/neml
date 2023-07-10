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
    hucocks
)
from neml.math import rotations, tensors, nemlmath
from neml import elasticity, drivers, interpolate

import matplotlib.pyplot as plt

if __name__ == "__main__":

    Ts = np.array([500.0, 550.0, 600.0, 650.0]) + 273.15


    # G = interpolate.PiecewiseLinearInterpolate(
        # list(Ts), [61068, 59541.0, 57633.6, 55725.2]
    # )




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
    T = 600.0 #300.0
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


    G = interpolate.PiecewiseLinearInterpolate(
        list(Ts), [mu, mu, mu, mu]
    )


    w = 1.0

    # Setup for [Cr,C] <-> M23C6
    am_car = 3.6e-10
    N0_car = 1.0e13
    Vm_car = 6e-6
    chi_car = 0.3
    D0_car = 1.5e-4
    Q0_car = 240e3
    c0_car = [
        interpolate.ConstantInterpolate(16.25 / 100.0),
        interpolate.ConstantInterpolate(0.0375 / 100.0),
    ]
    cp_car = [
        interpolate.PiecewiseLinearInterpolate(
            list(Ts), [69.85 / 100, 69.05 / 100, 68.32 / 100, 67.52 / 100]
        ),
        interpolate.PiecewiseLinearInterpolate(
            list(Ts), [5.13 / 100, 5.13 / 100, 5.13 / 100, 5.13 / 100]
        ),
    ]
    ceq_car = [
        interpolate.PiecewiseLinearInterpolate(
            list(Ts), [15.64 / 100, 15.69 / 100, 15.75 / 100, 15.83 / 100]
        ),
        interpolate.PiecewiseLinearInterpolate(
            list(Ts), [7.25e-6 / 100, 2.92e-5 / 100, 9.48e-5 / 100, 2.97e-4 / 100]
        ),
    ]
    Cf_car = interpolate.PiecewiseLinearInterpolate(list(Ts), [1.0, 1.0, 0.3, 0.03])

    carbide = hucocks.HuCocksPrecipitationModel(
        c0_car,
        cp_car,
        ceq_car,
        am_car,
        N0_car,
        Vm_car,
        chi_car,
        D0_car,
        Q0_car,
        Cf_car,
        w=w,
    )

    am_laves = 3.6e-10
    N0_laves = 5e14
    Vm_laves = 2e-6
    chi_laves = 0.25
    D0_laves = 7.4e-4
    Q0_laves = 283e3
    c0_laves = [2.33 / 100.0]
    cp_laves = [50.0 / 100.0]
    ceq_laves = [
        interpolate.PiecewiseLinearInterpolate(
            list(Ts), [0.25 / 100, 0.46 / 100.0, 0.76 / 100.0, 1.16 / 100.0]
        )
    ]
    Cf_laves = 1.0

    laves = hucocks.HuCocksPrecipitationModel(
        c0_laves,
        cp_laves,
        ceq_laves,
        am_laves,
        N0_laves,
        Vm_laves,
        chi_laves,
        D0_laves,
        Q0_laves,
        Cf_laves,
        w=w,
    )

    ap = 0.84
    ac = 0.000457

    tau_model = hucocks.HuCocksHardening(strengthmodel, [carbide, laves], ap, ac, b, G)


    g0 = 1.0
    n = 20.0

    N = 100

    nthreads = 4

    slipmodel = sliprules.PowerLawSlipRule(tau_model, g0, n)
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
