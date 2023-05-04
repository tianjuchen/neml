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
    addmaf,
)
from neml.math import rotations, tensors, nemlmath
from neml import elasticity

import matplotlib.pyplot as plt


def sgmf(d, dc, c):
    return 1 / (1 + np.exp(-c * (d / dc - 1)))


if __name__ == "__main__":
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

    strengthmodel = addmaf.AMModel(
        M,
        kw1,
        kw2,
        ki1,
        ki2,
        # b=b,
        # dc=dc,
        # c=c,
        # k0=k0,
        # kb=kb,
        # lamda=lamda,
        # omega=omega,
        # alpha_w=alphaw,
        # alpha_i=alphai,
        ftr=ftr,
        initsigma = 75.0
        # iniwvalue=5.0e-6,
        # iniivalue=1.0e-8,
        # inibvalue=5.5e2,
    )

    L = np.array([[-0.5, 0, 0], [0, 1.0, 0], [0, 0, -0.5]])
    # L = np.array([[0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])
    erate = 1.0e-4
    steps = 25
    emax = 0.5

    g0 = 1.0
    n = 20.0

    # Setup
    L *= erate
    dt = emax / steps / erate

    d = nemlmath.sym(0.5 * (L + L.T))
    w = nemlmath.skew(0.5 * (L - L.T))

    # strengthmodel = slipharden.VoceSlipHardening(ts, b, t0)
    slipmodel = sliprules.PowerLawSlipRule(strengthmodel, g0, n)
    imodel = inelasticity.AsaroInelasticity(slipmodel)
    emodel = elasticity.IsotropicLinearElasticModel(E, "youngs", nu, "poissons")
    kmodel = kinematics.StandardKinematicModel(emodel, imodel)

    lattice = crystallography.CubicLattice(1.0)
    lattice.add_slip_system([1, 1, 0], [1, 1, 1])

    model = singlecrystal.SingleCrystalModel(kmodel, lattice, verbose=False)

    T = 300.0

    h_n = model.init_store()

    s_n = np.zeros((6,))

    d_n = np.zeros((6,))
    w_n = np.zeros((3,))

    u_n = 0.0
    p_n = 0.0
    t_n = 0.0
    T_n = T

    e = [0.0]
    s = [0.0]

    for i in range(steps):
        print(i)
        d_np1 = d_n + d * dt
        w_np1 = w_n + w * dt
        t_np1 = t_n + dt
        T_np1 = T_n

        print("")
        print("internals are :")
        print(h_n)

        print("")
        print("wall fraction is :")
        f = omega * mu * b**3 / (kb * T) * 1 / h_n[8]
        # dd = omega * mu * b**3 / (kb * T) * (1 / 0.25)
        print(f)

        print("")
        print("crss is :")
        crss = alphaw * mu * b * np.sqrt(h_n[9]) * f * (
            1 - sgmf(h_n[8], dc, c)
        ) + alphai * mu * b * np.sqrt(h_n[21]) * (1 - f * (1 - sgmf(h_n[8], dc, c)))
        print(crss)

        s_np1, h_np1, A_np1, B_np1, u_np1, p_np1 = model.update_ld_inc(
            d_np1, d_n, w_np1, w_n, T_np1, T_n, t_np1, t_n, s_n, h_n, u_n, p_n
        )

        e.append(d_np1[1])
        s.append(s_np1[1])

        d_n = np.copy(d_np1)
        w_n = np.copy(w_np1)
        s_n = np.copy(s_np1)
        h_n = np.copy(h_np1)
        t_n = t_np1
        T_n = T_np1
        u_n = u_np1
        p_n = p_np1

    print(s[-1])
    plt.plot(e, s)
    plt.show()
