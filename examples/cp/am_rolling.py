#!/usr/bin/env python3

import sys

sys.path.append("../..")

import numpy as np

from neml.cp import (
    polycrystal,
    crystallography,
    slipharden,
    sliprules,
    inelasticity,
    kinematics,
    singlecrystal,
    polefigures,
    addmaf,
)
from neml.math import rotations, tensors, nemlmath
from neml import elasticity, drivers, history, interpolate

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # temperature levels
    Ts = np.array([25.0, 650.0]) + 273.0

    # Everyone's favorite FCC rolling example!
    N = 10

    nthreads = 1

    L = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]])
    erate = 1.0e-3  # 8.33e-6
    steps = 100
    emax = 0.3

    T = 650.0 + 273.0

    E = 200.0e3
    nu = 0.3
    mu = E / (2 * (1 + nu))
    mus = interpolate.PiecewiseLinearInterpolate(list(Ts), [mu, mu])
    M = np.array([mus] * 12)

    inis, kw1_v, kw2_v, ki1_v, ki2_v = 160.0, 0.7, 10, 0.2, 10

    kw_1 = interpolate.PiecewiseLinearInterpolate(list(Ts), [kw1_v, kw1_v])
    kw_2 = interpolate.PiecewiseLinearInterpolate(list(Ts), [kw2_v, kw2_v])
    ki_1 = interpolate.PiecewiseLinearInterpolate(list(Ts), [ki1_v, ki1_v])
    ki_2 = interpolate.PiecewiseLinearInterpolate(list(Ts), [ki2_v, ki2_v])

    kw1 = np.array([kw_1] * 12)
    kw2 = np.array([kw_2] * 12)
    ki1 = np.array([ki_1] * 12)
    ki2 = np.array([ki_2] * 12)

    g0 = 1.0
    n = 12.0

    # Setup
    L *= erate
    dt = emax / steps / erate
    orientations = rotations.random_orientations(N)

    lattice = crystallography.CubicLattice(1.0)
    lattice.add_slip_system([1, 1, 0], [1, 1, 1])

    strengthmodel = addmaf.AMModel(
        M,
        kw1,
        kw2,
        ki1,
        ki2,
        k0=10 ** (-6.9582),
        Q=113848.822,
        ftr=0.00539,
        initsigma=inis,  # 15.0 + 89.1 + 96.6 + 12.5,
        omega=588930.52,
        inibvalue=7.34e2,
        fb=0.0,
    )
    slipmodel = sliprules.PowerLawSlipRule(strengthmodel, g0, n)
    imodel = inelasticity.AsaroInelasticity(slipmodel)
    emodel = elasticity.IsotropicLinearElasticModel(E, "youngs", nu, "poissons")
    kmodel = kinematics.StandardKinematicModel(emodel, imodel)

    Q = rotations.CrystalOrientation(
        0.0, 0.0, 0.0, angle_type="degrees", convention="kocks"
    )
    model = singlecrystal.SingleCrystalModel(
        kmodel,
        lattice,
        update_rotation=True,
        verbose=True,
        linesearch=True,
        initial_rotation=Q,
        miter=100,
        max_divide=10,
    )

    pmodel = polycrystal.TaylorModel(model, orientations, nthreads=nthreads)

    res = drivers.uniaxial_test(
        model,
        erate,
        emax=emax,
        nsteps=steps,
        # sdir=sdir,
        T=T,
        miter=100,
        full_results=False,
    )

    fs = 23

    plt.plot(res["strain"], res["stress"], lw=3, label="Model")
    ax = plt.gca()
    plt.xlabel("Strain", fontsize=fs)
    plt.ylabel("Stress (MPa)", fontsize=fs)
    plt.legend()
    plt.tick_params(axis="both", which="major", labelsize=fs)
    plt.locator_params(axis="both", nbins=5)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)

    plt.tight_layout()
    # plt.grid(True)
    # plt.savefig("taylor-stress-strain-{}.pdf".format(nfile))
    plt.show()
    plt.close()

    sys.exit("stop")

    h_n = pmodel.init_store()

    d_inc = nemlmath.sym(0.5 * (L + L.T))
    w_inc = nemlmath.skew(0.5 * (L - L.T))

    s_n = np.zeros((6,))
    d_n = np.zeros((6,))
    w_n = np.zeros((3,))

    u_n = 0.0
    p_n = 0.0

    for i in range(steps):
        print(i)
        d_np1 = d_n + d_inc * dt
        w_np1 = w_n + w_inc * dt
        s_np1, h_np1, A_np1, B_np1, u_np1, p_np1 = pmodel.update_ld_inc(
            d_np1, d_n, w_np1, w_n, T, T, dt, 0, s_n, h_n, u_n, p_n
        )

        d_n = np.copy(d_np1)
        w_n = np.copy(w_np1)

        s_n = np.copy(s_np1)
        h_n = np.copy(h_np1)

        u_n = u_np1
        p_n = p_np1

    polefigures.pole_figure_discrete(orientations, [1, 1, 1], lattice)
    plt.title("Initial, <111>")
    plt.show()

    polefigures.pole_figure_discrete(pmodel.orientations(h_np1), [1, 1, 1], lattice)
    plt.title("Final, <111>")
    plt.show()

    polefigures.pole_figure_discrete(orientations, [1, 0, 0], lattice)
    plt.title("Initial, <100>")
    plt.show()

    polefigures.pole_figure_discrete(pmodel.orientations(h_np1), [1, 0, 0], lattice)
    plt.title("Final, <100>")
    plt.show()
