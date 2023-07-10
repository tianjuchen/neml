#!/usr/bin/env python3

import sys

sys.path.append("../..")

import numpy as np
import numpy.random as ra
import numpy.linalg as la
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
from neml import elasticity, drivers, history, interpolate

import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.optimize as opt
import pandas as pd


def interp(x, y, xnew):
    return interpolate.interp1d(x, y)(xnew)


def amodel(kw1_v, kw2_v, ki1_v, ki2_v, k0, Qv, ftr, T=650.0 + 273.0):

    # print(kw1_v, kw2_v, ki1_v, ki2_v, k0, Qv, ftr)
    # sys.exit("stop")

    sdir = np.array([1, 0, 0, 0, 0, 0])
    erate = 0.25 / 360000  # 8.33e-5  # 1.0e-4
    steps = 500
    emax = 0.25  # np.log(1 + 0.5)  # 0.25
    E = 200.0e3
    nu = 0.3
    mu = E / (2 * (1 + nu))
    M = np.ones((12,)) * mu

    kw1 = np.ones((12,)) * kw1_v
    kw2 = np.ones((12,)) * kw2_v
    ki1 = np.ones((12,)) * ki1_v
    ki2 = np.ones((12,)) * ki2_v

    strengthmodel = addmaf.AMModel(
        M,
        kw1,
        kw2,
        ki1,
        ki2,
        k0=10**(k0),
        #Q=Qv,
        ftr=ftr,
        initsigma=75.0,
        omega=588930.52,
        inibvalue=7.34e2,
    )

    g0 = 1.0
    n = 20.0

    N = 10
    nthreads = 20

    slipmodel = sliprules.PowerLawSlipRule(strengthmodel, g0, n)
    imodel = inelasticity.AsaroInelasticity(slipmodel)
    emodel = elasticity.IsotropicLinearElasticModel(E, "youngs", nu, "poissons")
    kmodel = kinematics.StandardKinematicModel(emodel, imodel)

    lattice = crystallography.CubicLattice(1.0)
    lattice.add_slip_system([1, 1, 0], [1, 1, 1])

    Q = rotations.CrystalOrientation(
        0.0, 0.0, 0.0, angle_type="degrees", convention="kocks"
    )

    smodel = singlecrystal.SingleCrystalModel(
        kmodel,
        lattice,
        update_rotation=True,
        verbose=False,
        linesearch=True,
        initial_rotation=Q,
        miter=100,
        max_divide=10,
    )

    orientations = rotations.random_orientations(N)
    dt = emax / erate / steps
    pmodel = polycrystal.TaylorModel(smodel, orientations, nthreads=nthreads)
    res = drivers.uniaxial_test(
        smodel,
        erate,
        emax=emax,
        nsteps=steps,
        sdir=sdir,
        T=T,
        miter=100,
        full_results=True,
    )
    return smodel, lattice, res


def usym(v):
    """
    Take a Mandel symmetric vector to the full matrix.
    """
    return np.array(
        [
            [v[0], v[5] / np.sqrt(2), v[4] / np.sqrt(2)],
            [v[5] / np.sqrt(2), v[1], v[3] / np.sqrt(2)],
            [v[4] / np.sqrt(2), v[3] / np.sqrt(2), v[2]],
        ]
    )


def hist(smodel, lattice, res):
    nt = len(res["history"])
    direct_from_model = np.zeros((nt, lattice.ntotal * 2 + 1))

    for i in range(1, len(res["history"])):
        hist = history.History()
        smodel.populate_hist(hist)
        hist.set_data(res["history"][i])
        stress = tensors.Symmetric(usym(res["stress"][i]))
        T = res["temperature"][i]
        Q = hist.get_orientation("rotation")

        fixed = history.History()

        dt = res["time"][i] - res["time"][i - 1]

        for g in range(lattice.ngroup):
            for j in range(lattice.nslip(g)):
                direct_from_model[i, 0] = hist.get_scalar("wall" + str(0))
                direct_from_model[i, lattice.flat(g, j) + 1] = hist.get_scalar(
                    "wslip" + str(lattice.flat(g, j) + 1)
                )
                direct_from_model[i, lattice.flat(g, j) + 13] = hist.get_scalar(
                    "islip" + str(lattice.flat(g, j) + 13)
                )
    return direct_from_model


def agingres(new_x, uf=1.0e9):
    x_exp = np.array([0, 5 * 3600, 25 * 3600, 100 * 3600])
    y_exp = np.array([7.34e-7, 8.14e-7, 7.7e-7, 9.25e-7]) * uf
    new_y = interp(x_exp, y_exp, new_x)
    return new_x, new_y


def calcomega(T):
    b = 0.256
    kb = 13806.49
    d0 = 7.34e2
    f0 = 0.18
    E = 200.0e3
    nu = 0.3
    mu = E / (2 * (1 + nu))
    return f0 * kb * T * d0 / (mu * b**3.0)


if __name__ == "__main__":

    # omega = calcomega(300.0)
    # print(omega)
    # sys.exit("stop")

    kw1_range = [0.8, 1.5]
    kw2_range = [10.0, 100.0]
    ki1_range = [0.08, 0.15]
    ki2_range = [10.0, 100.0]
    k0_range = [-7, -5]
    Qv_range = [1.0e2, 1.0e6]
    ftr_range = [0.0, 0.5]

    p0 = [
        ra.uniform(*kw1_range),
        ra.uniform(*kw2_range),
        ra.uniform(*ki1_range),
        ra.uniform(*ki2_range),
        ra.uniform(*k0_range),
        ra.uniform(*Qv_range),
        ra.uniform(*ftr_range),
    ]

    def R(params):
        smodel, lattice, res = amodel(*params)
        _, obs = agingres(res["time"])
        sim = hist(smodel, lattice, res)
        R = la.norm(np.array(sim)[1:, 0] - np.array(obs)[1:])
        print("Current residual: %e" % R)
        return R

    flag = False
    iq = 0
    while not flag:
        # res = minimize_parallel(
        res = opt.minimize(
            R,
            p0,
            bounds=[
                kw1_range,
                kw2_range,
                ki1_range,
                ki2_range,
                k0_range,
                Qv_range,
                ftr_range,
            ],
            method="L-BFGS-B",
        )
        print(res.success)
        if res.success == True:
            flag = True
            print(res.success)
            print(res.x)
        iq += 1
        if iq >= 10:
            raise ValueError("Not able to optimize the initialize")

    fmodel, flattice, final_res = amodel(*res.x)
    _, fobs = agingres(final_res["time"])
    fsim = hist(fmodel, flattice, final_res)

    plt.plot(np.array(final_res["time"])[1:], np.array(fobs)[1:], label="Exp")
    plt.plot(np.array(final_res["time"])[1:], np.array(fsim)[1:, 0], label="Model")
    plt.xlabel("Time")
    plt.ylabel("Wall size")
    plt.legend()
    # plt.grid(True)
    plt.savefig("optimize_initial.pdf")
    plt.show()
    plt.close()

    data = pd.DataFrame(
        {
            "kw1": res.x[0],
            "kw2": res.x[1],
            "ki1": res.x[2],
            "ki2": res.x[3],
            "k0": res.x[4],
            "Q": res.x[5],
            "ftr": res.x[6],
        },
        index=[0],
    )
    data.to_csv("optim_params.csv")
