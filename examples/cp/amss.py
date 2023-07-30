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
from scipy.optimize import curve_fit
import glob, os


def interp(x, y, xnew):
    return interpolate.interp1d(x, y)(xnew)


def load_file(path, targ_name="processed-ss316-print"):
    fnames = glob.glob(path + "*.csv")
    for f in fnames:
        fname = os.path.basename(f).split(".csv")[0]
        if fname == targ_name:
            df = pd.read_csv(f)
            return df


def amodel(inits, kw1_v, kw2_v, ki1_v, ki2_v, T=650.0 + 273.0):
    sdir = np.array([1, 0, 0, 0, 0, 0])
    hours = 500
    steps = 500
    emax = 0.03  # np.log(1 + 0.5)  # 0.25
    erate = emax / 8.33e-6  # (3600 * hours)  # 8.33e-5  # 1.0e-4
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
        k0=10 ** (-6.9582),
        Q=113848.822,
        ftr=0.00539,
        initsigma=inits, #15.0 + 89.1 + 96.6 + 12.5,
        omega=588930.52,
        inibvalue=7.34e2,
        fb=0.0,
        #iniwvalue=iniw,
    )

    g0 = 1.0
    n = 20.0

    N = 20
    Nthreads = 40

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
    pmodel = polycrystal.TaylorModel(smodel, orientations, nthreads=Nthreads)
    res = drivers.uniaxial_test(
        smodel,
        erate,
        emax=emax,
        nsteps=steps,
        sdir=sdir,
        T=T,
        miter=100,
        full_results=False,
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
    x_exp = np.array([0, 5 * 3600, 25 * 3600, 100 * 3600, 501 * 3600])
    # y_exp = np.array([7.34e-7, 8.14e-7, 7.7e-7, 9.25e-7]) * uf
    y_exp = np.array([7.34e-7, 7.75e-7, 6.22e-7, 6.30e-7, 8.49e-7]) * uf
    new_y = interp(x_exp, y_exp, new_x)
    return new_x, new_y


def ssres(new_strain, file):
    path = "/home/tianju.chen/neml/examples/cp/"
    df = load_file(path, targ_name=file)
    strain = np.array(df["strain"])
    stress = np.array(df["stress"])
    new_strain = np.array(new_strain)
    new_strain[new_strain < 0] = 0.0
    new_stress = interp(strain, stress, new_strain)
    return df, new_strain, new_stress


def calcomega(T):
    b = 0.256
    kb = 13806.49
    d0 = 7.34e2
    f0 = 0.18
    E = 200.0e3
    nu = 0.3
    mu = E / (2 * (1 + nu))
    return f0 * kb * T * d0 / (mu * b**3.0)


def vf(d, omega, T=650.0 + 273.0):
    E = 200.0e3
    nu = 0.3
    mu = E / (2 * (1 + nu))
    b = 0.256
    kb = 13806.49
    return mu * omega * b**3 / (kb * T * d)


if __name__ == "__main__":
    nfile = "processed-ss316-print-650c"

    inis_range = [90.0, 300.0]
    kw1_range = [5.0e-7, 2.0]
    kw2_range = [1.0, 1.0e4]
    ki1_range = [5.0e-9, 1.0]
    ki2_range = [1.0, 1.0e4]
    #n_range = [5.0, 30.0]
    #g0_range = [1.0, 5.0]
    iniw_range = [5.0e-10, 5.0e-6]

    p0 = [
        ra.uniform(*inis_range),
        ra.uniform(*kw1_range),
        ra.uniform(*kw2_range),
        ra.uniform(*ki1_range),
        ra.uniform(*ki2_range),
        #ra.uniform(*n_range),
        #ra.uniform(*g0_range),
        #ra.uniform(*iniw_range),
    ]

    def R(params):
        smodel, lattice, res = amodel(*params)
        _, _, obs = ssres(res["strain"], nfile)
        R = la.norm(np.array(res["stress"]) - obs)
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
                inis_range,
                kw1_range,
                kw2_range,
                ki1_range,
                ki2_range,
                #n_range,
                #g0_range,
                #iniw_range,
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
    df, _, fobs = ssres(final_res["strain"], nfile)

    plt.plot(df["strain"], df["stress"], label="Exp")
    plt.plot(final_res["strain"], final_res["stress"], label="Model")
    plt.xlabel("Strain")
    plt.ylabel("Stress (MPa)")
    plt.legend()
    # plt.grid(True)
    plt.savefig("stress-strain-{}.pdf".format(nfile))
    plt.show()
    plt.close()

    data = pd.DataFrame(
        {
            "inis": res.x[0],
            "kw1": res.x[1],
            "kw2": res.x[2],
            "ki1": res.x[3],
            "ki2": res.x[4],
            #"n": res.x[5],
            #"g0": res.x[6],
            #"iniw": res.x[4],
        },
        index=[0],
    )
    data.to_csv("params-{}.csv".format(nfile))
