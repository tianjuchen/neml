#!/usr/bin/env python3

import sys

sys.path.append("../..")

import os, glob
import scipy.interpolate as inter

import numpy as np
from neml import models, interpolate, elasticity, history
from neml.cp import (
    hucocks,
    crystallography,
    sliprules,
    slipharden,
    inelasticity,
    kinematics,
    singlecrystal,
    polycrystal,
    polefigures,
    postprocessors,
)
from neml.math import rotations, tensors, nemlmath, matrix
from neml import drivers


import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy.random as ra
import pandas as pd
import xarray as xr
import tqdm
import warnings

warnings.filterwarnings("ignore")


class extrapolate:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def value(self, T):
        func = np.poly1d(np.polyfit(self.xs, self.ys, deg=1))
        return func(T).tolist()


def fcc_singlecrystal(verbose=False, return_hardening=False, update_rotation=True):

    tau0 = np.array(
        [5000.0] * 2
        + [50] * 2
        + [5000.0] * 2
        + [5000.0] * 2
        + [50.0] * 2
        + [5000.0] * 2
    )

    # tau0 = (
        # np.array([170.0] * 3 + [90.5] * 3 + [210] * 6)
        # / 10.0
    # )
    
    lattice = crystallography.CubicLattice(1.0)
    lattice.add_slip_system([1, 1, 0], [1, 1, 1])
    # print(lattice.slip_planes)

    # Hardening coefficients for slip (H1) and twinning (H2)
    H1 = 10.0
    H2 = 10.0

    # Reference slip rate and rate sensitivity exponent
    g0 = 1.0
    n = 12.0

    E = 100000.0
    nu = 0.3

    # Sets up the interaction matrix
    M = matrix.SquareMatrix(12, type="diagonal_blocks", data=[H1, H2], blocks=[6, 6])

    strength = slipharden.SimpleLinearHardening(M, tau0)
    slipmodel = sliprules.PowerLawSlipRule(strength, g0, n)
    imodel = inelasticity.AsaroInelasticity(slipmodel)
    emodel = elasticity.IsotropicLinearElasticModel(E, "youngs", nu, "poissons")
    kmodel = kinematics.StandardKinematicModel(emodel, imodel)

    smodel = singlecrystal.SingleCrystalModel(
        kmodel,
        lattice,
        update_rotation=update_rotation,
        verbose=verbose,
        linesearch=True,
        # initial_rotation = rotations.Orientation(0,0,0,angle_type="degrees"),
        miter=100,
        max_divide=10,
    )

    if return_hardening:
        return smodel, strengthmodel
    else:
        return smodel


def interp(strain, stress, targets):
    """
    This is just to make sure all our values line up, in case the model
    adaptively integrated or something
    """
    return inter.interp1d(strain, stress)(targets)


def accumulate_history(start_index, end_index, store_history):
    save_index = np.arange(start_index, end_index)
    num_niternal = store_history.shape[-1]
    accum_hist_var = np.zeros(12)
    for j in save_index:
        for i in range(num_niternal):
            if i % num_niternal == j:
                accum_hist_var[j - start_index] += store_history[-1, i]
    return accum_hist_var


def history_plot(history_var, xname, yname, fname, T):
    x = np.arange(len(history_var)) + 1
    fig, ax = plt.subplots()
    ax.bar(x, history_var, color="k")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("{}".format(xname), fontsize=14)
    plt.ylabel("{}".format(yname), fontsize=14)
    # plt.legend(labels=["{}".format(yname)], prop={"size": 14}, frameon=False)
    plt.grid(False)
    plt.tight_layout()
    # plt.savefig("{}-{}.pdf".format(fname, T), dpi=300)
    plt.show()
    return plt.close()


if __name__ == "__main__":

    N = 1
    nthreads = 1
    orientations = rotations.random_orientations(N)
    smodel = fcc_singlecrystal()
    tmodel = polycrystal.TaylorModel(smodel, orientations, nthreads=nthreads)

    # tensile conditions
    rate = "8.33e-5"
    emax = 0.2
    erate = float(rate)
    T = 298.0
    t_dir = np.array([1, 0, 0, 0, 0, 0])
    full_res = True
    res = drivers.uniaxial_test(
        tmodel,
        erate=erate,
        emax=emax,
        sdir=t_dir,
        T=T,
        verbose=True,
        full_results=full_res,
    )

    store_history = np.array(res["history"])
    print("history is:", store_history[-1])

    accu_slip = accumulate_history(8, 20, store_history)
    print(accu_slip)

    _ = history_plot(
        accu_slip,
        "Slip System",
        "Accumulated Slip Strain",
        "slip-strain",
        T,
    )

    # plt.plot(res["strain"], res["stress"], "k-", lw=2)
    # plt.tight_layout()
    # plt.savefig("{}_ss_{}.pdf".format(prefix, int(T)), dpi=300)
    # plt.show()
    # plt.close()
