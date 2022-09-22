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


def hcp_singlecrystal(
    verbose=False, return_hardening=False, update_rotation=True, return_slip=False
):

    tau0 = np.array(
        [100000.0] * 3
        + [10.0]
        + [10000.0]
        + [10000.0]
        + [1000000.0] * 6
        + [1000000.0] * 6
        + [1000000.0] * 6
    )

    # Model
    a = 2.9511 * 0.1  # nm
    c = 4.68433 * 0.1  # nm

    # Sets up the lattice crystallography
    lattice = crystallography.HCPLattice(a, c)
    # Basal <a>
    lattice.add_slip_system([1, 1, -2, 0], [0, 0, 0, 1])  # direction, plane
    # Prismatic <a>
    lattice.add_slip_system([1, 1, -2, 0], [1, 0, -1, 0])
    # Pyramidal <c+a>
    lattice.add_slip_system([1, 1, -2, -3], [1, 1, -2, 2])
    # Tension twinning
    lattice.add_twin_system([-1, 0, 1, 1], [1, 0, -1, 2], [1, 0, -1, 1], [1, 0, -1, -2])
    # Compression twinning
    lattice.add_twin_system(
        [1, 1, -2, -3], [1, 1, -2, 2], [2, 2, -4, 3], [1, 1, -2, -4]
    )

    # Hardening coefficients for slip (H1) and twinning (H2)
    H1 = 10.0
    H2 = 10.0

    # Reference slip rate and rate sensitivity exponent
    g0 = 1.0
    n = 12.0

    E = 100000.0
    nu = 0.3

    twin_threshold = 0.75

    # Sets up the interaction matrix
    M = matrix.SquareMatrix(24, type="diagonal_blocks", data=[H1, H2], blocks=[12, 12])
    strength = slipharden.SimpleLinearHardening(M, tau0)
    # strength = slipharden.LANLTiModel(tau0, C_st, mu, k1, k2, X_s=0.9, inivalue=1.0)
    slipmodel = sliprules.PowerLawSlipRule(strength, g0, n)
    imodel = inelasticity.AsaroInelasticity(slipmodel)
    emodel = elasticity.IsotropicLinearElasticModel(E, "youngs", nu, "poissons")
    kmodel = kinematics.StandardKinematicModel(emodel, imodel)
    twinner = postprocessors.PTRTwinReorientation(twin_threshold)

    smodel = singlecrystal.SingleCrystalModel(
        kmodel,
        lattice,
        update_rotation=update_rotation,
        postprocessors=[],
        verbose=verbose,
        linesearch=True,
        miter=100,
        max_divide=10,
    )

    if return_hardening:
        return smodel, strengthmodel
    elif return_slip:
        return smodel, slipmodel, lattice
    else:
        return smodel


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

    # orientations = [
        # rotations.CrystalOrientation(
            # 0.0,
            # 0.0,
            # 0.0,
            # angle_type="degrees",
            # convention="kocks",
        # )
        # for _ in range(N)
    # ]
    orientations = rotations.random_orientations(N)
    smodel, slip_model, lattice = hcp_singlecrystal(return_slip=True)
    tmodel = polycrystal.TaylorModel(smodel, orientations, nthreads=nthreads)

    # tensile conditions
    rate = "8.33e-5"
    emax = 2.0
    erate = float(rate)
    T = 298.0
    t_dir = np.array([1, 0, 0, 0, 0, 0])
    full_res = True
    res = drivers.uniaxial_test(
        smodel,
        erate=erate,
        emax=emax,
        sdir=t_dir,
        T=T,
        verbose=True,
        full_results=full_res,
    )

    slip_rates = []
    if full_res:
        store_history = np.array(res["history"])

        for h in store_history:
            blank = history.History()
            hist = history.History()
            smodel.populate_history(hist)
            hist.set_data(h)
            current_orientation = smodel.get_active_orientation(hist)
            stress = smodel.strength(hist, T)
            print("current orientation:", current_orientation)
            for g in range(lattice.ngroup):
                for i in range(lattice.nslip(g)):
                    slip_rates.append(
                        slip_model.slip(
                            g,
                            i,
                            stress,
                            current_orientation,
                            hist,
                            lattice,
                            T,
                            blank,
                        )
                    )
        print("slip rates:", slip_rates)
        sys.exit("stop")

        # plot distribution of accumulated slip strain
        accu_slip = accumulate_history(8, 20, store_history)
        print(accu_slip)

        _ = history_plot(
            accu_slip,
            "Slip System",
            "Accumulated Slip Strain",
            "slip-strain",
            T,
        )

        # plot distribution of accumulated twin strain
        accu_twin = accumulate_history(20, 32, store_history)
        print(accu_twin)

        _ = history_plot(
            accu_twin,
            "Twin System",
            "Accumulated Twin Strain",
            "twin-strain",
            T,
        )
    else:
        plt.plot(res["strain"], res["stress"], "k-", lw=2)
        plt.tight_layout()
        # plt.savefig("{}_ss_{}.pdf".format(prefix, int(T)), dpi=300)
        plt.show()
        plt.close()
