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


def hcp_singlecrystal(Q, verbose=False, update_rotation=True, return_slip=False):

    tau0 = np.array(
        [1000000.0] * 3
        + [10.0] * 1
        + [1000000.0] * 1
        + [1000000.0] * 1
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
    slipmodel = sliprules.PowerLawSlipRule(strength, g0, n)
    imodel = inelasticity.AsaroInelasticity(slipmodel)
    emodel = elasticity.IsotropicLinearElasticModel(E, "youngs", nu, "poissons")
    kmodel = kinematics.StandardKinematicModel(emodel, imodel)
    twinner = postprocessors.PTRTwinReorientation(twin_threshold)

    smodel = singlecrystal.SingleCrystalModel(
        kmodel,
        lattice,
        # initial_rotation=Q,
        update_rotation=update_rotation,
        postprocessors=[],
        verbose=verbose,
        linesearch=True,
        miter=100,
        max_divide=10,
    )

    if return_slip:
        return smodel, slipmodel, lattice, strength
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


if __name__ == "__main__":
    Q = rotations.CrystalOrientation(
        30.0, 40.0, 50.0, angle_type="degrees", convention="kocks"
    )
    smodel, slip_model, lattice, hmodel = hcp_singlecrystal(
        Q, update_rotation=False, return_slip=True
    )

    # for g in range(lattice.ngroup):
    # for j in range(lattice.nslip(g)):
    # print(g, j)
    # sys.exit("stop")

    # tensile conditions
    erate = 8.33e-5
    emax = 0.5
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

    nt = len(res["history"])

    direct_from_model = np.zeros((nt, lattice.ntotal))
    integrated_ourselves = np.zeros((nt, lattice.ntotal))
    taus_from_model = np.zeros((nt, lattice.ntotal))
    slip_rates = np.zeros((nt, lattice.ntotal))
    resolved_shear_stresses = np.zeros((nt, lattice.ntotal))

    for i in range(1, len(res["history"])):
        hist = history.History()
        smodel.populate_history(hist)
        hist.set_data(res["history"][i])
        stress = tensors.Symmetric(usym(res["stress"][i]))
        T = res["temperature"][i]
        Q = hist.get_orientation("rotation")

        fixed = history.History()

        dt = res["time"][i] - res["time"][i - 1]

        for g in range(lattice.ngroup):
            for j in range(lattice.nslip(g)):
                slip_rate = slip_model.slip(g, j, stress, Q, hist, lattice, T, fixed)
                slip_rates[i, lattice.flat(g, j)] = slip_rate
                resolved_shear_stresses[i, lattice.flat(g, j)] = lattice.shear(
                    g, j, Q, stress
                )
                # if slip_rate < 0.0:
                # raise ValueError('Negative slip rate')
                integrated_ourselves[i, lattice.flat(g, j)] = (
                    integrated_ourselves[i - 1, lattice.flat(g, j)]
                    + np.abs(slip_rate) * dt
                )
                direct_from_model[i, lattice.flat(g, j)] = hist.get_scalar(
                    "slip" + str(lattice.flat(g, j))
                )
                taus_from_model[i, lattice.flat(g, j)] = hmodel.hist_to_tau(
                    g, j, hist, lattice, T, fixed
                )

    plt.plot(integrated_ourselves)
    plt.plot(direct_from_model, ls="--")
    plt.show()
    plt.close()

    for g in range(lattice.ngroup):
        for j in range(lattice.nslip(g)):
            i = lattice.flat(g, j)
            plt.plot(
                resolved_shear_stresses[:, i],
                ls="-",
                label="rss evolution of slip {} in group {}".format(i, g),
            )
        plt.legend()
        plt.show()
        plt.close()

    for g in range(lattice.ngroup):
        for j in range(lattice.nslip(g)):
            i = lattice.flat(g, j)
            plt.plot(
                taus_from_model[:, i],
                ls="-",
                label="crss evolution of slip {} in group {}".format(i, g),
            )
        plt.legend()
        plt.show()
        plt.close()

    for g in range(lattice.ngroup):
        for j in range(lattice.nslip(g)):
            i = lattice.flat(g, j)
            plt.plot(
                slip_rates[:, i],
                ls="-",
                label="slip rate evolution of slip {} in group {}".format(i, g),
            )
        plt.legend()
        plt.show()
        plt.close()

    for g in range(lattice.ngroup):
        for j in range(lattice.nslip(g)):
            i = lattice.flat(g, j)
            plt.plot(
                direct_from_model[:, i], ls="-", lw=2, label="direct-slip-{}".format(i)
            )
            plt.plot(
                integrated_ourselves[:, i],
                ls="--",
                lw=2,
                label="integrate-slip-{}".format(i),
            )
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Step", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        # plt.legend(labels=["{}".format(yname)], prop={"size": 14}, frameon=False)
        plt.grid(False)
        plt.tight_layout()
        plt.legend()
        plt.show()
        plt.close()

    _ = history_plot(
        direct_from_model[-1, :12],
        "Slip System",
        "Accumulated Slip Strain",
        "slip-strain",
        T,
    )

    _ = history_plot(
        direct_from_model[-1, 12:],
        "Twin System",
        "Accumulated Twin Strain",
        "twin-strain",
        T,
    )

    _ = history_plot(
        taus_from_model[-1, :],
        "slip/twin System",
        "hardening",
        "hardening-evolution",
        T,
    )
