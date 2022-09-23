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
    Q, verbose=False, update_rotation=True, PTR=True, return_isv=False
):

    # temperature levels
    Ts = np.array([298.0, 423.0, 523.0, 623.0, 773.0, 873.0, 973.0, 1073.0, 1173.0])
    # unit transformer
    ut = 1.0e-3

    # Model
    a = 2.9511 * 0.1 * ut  # nm
    c = 4.68433 * 0.1 * ut  # nm

    # Elastic constants in MPa
    C11 = interpolate.PiecewiseLinearInterpolate(
        list(Ts),
        [
            162400.0,
            155100.0,
            149500.0,
            144200.0,
            136800.0,
            132200.0,
            127600.0,
            123100.0,
            119600.0,
        ],
    )
    C33 = interpolate.PiecewiseLinearInterpolate(
        list(Ts),
        [
            180700.0,
            175300.0,
            171500.0,
            167800.0,
            162700.0,
            159300.0,
            156000.0,
            152900.0,
            150400.0,
        ],
    )
    C44 = interpolate.PiecewiseLinearInterpolate(
        list(Ts),
        [
            46700.0,
            44400.0,
            42400.0,
            40300.0,
            37000.0,
            34800.0,
            32600.0,
            30700.0,
            29100.0,
        ],
    )
    C12 = interpolate.PiecewiseLinearInterpolate(
        list(Ts),
        [
            92000.0,
            94300.0,
            96100.0,
            97300.0,
            98500.0,
            99100.0,
            99300.0,
            99600.0,
            99600.0,
        ],
    )
    C13 = interpolate.PiecewiseLinearInterpolate(
        list(Ts),
        [
            69000.0,
            69500.0,
            69200.0,
            69100.0,
            68800.0,
            68800.0,
            68800.0,
            68800.0,
            68800.0,
        ],
    )

    # Constant part of the strength for slip and twin
    taus_1 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), [200.0, 145.0, 100.0, 70.0, 53.0, 38.0, 18.0, 15.0, 9.0]
    )
    taus_2 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), [120.0, 70.5, 60.0, 60.0, 43.0, 38.0, 18.0, 15.0, 9.0]
    )
    taus_3 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), [230.0, 185.0, 145.0, 110.0, 87.0, 61.0, 35.0, 20.0, 11.0]
    )
    taut_1 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), [200.0, 170.0, 160.0, 150.0, 120.0, 110.0, 100.0, 100.0, 100.0]
    )
    taut_2 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), [250.0, 240.0, 230.0, 210.0, 200.0, 190.0, 180.0, 180.0, 180.0]
    )
    # tau0 = np.array([170.0]*3+[90.5]*3+[210]*6+[180.0]*6+[250.0]*6)
    # tau0 = np.array(
    # [taus_1] * 3 + [taus_2] * 3 + [taus_3] * 6 + [taut_1] * 6 + [taut_2] * 6
    # )

    tau0 = np.array(
        [1000000.0] * 3 + [120.0] * 3 + [100000.0] * 6 + [200.0] * 6 + [250.0] * 6
    )
    # Reference slip rate and rate sensitivity exponent
    g0 = 1.0
    # if strain rate is 1e-2
    # n = 12.0
    # if strain rate is 1e-3
    # n = 7.5
    applied_rate = 8.33e-5
    rates_control = np.array([1e-3, 1e-2])
    sense_control = np.array([7.5, 12.0])
    n = extrapolate(rates_control, sense_control).value(applied_rate)

    # Twin threshold
    twin_threshold = 0.25

    # Sets up the lattice crystallography
    lattice = crystallography.HCPLattice(a, c)
    # Basal <a>
    lattice.add_slip_system([1, 1, -2, 0], [0, 0, 0, 1])
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

    # Sets up the interaction matrix
    num_basal, num_prism, num_pyram = 3, 3, 6
    num_ttwin, num_ctwin = 6, 6
    ## basal plane
    C1_single = np.array([50.0] * num_ttwin + [50.0] * num_ttwin)
    C1 = np.stack([C1_single for _ in range(num_basal)])
    ## prismatic plane
    C2_single = np.array([100.0] * num_ttwin + [1000.0] * num_ttwin)
    C2 = np.stack([C2_single for _ in range(num_prism)])
    ## pyramidal plane
    C3_single = np.array([250.0] * num_ttwin + [170.0] * num_ttwin)
    C3 = np.stack([C3_single for _ in range(num_pyram)])
    ## stack up each slip system
    C_np = np.vstack((C1, C2, C3)).T

    C_st = matrix.SquareMatrix(12, type="dense", data=C_np.flatten())

    # source from Fracture of Titanium Alloys at High Strain Rates and under Stress Triaxiality
    # calculate temperature depdendent shear modulus of slip systems  u = 39.61-0.03223*T
    mu_slip = interpolate.PiecewiseLinearInterpolate(
        list(Ts),
        [
            35200.0,
            30400.0,
            26700.0,
            23400.0,
            19100.0,
            16600.0,
            14200.0,
            11800.0,
            10000.0,
        ],
    )
    # calculate temperature depdendent shear modulus of twin systems  u = 34.605-0.03223*T
    mu_twin = interpolate.PiecewiseLinearInterpolate(
        list(Ts),
        [
            25000.46,
            21591.31,
            18963.42,
            16619.62,
            13565.60,
            11789.99,
            10085.41,
            8381.28,
            7102.78,
        ],
    )

    mu = np.array([mu_slip] * 12 + [mu_twin] * 12)

    k1 = np.array([1.0] * 3 + [0.25] * 3 + [5.0] * 6) / ut

    k2_1 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), [250.0, 280.0, 330.0, 450.0, 500.0, 600.0, 1020.0, 1200.0, 1400.0]
    )
    k2_2 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), [250.0, 280.0, 330.0, 450.0, 500.0, 600.0, 1020.0, 1200.0, 1400.0]
    )
    k2_3 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), [250.0, 280.0, 330.0, 450.0, 500.0, 600.0, 1020.0, 1200.0, 1400.0]
    )

    k2 = np.array([k2_1] * 3 + [k2_2] * 3 + [k2_3] * 6)

    # Sets up the linear elastic tensor
    emodel = elasticity.TransverseIsotropicLinearElasticModel(
        C11, C33, C12, C13, C44, "components"
    )

    # Sets up the slip system strength model (this is what you'll change)
    # strength = slipharden.FixedStrengthHardening(tau0)
    strength = slipharden.LANLTiModel(tau0, C_st, mu, k1, k2, X_s=0.9, inivalue=1.0)
    # strength = slipharden.SimpleLinearHardening(M, tau0)
    # Sets up the slip rule
    slipmodel = sliprules.PowerLawSlipRule(strength, g0, n)
    # Sets up the model inelastic rate kinematics
    imodel = inelasticity.AsaroInelasticity(slipmodel)
    # Sets up the overall model kinematics
    kmodel = kinematics.StandardKinematicModel(emodel, imodel)

    # This is the object that causes twins to recrystallize
    twinner = postprocessors.PTRTwinReorientation(twin_threshold)

    # Sets up the single crystal model
    if PTR:
        single_model = singlecrystal.SingleCrystalModel(
            kmodel,
            lattice,
            update_rotation=update_rotation,
            postprocessors=[],
            verbose=False,
            linesearch=True,
            # initial_rotation=Q,
            miter=100,
            max_divide=10,
        )
    else:
        single_model = singlecrystal.SingleCrystalModel(
            kmodel,
            lattice,
            update_rotation=update_rotation,
            verbose=False,
            linesearch=True,
            # initial_rotation=Q,
            miter=100,
            max_divide=10,
        )

    if return_isv:
        return single_model, slipmodel, lattice, strength
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
    plt.savefig(path + "{}-{}.pdf".format(fname, T), dpi=300)
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


def deformed_texture(store_history, taylor_model, T, L):

    pf = taylor_model.orientations(store_history[-1])
    polefigures.pole_figure_discrete(pf, [0, 0, 0, 1], lattice)
    plt.title("Final, <0001>")
    plt.savefig(path + "deformpf-%i-C.pdf" % int(T - 273.15), dpi=300)
    plt.show()
    plt.close()
    return pf


def plot_initial_pf(Q, T, L):
    polefigures.pole_figure_discrete(Q, [0, 0, 0, 1], lattice)
    plt.title("Initial, <0001>")
    plt.savefig(path + "initialpf-%i-C.pdf" % int(T - 273.15), dpi=300)
    plt.show()
    return plt.close()




if __name__ == "__main__":

    path = "/mnt/c/Users/ladmin/Desktop/argonne/neml/neml/examples/cp/try/"
    Q = rotations.CrystalOrientation(
        0.0, 0.0, 0.0, angle_type="degrees", convention="kocks"
    )
    smodel, slip_model, lattice, hmodel = hcp_singlecrystal(Q, return_isv=True)
    N, nthreads = 1, 1
    orientations = [Q for _ in range(N)]
    # orientations = rotations.random_orientations(N)
    tmodel = polycrystal.TaylorModel(smodel, orientations, nthreads=nthreads)
    # tensile conditions
    erate = 8.33e-5
    emax = 2.5
    T = 298.0
    t_dir = np.array([0, 0, 1, 0, 0, 0])

    # Plots an initial basal pole figure
    _ = plot_initial_pf(orientations, T, lattice)

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

    # Plots deformed basal pole figure
    pf = deformed_texture(np.array(res["history"]), tmodel, T, lattice)

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
                if lattice.flat(g, j) < 12:
                    direct_from_model[i, lattice.flat(g, j)] = hist.get_scalar(
                        "slip" + str(lattice.flat(g, j) + 24)
                    )
                else:
                    direct_from_model[i, lattice.flat(g, j)] = hist.get_scalar(
                        "twin" + str(lattice.flat(g, j))
                    )
                taus_from_model[i, lattice.flat(g, j)] = hmodel.hist_to_tau(
                    g, j, hist, lattice, T, fixed
                )

    plt.plot(integrated_ourselves)
    plt.plot(direct_from_model, ls="--")
    plt.savefig(path + "integration-compare.pdf", dpi=300)
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
        plt.savefig(path + "rss-group-{}.pdf".format(g), dpi=300)
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
        plt.savefig(path + "crss-group-{}.pdf".format(g), dpi=300)
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
        plt.savefig(path + "slip-rate-group-{}.pdf".format(g), dpi=300)
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
        plt.savefig(path + "integrate-slip-strain-group-{}.pdf".format(g), dpi=300)
        plt.show()
        plt.close()

    _ = history_plot(
        integrated_ourselves[-1, :12],
        "Slip System",
        "Accumulated Slip Strain",
        "slip-strain",
        T,
    )

    _ = history_plot(
        integrated_ourselves[-1, 12:],
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
