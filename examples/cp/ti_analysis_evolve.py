#!/usr/bin/env python3

import sys
from Ti_maker import *
from neml import drivers
from ti_texture import *
import numpy as np
import numpy.linalg as la
import numpy.random as ra
import scipy.interpolate as inter
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd

import time
import concurrent.futures
from multiprocessing import Pool
from optimparallel import minimize_parallel

from scipy.optimize import curve_fit

from neml.cp import (
    hucocks,
    crystallography,
    sliprules,
    slipharden,
    inelasticity,
    kinematics,
    singlecrystal,
    polycrystal,
)
from neml.math import rotations

# ================================================#
def make_Ti_polycrystal(N, nthreads):
    # ================================================#
    smodel = Ti_singlecrystal(
        verbose=True, PTR=True, return_hardening=False, update_rotation=True
    )

    path = "/home/tianju.chen/neml/examples/cp/"
    orientations = load_texture_input(path)
    # orientations = rotations.random_orientations(N)

    model = polycrystal.TaylorModel(smodel, orientations, nthreads=nthreads)

    return model


# ================================================#
def load_file(path, T, rate):
    # ================================================#
    fnames = glob.glob(path + "*.csv")
    for f in fnames:
        strain_rate = os.path.basename(f).split("_")[0]
        if strain_rate == rate:
            temp = os.path.basename(f).split("_")[1].split(".")[0]
        if strain_rate == rate and temp == str(int(T)) + "k":
            df = pd.read_csv(
                f, usecols=[0, 1], names=["True_strain", "True_stress"], header=None
            )
            return df


# ================================================#
def history_plot(history_var, xname, yname, fname):
    # ================================================#
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
    plt.savefig("{}-{}.pdf".format(fname, T), dpi=300)
    # plt.show()
    return plt.close()


# ================================================#
def accumulate_history(start_index, end_index, store_history, accum=True):
    # ================================================#
    save_index = np.arange(start_index, end_index)
    num_niternal = store_history.shape[-1]

    if accum:
        accum_hist_var = np.zeros(12)
        for j in save_index:
            for i in range(num_niternal):
                if i % num_niternal == j:
                    accum_hist_var[j - start_index] += store_history[-1, i]

    else:
        accum_hist_var = np.zeros((store_history.shape[0], 12))
        for j in save_index:
            for i in range(num_niternal):
                if i % num_niternal == j:
                    accum_hist_var[:, j - start_index] += store_history[:, i]

    return accum_hist_var


if __name__ == "__main__":

    # set up model grains and threads
    Ngrains = 500
    nthreads = 30
    # tensile conditions
    rate = "8.33e-5"
    emax = 0.2
    erate = float(rate)
    Ts = np.array(
        [298.0, 423.0, 523.0, 623.0, 773.0, 873.0, 973.0, 1023.0, 1073.0, 1173.0]
    )

    tmodel = make_Ti_polycrystal(Ngrains, nthreads)
    c_dir = np.array([-1, 0, 0, 0, 0, 0])
    t_dir = np.array([1, 0, 0, 0, 0, 0])
    dirs = [t_dir, c_dir]
    prefixs = ["tension", "compression"]
    full_res = True

    for T in Ts:
        for l_dir, prefix in zip(dirs, prefixs):
            print(T, l_dir, prefix)
            res = drivers.uniaxial_test(
                tmodel,
                erate=erate,
                emax=emax,
                sdir=l_dir,
                T=T,
                verbose=True,
                full_results=full_res,
            )

            if full_res:

                store_history = np.array(res["history"])
                print("shape of history is:", store_history.shape)

                # calculate internal dislocation density
                accu_density = accumulate_history(8, 20, store_history)
                print(accu_density)

                # calculate internal accumulated twin strain
                accu_twin = accumulate_history(20, 32, store_history)
                print(accu_twin)

                # calculate internal accumulated twin strain
                accu_slip = accumulate_history(32, 44, store_history)
                print(accu_slip)

                data = pd.DataFrame(
                    {
                        "dis_density": accu_density**2 * 1.0e12,
                        "accu_twin": accu_twin,
                        "accu_slip": accu_slip,
                    }
                )
                data.to_csv("{}_res_{}.csv".format(prefix, int(T)))

                evolve_density = accumulate_history(8, 20, store_history, accum=False)
                evolve_twin = accumulate_history(20, 32, store_history, accum=False)
                evolve_slip = accumulate_history(32, 44, store_history, accum=False)
                print("evolve dd is:", evolve_density)

                data = pd.DataFrame(
                    {
                        "evolve_density_1": evolve_density[:, 0] ** 2 * 1.0e12,
                        "evolve_density_2": evolve_density[:, 1] ** 2 * 1.0e12,
                        "evolve_density_3": evolve_density[:, 2] ** 2 * 1.0e12,
                        "evolve_density_4": evolve_density[:, 3] ** 2 * 1.0e12,
                        "evolve_density_5": evolve_density[:, 4] ** 2 * 1.0e12,
                        "evolve_density_6": evolve_density[:, 5] ** 2 * 1.0e12,
                        "evolve_density_7": evolve_density[:, 6] ** 2 * 1.0e12,
                        "evolve_density_8": evolve_density[:, 7] ** 2 * 1.0e12,
                        "evolve_density_9": evolve_density[:, 8] ** 2 * 1.0e12,
                        "evolve_density_10": evolve_density[:, 9] ** 2 * 1.0e12,
                        "evolve_density_11": evolve_density[:, 10] ** 2 * 1.0e12,
                        "evolve_density_12": evolve_density[:, 11] ** 2 * 1.0e12,
                        "evolve_twin_1": evolve_twin[:, 0],
                        "evolve_twin_2": evolve_twin[:, 1],
                        "evolve_twin_3": evolve_twin[:, 2],
                        "evolve_twin_4": evolve_twin[:, 3],
                        "evolve_twin_5": evolve_twin[:, 4],
                        "evolve_twin_6": evolve_twin[:, 5],
                        "evolve_twin_7": evolve_twin[:, 6],
                        "evolve_twin_8": evolve_twin[:, 7],
                        "evolve_twin_9": evolve_twin[:, 8],
                        "evolve_twin_10": evolve_twin[:, 9],
                        "evolve_twin_11": evolve_twin[:, 10],
                        "evolve_twin_12": evolve_twin[:, 11],
                        "evolve_slip_1": evolve_slip[:, 0],
                        "evolve_slip_2": evolve_slip[:, 1],
                        "evolve_slip_3": evolve_slip[:, 2],
                        "evolve_slip_4": evolve_slip[:, 3],
                        "evolve_slip_5": evolve_slip[:, 4],
                        "evolve_slip_6": evolve_slip[:, 5],
                        "evolve_slip_7": evolve_slip[:, 6],
                        "evolve_slip_8": evolve_slip[:, 7],
                        "evolve_slip_9": evolve_slip[:, 8],
                        "evolve_slip_10": evolve_slip[:, 9],
                        "evolve_slip_11": evolve_slip[:, 10],
                        "evolve_slip_12": evolve_slip[:, 11],
                    }
                )
                data.to_csv("{}_hist_{}.csv".format(prefix, int(T)))

                # plot distribution of dislocation density
                _ = history_plot(
                    accu_density**2 * 1.0e12,
                    "Slip System",
                    "Accumulated Dislocation Density",
                    "{}-dislocation-density".format(prefix),
                )

                # plot distribution of accumulated twin strain
                _ = history_plot(
                    accu_twin,
                    "Twin System",
                    "Accumulated Twin Strain",
                    "{}-twin-strain".format(prefix),
                )

                # plot distribution of accumulated slip strain
                _ = history_plot(
                    accu_slip,
                    "Slip System",
                    "Accumulated Slip Strain",
                    "{}-slip-strain".format(prefix),
                )

            else:
                plt.plot(res["strain"], res["stress"], "k-", lw=2)
                plt.tight_layout()
                plt.savefig("{}-{}.pdf".format(prefix, int(T)), dpi=300)
                plt.show()
                plt.close()
