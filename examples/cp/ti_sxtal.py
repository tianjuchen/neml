#!/usr/bin/env python3

import sys
from Ti_maker import *
from neml import drivers

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

    #orientations = rotations.random_orientations(N)
    
    orientations = [
        rotations.CrystalOrientation(
            0.0,
            0.0,
            0.0,
            angle_type="degrees",
            convention="kocks",
        )
        for _ in range(N)
    ]
   
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
def accumulate_history(start_index, end_index, store_history):
# ================================================#
    save_index = np.arange(start_index, end_index)
    num_niternal = store_history.shape[-1]
    accum_hist_var = np.zeros(12)
    for j in save_index:
        for i in range(num_niternal):
            if i % num_niternal == j:
                accum_hist_var[j - start_index] += store_history[-1, i]
    return accum_hist_var


def deformed_texture(store_history, taylor_model, T):

    # unit transformer
    ut = 1.0e-3

    # Model
    a = 2.9511 * 0.1 * ut  # nm
    c = 4.68433 * 0.1 * ut  # nm

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

    pf = taylor_model.orientations(store_history[-1])
    polefigures.pole_figure_discrete(pf, [0, 0, 0, 1], lattice)
    plt.title("Final, <0001>")
    plt.savefig("deformpf-%i-C-1.pdf" % int(T - 273.15), dpi=300)
    plt.show()
    plt.close()
    return pf


if __name__ == "__main__":

    # set up model grains and threads
    Ngrains = 1
    nthreads = 20
    # tensile conditions
    rate = "4.33e5" #"8.33e-5"
    emax = 2.0 #0.2
    erate = float(rate)
    Ts = np.array(
        [298.0]#, 423.0, 523.0, 623.0, 773.0, 873.0, 973.0, 1023.0, 1073.0, 1173.0]
    )

    tmodel = make_Ti_polycrystal(Ngrains, nthreads)
    c_dir = np.array([0, -1, 0, 0, 0, 0])
    t_dir = np.array([0, 0, -1, 0, 0, 0])
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

            full_results = full_res

            if full_results:
            
                pf = deformed_texture(res["history"], tmodel, T)
                
                store_history = np.array(res["history"])
                print("shape of history is:", store_history.shape)
                print(store_history)
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
                        # "strain": res["strain"],
                        # "stress": res["stress"],
                        "dis_density": accu_density**2 * 1.0e12,
                        "accu_twin": accu_twin,
                        "accu_slip": accu_slip,
                    }
                )
                data.to_csv("{}_res_{}.csv".format(prefix, int(T)))

                data_hist = pd.DataFrame(
                    {
                        "history": res["history"],
                    }
                )
                data_hist.to_csv("{}_history_{}.csv".format(prefix, int(T)))

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
                sys.exit("stop first")
            else:

                data = pd.DataFrame(
                    {
                        "strain": res["strain"],
                        "stress": res["stress"],
                    }
                )
                data.to_csv("{}_stress_strain_{}.csv".format(prefix, int(T)))

                plt.plot(res["strain"], res["stress"], "k-", lw=2)
                plt.tight_layout()
                plt.savefig("{}_ss_{}.pdf".format(prefix, int(T)), dpi=300)
                plt.show()
                plt.close()
