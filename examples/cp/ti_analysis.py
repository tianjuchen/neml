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

    orientations = rotations.random_orientations(N)

    model = polycrystal.TaylorModel(smodel, orientations, nthreads=nthreads)

    return smodel


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


def history_plot(history_var, xname, yname, fname):

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


if __name__ == "__main__":

    # set up model grains and threads
    Ngrains = 1
    nthreads = 1
    # tensile conditions
    rate = "8.33e-5"
    emax = 0.2
    erate = float(rate)
    Ts = np.array(
        [298.0, 423.0, 523.0, 623.0, 773.0, 873.0, 973.0, 1023.0, 1073.0, 1173.0]
    )

    # path_1 = "/home/tianju.chen/RTRC_data_extract/Huang-2007-MSEA/"
    # path_2 = "/home/tianju.chen/RTRC_data_extract/Yapici-2014-MD/"

    for T in Ts:
        print(T)
        tmodel = make_Ti_polycrystal(Ngrains, nthreads)
        res = drivers.uniaxial_test(
            tmodel,
            erate=erate,
            emax=emax,
            sdir=np.array([1, 0, 0, 0, 0, 0]),
            T=T,
            verbose=True,
            full_results=False,
        )
        
        full_results = True
        
        if full_results:
        
            store_history = np.array(res["history"])
            print("shape of history is:", store_history.shape)
            
            # calculate internal dislocation density
            save_index = np.arange(8, 20)
            num_niternal = store_history.shape[-1]
            accu_density = np.zeros(12)

            for j in save_index:
                for i in range(num_niternal):
                    if i % num_niternal == j:
                        accu_density[j - 8] += store_history[-1, i]

            print(accu_density)

            # calculate internal accumulated twin strain
            save_index = np.arange(20, 32)
            accu_twin = np.zeros(12)
            for j in save_index:
                for i in range(num_niternal):
                    if i % num_niternal == j:
                        accu_twin[j - 20] += store_history[-1, i]
            print(accu_twin)

            # calculate internal accumulated twin strain
            save_index = np.arange(32, 44)
            accu_slip = np.zeros(12)
            for j in save_index:
                for i in range(num_niternal):
                    if i % num_niternal == j:
                        accu_slip[j - 32] += store_history[-1, i]
            print(accu_slip)

            
            data = pd.DataFrame(
                {
                    "strain": res["strain"],
                    "stress": res["stress"],
                    "dis_density": accu_density **2 * 1.0e12 ,
                    "accu_twin": accu_twin,
                    "accu_slip": accu_slip,
                }
            )
            data.to_csv("res_{}.csv".format(int(T)))

            # plot distribution of dislocation density
            _ = history_plot(
                accu_density,
                "Slip System",
                "Accumulated Dislocation Density",
                "dislocation-density",
            )

            # plot distribution of accumulated twin strain
            _ = history_plot(
                accu_twin, "Twin System", "Accumulated Twin Strain", "twin-strain"
            )

            # plot distribution of accumulated slip strain
            _ = history_plot(
                accu_slip, "Slip System", "Accumulated Slip Strain", "slip-strain"
            )
        
        else:
            plt.plot(res["strain"], res["stress"], 'k-', lw=2)
            plt.tight_layout()
            plt.savefig("tensile.pdf", dpi=300)
            plt.show()
            plt.close()        plt.plot(res["strain"], res["stress"], 'k-', lw=2)
            plt.tight_layout()
            plt.savefig("tensile.pdf", dpi=300)
            plt.show()
            plt.close()
