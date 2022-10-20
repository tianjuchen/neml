#!/usr/bin/env python3

import sys

sys.path.append("../..")

import os, glob
import scipy.interpolate as inter

import numpy as np
from neml import models, interpolate, elasticity, history, parse
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
from maker import hcp_singlecrystal, ti_singlecrystal
from optimparallel import minimize_parallel
import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy.random as ra
import pandas as pd
import xarray as xr
import tqdm
import warnings

warnings.filterwarnings("ignore")


def Q_maker(N, random=True):

    oripath = "/mnt/c/Users/ladmin/Desktop/argonne/neml/neml/examples/cp/"

    if random:
        orientations = rotations.random_orientations(N)
    else:
        fnames = glob.glob(oripath + "*.csv")
        orientation_angles = np.zeros((N, 3))
        for f in fnames:
            file_name = os.path.basename(f).split(".csv")[0]
            if file_name == "basal_extract":
                df = pd.read_csv(f)
                ori_1 = df["ori_1"]
                ori_2 = df["ori_2"]
                ori_3 = df["ori_3"]

                for i, (euler_1, euler_2, euler_3) in enumerate(
                    zip(ori_1, ori_2, ori_3)
                ):
                    orientation_angles[i, 0] = euler_1
                    orientation_angles[i, 1] = euler_2
                    orientation_angles[i, 2] = euler_3

            orientations = np.array(
                [
                    rotations.CrystalOrientation(
                        texture_1,
                        texture_2,
                        texture_3,
                        angle_type="degrees",
                        convention="kocks",
                    )
                    for texture_1, texture_2, texture_3 in zip(
                        orientation_angles[:, 0],
                        orientation_angles[:, 1],
                        orientation_angles[:, 2],
                    )
                ]
            )
    return orientations


ori = None


def deforming(
    erate=1.0e-2, T=298.0, N=50, orientations=ori, nthreads=20, save_xml=False
):

    smodel = ti_singlecrystal(erate=erate)
    # smodel = parse.parse_xml("update_materials.xml", "ti_cp")
    
    if save_xml:
        return smodel.save("refit.xml", "ti_cp")
    else:
        tmodel = polycrystal.TaylorModel(smodel, orientations, nthreads=nthreads)

        res = drivers.uniaxial_test(
            tmodel,
            erate=erate,
            emax=0.25,
            sdir=np.array([-1, 0, 0, 0, 0, 0]),
            T=T,
            verbose=True,
            full_results=True,
        )
        return res


def accumulate_history(res, start_index, end_index, accum=True):
    store_history = np.array(res["history"])
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


def load_file(path, T, rate):
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


if __name__ == "__main__":

    # Load the experiment database
    path_base = "/mnt/c/Users/ladmin/Desktop/argonne/RTRC_data_extract/"
    path_1 = path_base + "Huang-2007-MSEA/"
    path_2 = path_base + "Yapici-2014-MD/"

    Ts = np.array([298.0, 423.0, 523.0, 623.0, 773.0, 873.0, 973.0, 1073.0, 1173.0])

    # Set up the starting texture and grain numbers
    N = 500
    Q = Q_maker(N, random=True)
    erate = "1e-2"

    # Save xml file or not
    output_xml = True

    if output_xml:
        file = deforming(orientations=Q, save_xml=output_xml)
    else:
        for T in Ts:
            print("current temperature is:", T)
            path = "/home/tianju.chen/neml/examples/cp/twinfit/"
            save_path = os.path.join(path, str(int(T)) + "/")

            if T < 1000.0:
                path = path_1
            else:
                path = path_2

            # convert to engineering stress-strain
            df = load_file(path, T, erate)
            eng_strain = np.exp(df["True_strain"]) - 1.0
            eng_stress = df["True_stress"] / (1 + eng_strain)

            res = deforming(erate=float(erate), T=T, N=N, orientations=Q)
            twin_fraction_slip = accumulate_history(res, 8, 20)
            twin_fraction = np.sum(twin_fraction_slip)
            print("twin vloume fraction is:", twin_fraction)

            # Plotting the prediction
            plt.plot(eng_strain, eng_stress, "k-", label="experiment")
            plt.plot(res["eqstrain"], res["eqstress"], "r--", label="optimization")
            plt.xlabel("Strain (mm/mm)")
            plt.ylabel("Stress (MPa)")
            plt.legend()
            plt.savefig(save_path + "calibration-{}-{}.pdf".format(int(T), erate))
            # plt.show()
            plt.close()

            # Store the optimized parameters
            data = pd.DataFrame(
                {
                    "strain": res["eqstrain"],
                    "stress": res["eqstress"],
                    "tf": twin_fraction,
                },
                # index=[0],
            )

            data.to_csv(save_path + "flow-{}-{}.csv".format(int(T), erate))
