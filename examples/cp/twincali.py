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
from maker import hcp_singlecrystal
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


def deforming(s1, s2, s3, t1, t2, T=298.0, N=50, orientations=ori, nthreads=20):

    smodel = hcp_singlecrystal(s1, s2, s3, 156.4064, t2)

    tmodel = polycrystal.TaylorModel(smodel, orientations, nthreads=nthreads)

    res = drivers.uniaxial_test(
        tmodel,
        erate=1.0e-2,
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


def adjust_range(resx, param_range):
    iq = 0
    for index, value in enumerate(resx):
        if value == param_range[index][0]:
            param_range[index][0] = param_range[index][0] * 0.5
            iq += 1
        elif value == param_range[index][1]:
            param_range[index][1] = param_range[index][1] * 1.5
            iq += 1

    # flag = iq == 0
    flag = True

    s1_range = param_range[0]
    s2_range = param_range[1]
    s3_range = param_range[2]
    t1_range = param_range[3]
    t2_range = param_range[4]

    return (
        flag,
        s1_range,
        s2_range,
        s3_range,
        t1_range,
        t2_range,
    )


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


def interpolate(time, strain, targets):
    return inter.interp1d(time, strain)(targets)


if __name__ == "__main__":

    # Load the experiment database
    path_base = "/mnt/c/Users/ladmin/Desktop/argonne/RTRC_data_extract/"
    path_1 = path_base + "Huang-2007-MSEA/"
    path_2 = path_base + "Yapici-2014-MD/"

    T = 298.0
    path = "/mnt/c/Users/ladmin/Desktop/argonne/neml/neml/examples/cp/twinfit/"
    save_path = os.path.join(path, str(T) + "/")

    if T < 1000.0:
        path = path_1
    else:
        path = path_2

    # convert to engineering stress-strain
    df = load_file(path, T, "1e-2")
    eng_strain = np.exp(df["True_strain"]) - 1.0
    eng_stress = df["True_stress"] / (1 + eng_strain)

    # Set up the starting texture and grain numbers
    N = 50
    Q = Q_maker(N, random=True)

    # Set up parameters' ranges
    param_range = np.array(
        [
            [170.0, 200.0],  # s1
            [90.5, 120.0],  # s2
            [210.0, 230.0],  # s3
            [180.0, 200.0],  # t1
            [200.0, 250.0],  # t2
        ]
    )

    # define the range of model parameter
    s1_range = param_range[0]
    s2_range = param_range[1]
    s3_range = param_range[2]
    t1_range = param_range[3]
    t2_range = param_range[4]

    p0 = [
        ra.uniform(*s1_range),
        ra.uniform(*s2_range),
        ra.uniform(*s3_range),
        ra.uniform(*t1_range),
        ra.uniform(*t2_range),
    ]

    def R(params):
        evalues = np.linspace(0, 0.2, 100)
        res = deforming(*params, T=T, N=N, orientations=Q)
        pred = interpolate(eng_strain, eng_stress, evalues)
        actual = interpolate(res["eqstrain"], res["eqstress"], evalues)

        #tf_pred = np.concatenate(
            #(pred, np.array([np.sum(accumulate_history(res, 8, 20))]))
        #)
        #tf_actual = np.concatenate((actual, np.array([0.1])))
        R = la.norm(pred - actual)
        print("Current residual: %e" % R)
        return R

    flag = False
    iq = 0
    while not flag:
        res = minimize_parallel(
            # res = opt.minimize(
            R,
            p0,
            bounds=[
                s1_range,
                s2_range,
                s3_range,
                t1_range,
                t2_range,
            ],
            # method="L-BFGS-B",
            parallel={"max_workers": 10},
        )
        print(res.success)
        if res.success == True:
            (
                flag,
                s1_range,
                s2_range,
                s3_range,
                t1_range,
                t2_range,
            ) = adjust_range(res.x, param_range)
            if flag:
                print("")
                print("all latent params are in range")
                print("")
                print(res.success)
                print(res.x)
                print(param_range)
        iq += 1
        if iq >= 10:
            raise ValueError("Not able to optimize the initialize")

    # Evaluate the optimization
    opt_params = np.array([9.0, 4.0, 13.0, 6.0, 17.0])

    # opt_params = res.x
    res = deforming(*opt_params, T=T, N=N, orientations=Q)
    twin_fraction_slip = accumulate_history(res, 8, 20)
    twin_fraction = np.sum(twin_fraction_slip)
    print("twin vloume fraction is:", twin_fraction)

    # Plotting the prediction
    plt.plot(eng_strain, eng_stress, "k-", label="experiment")
    plt.plot(res["eqstrain"], res["eqstress"], "r--", label="optimization")
    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")
    plt.legend()
    plt.savefig(save_path + "calibration-{}.pdf".format(int(T)))
    plt.show()
    plt.close()

    # Store the optimized parameters
    data = pd.DataFrame(
        {
            "s1": opt_params[0],
            "s2": opt_params[1],
            "s3": opt_params[2],
            "s4": opt_params[3],
            "s5": opt_params[4],
        },
        index=[0],
    )

    data.to_csv(save_path + "parameters-{}.csv".format(int(T)))

