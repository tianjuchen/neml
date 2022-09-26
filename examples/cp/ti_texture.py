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
    polycrystal,
    crystallography,
    slipharden,
    sliprules,
    inelasticity,
    kinematics,
    singlecrystal,
    polefigures,
    postprocessors,
)
from neml.math import rotations


def load_texture_input(path):

    fnames = glob.glob(path + "*.csv")
    orientation_angles = np.zeros((500, 3))
    for f in fnames:
        # temp = os.path.basename(f).split("_")[2].split(".csv")[0]
        # ldir = os.path.basename(f).split("_")[0]
        # output = os.path.basename(f).split("_")[1]
        file_name = os.path.basename(f).split(".csv")[0]
        # print("file_name:", file_name)
        if file_name == "tension_texture_298":
            df = pd.read_csv(f)
            ori_1 = df["ori_1"]
            ori_2 = df["ori_2"]
            ori_3 = df["ori_3"]

            for i, (euler_1, euler_2, euler_3) in enumerate(zip(ori_1, ori_2, ori_3)):
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

            orientations_filp = np.array(
                [
                    rotations.CrystalOrientation(
                        texture_1,
                        texture_2,
                        texture_3,
                        angle_type="degrees",
                        convention="kocks",
                    ).flip()
                    for texture_1, texture_2, texture_3 in zip(
                        orientation_angles[:, 0],
                        orientation_angles[:, 1],
                        orientation_angles[:, 2],
                    )
                ]
            )

    return orientations, orientations_filp


if __name__ == "__main__":

    path = "/mnt/c/Users/ladmin/Desktop/argonne/neml/neml/examples/cp/simple_4/simplehard/"
    orientations, orientations_filp = load_texture_input(path)

    # orientations = [
    # rotations.CrystalOrientation(
    # 140.76847951640775,
    # 65.9051574478893,
    # 26.565051177077986,
    # angle_type="degrees",
    # convention="kocks",
    # )
    # for _ in range(500)
    # ]

    # orientations_2 = [
    # rotations.CrystalOrientation(
    # 140.76847951640775,
    # 65.9051574478893,
    # 26.565051177077986,
    # angle_type="degrees",
    # convention="kocks",
    # ).flip()
    # for _ in range(500)
    # ]
    # print(orientations)
    # orientations = rotations.random_orientations(2)
    # print(orientations)

    # Model
    a = 2.9511 * 0.1  # nm
    c = 4.68433 * 0.1  # nm

    # Sets up the lattice crystallography
    lattice = crystallography.HCPLattice(a, c)

    # Plots an initial basal pole figure
    polefigures.pole_figure_discrete(
        orientations,
        [0, 0, 0, 1],
        lattice,
        x=tensors.Vector([1.0, 0, 0]),
        y=tensors.Vector([0, 1.0, 0]),
        axis_labels=["X", "Y"],
    )
    """
    polefigures.pole_figure_discrete(
        orientations_filp,
        [0, 0, 0, 1],
        lattice,
        # x=tensors.Vector([0, 1.0, 0]),
        # y=tensors.Vector([1.0, 0, 0]),
        # axis_labels=["Y", "X"],
    )
    """
    plt.title("Deformed, <0001>")
    # plt.savefig(path + "Deformed_orientation.pdf", dpi=300)
    plt.show()
    plt.close()

    polefigures.pole_figure_discrete(
        orientations,
        [1, 1, -2, 0],
        lattice,
        x=tensors.Vector([0, 1.0, 0]),
        y=tensors.Vector([1.0, 0, 0]),
        axis_labels=["Y", "X"],
    )
    """
    polefigures.pole_figure_discrete(
        orientations_filp,
        [1, 1, -2, 0],
        lattice,
        # x=tensors.Vector([0, 1.0, 0]),
        # y=tensors.Vector([1.0, 0, 0]),
        # axis_labels=["Y", "X"],
    )
    """
    plt.title("Deformed, <11-20>")
    plt.show()
    plt.close()


    polefigures.pole_figure_discrete(
        orientations,
        [1, 0, -1, 0],
        lattice,
        x=tensors.Vector([0, 1.0, 0]),
        y=tensors.Vector([1.0, 0, 0]),
        axis_labels=["Y", "X"],
    )
    """
    polefigures.pole_figure_discrete(
        orientations_filp,
        [1, 0, -1, 0],
        lattice,
        # x=tensors.Vector([0, 1.0, 0]),
        # y=tensors.Vector([1.0, 0, 0]),
        # axis_labels=["Y", "X"],
    )
    """
    plt.title("Deformed, <10-10>")
    plt.show()
    plt.close()


    polefigures.pole_figure_discrete(
        orientations,
        [1, 0, -1, 1],
        lattice,
        x=tensors.Vector([0, 1.0, 0]),
        y=tensors.Vector([1.0, 0, 0]),
        axis_labels=["Y", "X"],
    )
    """
    polefigures.pole_figure_discrete(
        orientations_filp,
        [1, 0, -1, 1],
        lattice,
        # x=tensors.Vector([0, 1.0, 0]),
        # y=tensors.Vector([1.0, 0, 0]),
        # axis_labels=["Y", "X"],
    )
    """
    plt.title("Deformed, <10-11>")
    plt.show()
    plt.close()


    sys.exit("stop")
