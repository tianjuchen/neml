#!/usr/bin/env python3

import sys
from ti_single_crystal_maker import Ti_singxtal
from neml import drivers, parse

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
from neml.math import rotations, tensors


def generate_euler(u=tensors.Vector([0, 0, 1])):

    A = input("crystal axis index one  ")
    B = input("crystal axis index one  ")
    C = input("crystal axis index one  ")
    A, B, C = float(A), float(B), float(C)

    v = tensors.Vector([A, B, C])

    orientations = rotations.rotate_to(u, v)

    M = orientations.to_matrix()
    theta = np.arccos(M[2, 2])
    if abs(M[2, 2]) != 1.0:
        sth = np.sin(theta)
        psi = np.arctan2(M[2, 1] / sth, M[2, 0] / sth)
        phi = np.arctan2(M[1, 2] / sth, M[0, 2] / sth)
    else:
        psi = 0
        phi = np.arctan2(-M[1, 0], -M[0, 0])

    psi = psi / np.pi * 180
    theta = theta / np.pi * 180
    phi = phi / np.pi * 180

    return psi, theta, phi


# ================================================#
def make_Ti_polycrystal(N, nthreads):
    # ================================================#

    smodel = Ti_singxtal([0, 0, 0])
    # orientations = rotations.random_orientations(N)

    theta_1, theta_2, theta_3 = generate_euler()
    orientations_1 = np.array(
        [
            rotations.CrystalOrientation(
                theta_1, theta_2, theta_3, angle_type="degrees", convention="kocks"
            )
        ]
        * int(N / 2)
    )

    theta_1, theta_2, theta_3 = generate_euler()
    orientations_2 = np.array(
        [
            rotations.CrystalOrientation(
                theta_1, theta_2, theta_3, angle_type="degrees", convention="kocks"
            )
        ]
        * int(N / 2)
    )

    orientations = np.concatenate((orientations_1, orientations_2))
    model = polycrystal.TaylorModel(smodel, orientations, nthreads=nthreads)

    # smodel.save("ti_tpsingle_check.xml", "forest")
    return model


if __name__ == "__main__":

    # set up model grains and threads
    Ngrains = 50
    nthreads = 10

    # loading conditions
    emax = 0.2
    erate = 1.0e-2
    T = 1173.0
    tmodel = make_Ti_polycrystal(Ngrains, nthreads)

    res = drivers.uniaxial_test(
        tmodel,
        erate=erate,
        emax=emax,
        sdir=np.array([0, 0, -1, 0, 0, 0]),
        T=T,
        verbose=True,
        full_results=False,
    )

    # convert to engineering stress-strain
    plt.plot(res["strain"], res["stress"], "k-", label="Model-{}".format(int(T)))
    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")
    plt.legend()
    plt.grid(True)
    # plt.savefig("Engineering-stress-strain-{}.png".format(int(T)))
    plt.show()
    plt.close()
