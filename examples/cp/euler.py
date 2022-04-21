#!/usr/bin/env python3

from short_extrapolate_Ti import *
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


def cubic_to_hex(A, B, C):
    h = 1 / 3 * (2 * A - B)
    k = 1 / 3 * (2 * B - A)
    i = -1 / 3 * (A + B)
    l = C

    # clearing of fractions to lowest integers
    indice = np.array([h, k, i, l])
    print("indice: ", indice)
    cf = np.amin(np.abs(indice[np.nonzero(indice)]))
    miller_indice = indice / cf
    print("miller_indice: ", miller_indice)
    
    return (
        int(miller_indice[0]),
        int(miller_indice[1]),
        int(miller_indice[2]),
        int(miller_indice[3]),
    )

if __name__ == "__main__":

    # crystal structure
    hcp = True

    # generate the euler angle for single crystal model
    A = input("crystal axis index one  ")
    B = input("crystal axis index one  ")
    C = input("crystal axis index one  ")
    A, B, C = float(A), float(B), float(C)
    u = tensors.Vector([0, 0, 1])
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
    print("psi = ", psi)
    print("theta = ", theta)
    print("phi = ", phi)

    # for copy data from terminal
    print(psi, theta, phi)

    N = 1

    orientations = np.array(
        [
            rotations.CrystalOrientation(
                psi, theta, phi, angle_type="degrees", convention="kocks"
            )
        ]
        * N
    )

    temp_ori = rotations.CrystalOrientation(
        psi, theta, phi, angle_type="degrees", convention="kocks"
    )
    print(temp_ori.quat)

    if hcp:
        # Model
        a = 2.9511 * 0.1  # nm
        c = 4.68433 * 0.1  # nm
        # Sets up the lattice crystallography
        lattice = crystallography.HCPLattice(a, c)

        # Plots an initial basal pole figure
        h, k, i, l = cubic_to_hex(A, B, C)
        print(h, k, i, l)
        polefigures.pole_figure_discrete(orientations, [1, 1, -2, 6], lattice)
        plt.title("Initial, <{}{}{}{}>".format(h, k, i, l))
        plt.show()
    else:
        lattice = crystallography.CubicLattice(1.0)
        polefigures.pole_figure_discrete(
            orientations, [int(A), int(B), int(C)], lattice
        )
        plt.title("Initial, <{}{}{}>".format(int(A), int(B), int(C)))
        plt.show()
        plt.close()
