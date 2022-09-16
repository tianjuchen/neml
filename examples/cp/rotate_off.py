#!/usr/bin/env python3

# from short_extrapolate_Ti import *
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
    polefigures,
)
from neml.math import rotations, tensors


if __name__ == "__main__":

    # set up the off degree
    offset = 1.0e-4
    angle = np.zeros(3)
    logic = False
    while not logic:
        angle[0] = 360 * ra.rand(1)
        angle[1] = 180 * ra.rand(1)
        angle[2] = 360 * ra.rand(1)
        R = np.array(
            [
                [
                    -np.sin(angle[2]) * np.sin(angle[0])
                    - np.cos(angle[2]) * np.cos(angle[0]) * np.cos(angle[1]),
                    np.sin(angle[0]) * np.cos(angle[2])
                    - np.cos(angle[0]) * np.sin(angle[2]) * np.cos(angle[1]),
                    np.cos(angle[0]) * np.sin(angle[1]),
                ],
                [
                    np.cos(angle[0]) * np.sin(angle[2])
                    - np.sin(angle[0]) * np.cos(angle[2]) * np.cos(angle[1]),
                    -np.cos(angle[2]) * np.cos(angle[0])
                    - np.sin(angle[2]) * np.sin(angle[0]) * np.cos(angle[1]),
                    np.sin(angle[0]) * np.sin(angle[1]),
                ],
                [
                    np.cos(angle[2]) * np.sin(angle[1]),
                    np.sin(angle[2]) * np.sin(angle[1]),
                    np.cos(angle[1]),
                ],
            ]
        )
        new = np.dot(R, [0, 0, 1])
        alpha = np.arccos(new[2] / np.sqrt(new[0] ** 2 + new[1] ** 2 + new[2] ** 2))
        alpha = alpha / np.pi * 180.0
        if alpha >= 0 and alpha <= offset:
            logic = True

    u = tensors.Vector([0, 0, 1])  # define the crystal orientation that is rotated from and will be rotated to [0,0,1]
    v = tensors.Vector([angle[0], angle[1], angle[2]])
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

    lattice = crystallography.CubicLattice(1.0)
    lattice.add_slip_system([1, 1, 0], [1, 1, 1])

    N = 1000

    orientations = np.array(
        [
            rotations.CrystalOrientation(
                psi, theta, phi, angle_type="degrees", convention="kocks"
            )
        ]
        * N
    )

    polefigures.pole_figure_discrete(orientations, [0, 0, 1], lattice)
    plt.title("Initial, <001>")
    plt.show()
    plt.close()
