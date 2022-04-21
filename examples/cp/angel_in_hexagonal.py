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


def angle_between_cubic_in_hex(dir_1, dir_2, a=2.9511, c=4.68433):

    u1, v1, w1 = dir_1[0], dir_1[1], dir_1[2]
    u2, v2, w2 = dir_2[0], dir_2[1], dir_2[2]

    dot_product = (
        a**2 * (u1 * u2 + v1 * v2 - 1 / 2 * (u1 * v2 + v1 * u2)) + c**2 * w1 * w2
    )
    norm_1 = a**2 * (u1**2 - u1 * v1 + v1**2) + (c**2) * (w1**2)
    norm_2 = a**2 * (u2**2 - u2 * v2 + v2**2) + (c**2) * (w2**2)

    theta = np.arccos(dot_product / (np.sqrt(norm_1) * np.sqrt(norm_2)))

    return theta / np.pi * 180


def angle_between_hex_in_hex(dir_1, dir_2, a=2.9511, c=4.68433):

    u1, v1, t1, w1 = dir_1[0], dir_1[1], dir_1[2], dir_1[3]
    u2, v2, t2, w2 = dir_2[0], dir_2[1], dir_2[2], dir_2[3]

    dot_product = (
        a**2 * (3 * (u1 * u2 + v1 * v2) + 3 / 2 * (u1 * v2 + v1 * u2))
        + c**2 * w1 * w2
    )
    norm_1 = 3 * a**2 * (u1**2 + u1 * v1 + v1**2) + (c**2) * (w1**2)
    norm_2 = 3 * a**2 * (u2**2 + u2 * v2 + v2**2) + (c**2) * (w1**2)

    theta = np.arccos(dot_product / (np.sqrt(norm_1) * np.sqrt(norm_2)))

    return theta / np.pi * 180


if __name__ == "__main__":

    logic = False
    dir_cal = np.zeros(3)
    dir_given = np.array([1, 1, 2])
    while not logic:
        dir_cal[0] = ra.randint(low=1, high=10)
        dir_cal[1] = ra.randint(low=1, high=10)
        dir_cal[2] = ra.randint(low=1, high=10)
        theta = angle_between_cubic_in_hex(dir_cal, dir_given)
        if theta == 1.0:
            logic = True

    print("The perpendicular direction is: ", dir_cal)

    hex_dir_cal = cubic_to_hex(dir_cal[0], dir_cal[1], dir_cal[2])

    print("")
    print("The perpendicular hex direction is: ", hex_dir_cal)

    cubic_theta = angle_between_cubic_in_hex(dir_cal, dir_given)

    hex_theta = angle_between_hex_in_hex(
        hex_dir_cal, cubic_to_hex(dir_given[0], dir_given[1], dir_given[2])
    )

    print("cubic theta is:", cubic_theta)
    print("")
    print("hex theta is:", hex_theta)
