#!/usr/bin/env python3
import sys
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
    cf = np.amin(np.abs(indice[np.nonzero(indice)]))
    miller_indice = indice / cf

    return (
        int(miller_indice[0]),
        int(miller_indice[1]),
        int(miller_indice[2]),
        int(miller_indice[3]),
    )


def hex_to_cubix(h, k, i, l):
    """
    convert from [U, V, T, W] to [u, v, w]
    u = 2*U + V
    v = 2*V + U
    w = W
    """
    A = 2 * h + k
    B = 2 * k + h
    C = l

    return (
        int(A),
        int(B),
        int(C),
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


def angle_between_hex_in_hex(dir_1, dir_2, a=2.9511, c=4.68433, cos_res=False):

    u1, v1, t1, w1 = dir_1[0], dir_1[1], dir_1[2], dir_1[3]
    u2, v2, t2, w2 = dir_2[0], dir_2[1], dir_2[2], dir_2[3]

    dot_product = (
        a**2 * (3 * (u1 * u2 + v1 * v2) + 3 / 2 * (u1 * v2 + v1 * u2))
        + c**2 * w1 * w2
    )
    norm_1 = 3 * a**2 * (u1**2 + u1 * v1 + v1**2) + (c**2) * (w1**2)
    norm_2 = 3 * a**2 * (u2**2 + u2 * v2 + v2**2) + (c**2) * (w1**2)

    theta = np.arccos(dot_product / (np.sqrt(norm_1) * np.sqrt(norm_2)))

    if cos_res:
        return dot_product / (np.sqrt(norm_1) * np.sqrt(norm_2))
    else:
        return theta / np.pi * 180


def lowest_sf_ori_generator(planes, directions):

    dir_cal = np.zeros(4)
    logic = False
    sfs = []
    while not logic:
        dir_cal[0] = ra.randint(low=1, high=10)
        dir_cal[1] = ra.randint(low=1, high=10)
        dir_cal[2] = -(dir_cal[0] + dir_cal[1])
        dir_cal[3] = ra.randint(low=1, high=10)

        for plan, direction in zip(planes, directions):
            theta_1 = angle_between_hex_in_hex(dir_cal, planes, cos_res=True)
            theta_2 = angle_between_hex_in_hex(dir_cal, directions, cos_res=True)
            sf = theta_1 * theta_2
            sfs.append(sf)

        if (0.5 * len(planes) - np.sum(np.abs(sfs)) >= 3.2) and (
            np.any(np.abs(sfs)[:, -1] <= 0.02)
        ):
            logic = True
        else:
            sfs = []

    return dir_cal, sfs


if __name__ == "__main__":

    planes = np.array(
        [
            [0, 0, 0, 1],
            [1, 0, -1, 0],
            [1, 1, -2, 2],
            [1, 0, -1, 2],
            [1, 0, -1, -2],
            [1, 1, -2, 2],
            [1, 1, -2, -4],
        ]
    )

    directions = np.array(
        [
            [1, 1, -2, 0],
            [1, 1, -2, 0],
            [1, 1, -2, -3],
            [-1, 0, 1, 1],
            [1, 0, -1, 1],
            [1, 1, -2, -3],
            [2, 2, -4, 3],
        ]
    )

    dir_cal, sfs = lowest_sf_ori_generator(planes, directions)
    print(dir_cal, sfs, np.abs(sfs))

    cubic_cal = hex_to_cubix(dir_cal[0], dir_cal[1], dir_cal[2], dir_cal[3])
    print(cubic_cal)
