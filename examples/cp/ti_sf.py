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
    cf = np.amin([h, k, i, l])
    [h, k, i, l] = [h, k, i, l] / cf
    return int(h), int(k), int(i), int(l)


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
    # clearing of fractions to lowest integers
    
    
    
    
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


