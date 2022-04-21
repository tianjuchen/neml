#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    A = input("crystal axis index one  ")
    B = input("crystal axis index one  ")
    C = input("crystal axis index one  ")
    A, B, C = float(A), float(B), float(C)
    crystal = np.array(
        [
            A / np.sqrt(A**2.0 + B**2.0 + C**2.0),
            B / np.sqrt(A**2.0 + B**2.0 + C**2.0),
            C / np.sqrt(A**2.0 + B**2.0 + C**2.0),
        ]
    )
    a = input("sample axis index one  ")
    b = input("sample axis index two  ")
    c = input("sample axis index three  ")
    a, b, c = float(a), float(b), float(c)
    sample = np.array(
        [
            a / np.sqrt(a**2 + b**2 + c**2),
            b / np.sqrt(a**2 + b**2 + c**2),
            c / np.sqrt(a**2 + b**2 + c**2),
        ]
    )

    # sample to crystal
    r = R.align_vectors(sample.reshape(1, -1), crystal.reshape(1, -1))

    M_tuple = r[0].as_matrix()
    M = np.asarray(M_tuple)
    # print(crystal.reshape(1, -1).T.shape)
    # M = np.matmul(crystal.reshape(1, -1).T, sample.reshape(1, -1))
    print(M)
    theta = np.arccos(M[2, 2])
    if abs(M[2, 2]) != 1.0:
        sth = np.sin(theta)
        psi = np.arctan2(M[1, 2] / sth, M[0, 2] / sth)
        phi = np.arctan2(M[2, 1] / sth, M[2, 0] / sth)
    else:
        psi = 0
        phi = np.arctan2(-M[0, 1], -M[0, 0])

    psi = psi / np.pi * 180
    theta = theta / np.pi * 180
    phi = phi / np.pi * 180
    print("psi = ", psi)
    print("theta = ", theta)
    print("phi = ", phi)
