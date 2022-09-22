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


import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy.random as ra
import pandas as pd
import xarray as xr
import tqdm
import warnings

warnings.filterwarnings("ignore")


def denormalize(u, v, w):

    u_dividor = (u).as_integer_ratio()[1]
    v_dividor = (v).as_integer_ratio()[1]
    w_dividor = (w).as_integer_ratio()[1]
    
    u = u * u_dividor * v_dividor * w_dividor
    v = v * u_dividor * v_dividor * w_dividor
    w = w * u_dividor * v_dividor * w_dividor
    
    temp_vector = [u, v, w]
    X = [int(i) for i in temp_vector if i != 0]
    # gcd = np.gcd.reduce([int(u), int(v), int(w)])
    gcd = np.gcd.reduce(X)

    dn_vector = np.array([u, v, w]) / gcd
    
    return dn_vector

def normalize(u, v, w):
    vector = np.array([u, v, w])
    return vector/np.linalg.norm(vector)

if __name__ == "__main__":

    A = input("vector index one  ")
    B = input("vector index two  ")
    C = input("vector index three  ")
    A, B, C = float(A), float(B), float(C)
   
    print(denormalize(A, B, C))
    # print(normalize(A, B, C))
    