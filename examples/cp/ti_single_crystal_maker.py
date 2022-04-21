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


class extrapolate:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def value(self, T):
        func = np.poly1d(np.polyfit(self.xs, self.ys, deg=1))
        return func(T).tolist()


def Ti_singxtal(
    crystal_dir, verbose=True, PTR=True, return_hardening=False, update_rotation=True
):
    # temperature levels
    Ts_old = np.array([298.0, 423.0, 523.0, 623.0, 773.0, 873.0, 973.0, 1073.0, 1173.0])
    # Ts = np.array([298.0, 423.0, 523.0, 623.0, 773.0, 873.0, 973.0, 1073.0, 1173.0, 1255.372, 1366.483])
    Ts = np.array([1073.0, 1173.0, 1255.372])
    # unit transformer
    ut = 1.0e-3
    # Model
    a = 2.9511 * 0.1 * ut  # nm
    c = 4.68433 * 0.1 * ut  # nm

    # Elastic constants in MPa
    C11_old = np.array(
        [
            162400.0,
            155100.0,
            149500.0,
            144200.0,
            136800.0,
            132200.0,
            127600.0,
            123100.0,
            119600.0,
        ]
    )
    C33_old = np.array(
        [
            180700.0,
            175300.0,
            171500.0,
            167800.0,
            162700.0,
            159300.0,
            156000.0,
            152900.0,
            150400.0,
        ]
    )
    C44_old = np.array(
        [
            46700.0,
            44400.0,
            42400.0,
            40300.0,
            37000.0,
            34800.0,
            32600.0,
            30700.0,
            29100.0,
        ]
    )
    C12_old = np.array(
        [
            92000.0,
            94300.0,
            96100.0,
            97300.0,
            98500.0,
            99100.0,
            99300.0,
            99600.0,
            99600.0,
        ]
    )
    C13_old = np.array(
        [
            69000.0,
            69500.0,
            69200.0,
            69100.0,
            68800.0,
            68800.0,
            68800.0,
            68800.0,
            68800.0,
        ]
    )

    C11 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], C11_old[-2:]).value(Ts)
    )
    C33 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], C33_old[-2:]).value(Ts)
    )
    C44 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], C44_old[-2:]).value(Ts)
    )
    C12 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], C12_old[-2:]).value(Ts)
    )
    C13 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], C13_old[-2:]).value(Ts)
    )

    # Constant part of the strength for slip and twin
    taus_1_old = np.array([200.0, 145.0, 100.0, 70.0, 53.0, 38.0, 18.0, 15.0, 9.0])
    taus_2_old = np.array([120.0, 70.5, 60.0, 60.0, 43.0, 38.0, 18.0, 15.0, 9.0])
    taus_3_old = np.array([230.0, 185.0, 145.0, 110.0, 87.0, 61.0, 35.0, 20.0, 11.0])
    taut_1_old = np.array(
        [200.0, 170.0, 160.0, 150.0, 120.0, 110.0, 100.0, 100.0, 100.0]
    )
    taut_2_old = np.array(
        [250.0, 240.0, 230.0, 210.0, 200.0, 190.0, 180.0, 180.0, 180.0]
    )

    taus_1 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], taus_1_old[-2:]).value(Ts)
    )
    taus_2 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], taus_2_old[-2:]).value(Ts)
    )
    taus_3 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], taus_3_old[-2:]).value(Ts)
    )
    taut_1 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], taut_1_old[-2:]).value(Ts)
    )
    taut_2 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], taut_2_old[-2:]).value(Ts)
    )
    # tau0 = np.array([170.0]*3+[90.5]*3+[210]*6+[180.0]*6+[250.0]*6)
    tau0 = np.array(
        [taus_1] * 3 + [taus_2] * 3 + [taus_3] * 6 + [taut_1] * 6 + [taut_2] * 6
    )

    # Reference slip rate and rate sensitivity exponent
    g0 = 1.0
    # if strain rate is 1e-2
    n = 12.0
    # if strain rate is 1e-3
    # n = 7.5

    # Twin threshold
    twin_threshold = 0.75

    # Sets up the lattice crystallography
    lattice = crystallography.HCPLattice(a, c)
    # Basal <a>
    lattice.add_slip_system([1, 1, -2, 0], [0, 0, 0, 1]) #direction, planes
    # Prismatic <a>
    lattice.add_slip_system([1, 1, -2, 0], [1, 0, -1, 0])
    # Pyramidal <c+a>
    lattice.add_slip_system([1, 1, -2, -3], [1, 1, -2, 2])
    # Tension twinning
    lattice.add_twin_system([-1, 0, 1, 1], [1, 0, -1, 2], [1, 0, -1, 1], [1, 0, -1, -2])
    # Compression twinning
    lattice.add_twin_system(
        [1, 1, -2, -3], [1, 1, -2, 2], [2, 2, -4, 3], [1, 1, -2, -4]
    )

    # Sets up the interaction matrix
    num_basal, num_prism, num_pyram = 3, 3, 6
    num_ttwin, num_ctwin = 6, 6
    ## basal plane
    C1_single = np.array([50.0] * num_ttwin + [50.0] * num_ttwin)
    C1 = np.stack([C1_single for _ in range(num_basal)])
    ## prismatic plane
    C2_single = np.array([100.0] * num_ttwin + [1000.0] * num_ttwin)
    C2 = np.stack([C2_single for _ in range(num_prism)])
    ## pyramidal plane
    C3_single = np.array([250.0] * num_ttwin + [170.0] * num_ttwin)
    C3 = np.stack([C3_single for _ in range(num_pyram)])
    ## stack up each slip system
    C_np = np.vstack((C1, C2, C3)).T

    C_st = matrix.SquareMatrix(12, type="dense", data=C_np.flatten())

    # source from Fracture of Titanium Alloys at High Strain Rates and under Stress Triaxiality
    # calculate temperature depdendent shear modulus of slip systems  u = 39.61-0.03223*T
    mu_slip_old = np.array(
        [
            35200.0,
            30400.0,
            26700.0,
            23400.0,
            19100.0,
            16600.0,
            14200.0,
            11800.0,
            10000.0,
        ]
    )
    mu_twin_old = np.array(
        [
            25000.46,
            21591.31,
            18963.42,
            16619.62,
            13565.60,
            11789.99,
            10085.41,
            8381.28,
            7102.78,
        ]
    )

    mu_slip = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], mu_slip_old[-2:]).value(Ts)
    )
    # calculate temperature depdendent shear modulus of twin systems  u = 34.605-0.03223*T
    mu_twin = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], mu_twin_old[-2:]).value(Ts)
    )

    mu = np.array([mu_slip] * 12 + [mu_twin] * 12)

    k1 = np.array([1.0] * 3 + [0.25] * 3 + [5.0] * 6) / (ut) * 1.0e-6

    k2_1_old = (
        np.array([250.0, 280.0, 330.0, 450.0, 500.0, 600.0, 1020.0, 1200.0, 1400.0])
        * 1.0e-6
    )
    k2_2_old = (
        np.array([250.0, 280.0, 330.0, 450.0, 500.0, 600.0, 1020.0, 1200.0, 1400.0])
        * 1.0e-6
    )
    k2_3_old = (
        np.array([250.0, 280.0, 330.0, 450.0, 500.0, 600.0, 1020.0, 1200.0, 1400.0])
        * 1.0e-6
    )

    k2_1 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], k2_1_old[-2:]).value(Ts)
    )
    k2_2 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], k2_2_old[-2:]).value(Ts)
    )
    k2_3 = interpolate.PiecewiseLinearInterpolate(
        list(Ts), extrapolate(Ts_old[-2:], k2_3_old[-2:]).value(Ts)
    )

    k2 = np.array([k2_1] * 3 + [k2_2] * 3 + [k2_3] * 6)

    # Sets up the linear elastic tensor
    emodel = elasticity.TransverseIsotropicLinearElasticModel(
        C11, C33, C12, C13, C44, "components"
    )

    # Sets up the slip system strength model (this is what you'll change)
    # strength = slipharden.FixedStrengthHardening(tau0)
    strength = slipharden.LANLTiModel(tau0, C_st, mu, k1, k2, X_s=0.9, inivalue=1.0)
    # strength = slipharden.SimpleLinearHardening(M, tau0)
    # Sets up the slip rule
    slipmodel = sliprules.PowerLawSlipRule(strength, g0, n)
    # Sets up the model inelastic rate kinematics
    imodel = inelasticity.AsaroInelasticity(slipmodel)
    # Sets up the overall model kinematics
    kmodel = kinematics.StandardKinematicModel(emodel, imodel)

    # This is the object that causes twins to recrystallize
    twinner = postprocessors.PTRTwinReorientation(twin_threshold)

    # Sets up the single crystal model
    if PTR:
        single_model = singlecrystal.SingleCrystalModel(
            kmodel,
            lattice,
            update_rotation=update_rotation,
            postprocessors=[],
            verbose=False,
            linesearch=True,
            initial_rotation=rotations.CrystalOrientation(
                crystal_dir[0],
                crystal_dir[1],
                crystal_dir[2],
                angle_type="degrees",
                convention="kocks",
            ),
            miter=100,
            max_divide=10,
        )
    else:
        single_model = singlecrystal.SingleCrystalModel(
            kmodel,
            lattice,
            update_rotation=update_rotation,
            verbose=False,
            linesearch=True,
            initial_rotation=rotations.CrystalOrientation(
                crystal_dir[0],
                crystal_dir[1],
                crystal_dir[2],
                angle_type="degrees",
                convention="kocks",
            ),
            miter=100,
            max_divide=10,
        )

    if return_hardening:
        return single_model, strength
    else:
        return single_model


def interp(strain, stress, targets):
    """
    This is just to make sure all our values line up, in case the model
    adaptively integrated or something
    """
    return inter.interp1d(strain, stress)(targets)


if __name__ == "__main__":

    Ts = np.array([298.0, 423.0, 523.0, 623.0, 773.0, 873.0, 973.0, 1073.0, 1173.0])
    C11 = np.array(
        [
            162400.0,
            155100.0,
            149500.0,
            144200.0,
            136800.0,
            132200.0,
            127600.0,
            123100.0,
            119600.0,
        ]
    )
    Ts_new = np.array(
        [298.0, 423.0, 523.0, 623.0, 773.0, 873.0, 973.0, 1073.0, 1173.0, 1255.372]
    )
    C11_new = extrapolate(Ts, C11).value(Ts_new)
    print(Ts[-2:])
