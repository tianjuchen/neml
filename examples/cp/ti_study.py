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


class hcp_model:
    def __init__(
        self, Q, N, threads, T, path, t_dir, c_dir, prefix, erate, emax, oripath
    ):
        self.Q = Q
        self.N = N
        self.threads = threads
        self.T = T
        self.path = path
        self.erate = erate
        self.emax = emax
        self.tension = t_dir
        self.compression = c_dir
        self.prefix = prefix
        self.oripath = oripath

    def hcp_singlecrystal(
        self, verbose=False, update_rotation=True, PTR=True, return_isv=False
    ):

        # temperature levels
        Ts = np.array([298.0, 423.0, 523.0, 623.0, 773.0, 873.0, 973.0, 1073.0, 1173.0])
        # unit transformer
        ut = 1.0e-3

        # Model
        a = 2.9511 * 0.1 * ut  # nm
        c = 4.68433 * 0.1 * ut  # nm

        # Elastic constants in MPa
        C11 = interpolate.PiecewiseLinearInterpolate(
            list(Ts),
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
            ],
        )
        C33 = interpolate.PiecewiseLinearInterpolate(
            list(Ts),
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
            ],
        )
        C44 = interpolate.PiecewiseLinearInterpolate(
            list(Ts),
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
            ],
        )
        C12 = interpolate.PiecewiseLinearInterpolate(
            list(Ts),
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
            ],
        )
        C13 = interpolate.PiecewiseLinearInterpolate(
            list(Ts),
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
            ],
        )

        # Constant part of the strength for slip and twin
        taus_1 = interpolate.PiecewiseLinearInterpolate(
            list(Ts), [200.0, 145.0, 100.0, 70.0, 53.0, 38.0, 18.0, 15.0, 9.0]
        )
        taus_2 = interpolate.PiecewiseLinearInterpolate(
            list(Ts), [120.0, 70.5, 60.0, 60.0, 43.0, 38.0, 18.0, 15.0, 9.0]
        )
        taus_3 = interpolate.PiecewiseLinearInterpolate(
            list(Ts), [230.0, 185.0, 145.0, 110.0, 87.0, 61.0, 35.0, 20.0, 11.0]
        )
        taut_1 = interpolate.PiecewiseLinearInterpolate(
            list(Ts), [200.0, 170.0, 160.0, 150.0, 120.0, 110.0, 100.0, 100.0, 100.0]
        )
        taut_2 = interpolate.PiecewiseLinearInterpolate(
            list(Ts), [250.0, 240.0, 230.0, 210.0, 200.0, 190.0, 180.0, 180.0, 180.0]
        )
        # tau0 = np.array([170.0]*3+[90.5]*3+[210]*6+[180.0]*6+[250.0]*6)
        tau0 = np.array(
            [taus_1] * 3 + [taus_2] * 3 + [taus_3] * 6 + [taut_1] * 6 + [taut_2] * 6
        )

        # Reference slip rate and rate sensitivity exponent
        g0 = 1.0
        # if strain rate is 1e-2
        # n = 12.0
        # if strain rate is 1e-3
        # n = 7.5
        applied_rate = 8.33e-5
        rates_control = np.array([1e-3, 1e-2])
        sense_control = np.array([7.5, 12.0])
        n = extrapolate(rates_control, sense_control).value(applied_rate)

        # Twin threshold
        twin_threshold = 0.75

        # Sets up the lattice crystallography
        lattice = crystallography.HCPLattice(a, c)
        # Basal <a>
        lattice.add_slip_system([1, 1, -2, 0], [0, 0, 0, 1])
        # Prismatic <a>
        lattice.add_slip_system([1, 1, -2, 0], [1, 0, -1, 0])
        # Pyramidal <c+a>
        lattice.add_slip_system([1, 1, -2, -3], [1, 1, -2, 2])
        # Tension twinning
        lattice.add_twin_system(
            [-1, 0, 1, 1], [1, 0, -1, 2], [1, 0, -1, 1], [1, 0, -1, -2]
        )
        # Compression twinning
        lattice.add_twin_system(
            [1, 1, -2, -3], [1, 1, -2, 2], [2, 2, -4, 3], [1, 1, -2, -4]
        )

        # Sets up the interaction matrix
        num_basal, num_prism, num_pyram = 3, 3, 6
        num_ttwin, num_ctwin = 6, 6
        ## basal plane
        C1_single = np.array([50.0] * num_ttwin + [50.0] * num_ctwin)
        C1 = np.stack([C1_single for _ in range(num_basal)])
        ## prismatic plane
        C2_single = np.array([100.0] * num_ttwin + [1000.0] * num_ctwin)
        C2 = np.stack([C2_single for _ in range(num_prism)])
        ## pyramidal plane
        C3_single = np.array([250.0] * num_ttwin + [170.0] * num_ctwin)
        C3 = np.stack([C3_single for _ in range(num_pyram)])
        ## stack up each slip system
        C_np = np.vstack((C1, C2, C3)).T

        C_st = matrix.SquareMatrix(12, type="dense", data=C_np.flatten())

        # source from Fracture of Titanium Alloys at High Strain Rates and under Stress Triaxiality
        # calculate temperature depdendent shear modulus of slip systems  u = 39.61-0.03223*T
        mu_slip = interpolate.PiecewiseLinearInterpolate(
            list(Ts),
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
            ],
        )
        # calculate temperature depdendent shear modulus of twin systems  u = 34.605-0.03223*T
        mu_twin = interpolate.PiecewiseLinearInterpolate(
            list(Ts),
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
            ],
        )

        mu = np.array([mu_slip] * 12 + [mu_twin] * 12)

        k1 = np.array([1.0] * 3 + [0.25] * 3 + [5.0] * 6) / ut

        k2_1 = interpolate.PiecewiseLinearInterpolate(
            list(Ts), [250.0, 280.0, 330.0, 450.0, 500.0, 600.0, 1020.0, 1200.0, 1400.0]
        )
        k2_2 = interpolate.PiecewiseLinearInterpolate(
            list(Ts), [250.0, 280.0, 330.0, 450.0, 500.0, 600.0, 1020.0, 1200.0, 1400.0]
        )
        k2_3 = interpolate.PiecewiseLinearInterpolate(
            list(Ts), [250.0, 280.0, 330.0, 450.0, 500.0, 600.0, 1020.0, 1200.0, 1400.0]
        )

        k2 = np.array([k2_1] * 3 + [k2_2] * 3 + [k2_3] * 6)

        # Sets up the linear elastic tensor
        emodel = elasticity.TransverseIsotropicLinearElasticModel(
            C11, C33, C12, C13, C44, "components"
        )

        # Sets up the slip system strength model (this is what you'll change)
        strength = slipharden.LANLTiModel(tau0, C_st, mu, k1, k2, X_s=0.9, inivalue=1.0)
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
                initial_rotation=self.Q,
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
                initial_rotation=self.Q,
                miter=100,
                max_divide=10,
            )

        if return_isv:
            return single_model, slipmodel, lattice, strength
        else:
            return smodel

    def Lattice(self):
        single_model, slipmodel, lattice, strength = self.hcp_singlecrystal(
            return_isv=True
        )
        return lattice

    def slip_model(self):
        single_model, slipmodel, lattice, strength = self.hcp_singlecrystal(
            return_isv=True
        )
        return slipmodel

    def hardening_model(self):
        single_model, slipmodel, lattice, strength = self.hcp_singlecrystal(
            return_isv=True
        )
        return strength

    def sxtal_model(self):
        single_model, slipmodel, lattice, strength = self.hcp_singlecrystal(
            return_isv=True
        )
        return single_model

    def orientations(self, random=False, initial=True):
        if random:
            orientations = rotations.random_orientations(self.N)
        elif initial:
            fnames = glob.glob(self.oripath + "*.csv")
            orientation_angles = np.zeros((500, 3))
            for f in fnames:
                file_name = os.path.basename(f).split(".csv")[0]
                if file_name == "history":
                    df = pd.read_csv(f)
                    ori_1 = df["ori_1"]
                    ori_2 = df["ori_2"]
                    ori_3 = df["ori_3"]

                    for i, (euler_1, euler_2, euler_3) in enumerate(
                        zip(ori_1, ori_2, ori_3)
                    ):
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
        else:
            orientations = [self.Q for _ in range(self.N)]

        return orientations

    def taylor_model(self):
        single_model = self.sxtal_model()
        orientations = self.orientations()
        tmodel = polycrystal.TaylorModel(
            single_model, orientations, nthreads=self.threads
        )
        return tmodel

    def plot_initial_pf(self, display=True, savefile=False):
        orientations = self.orientations()
        polefigures.pole_figure_discrete(orientations, [0, 0, 0, 1], self.Lattice())
        plt.title("Initial, <0001>")
        if savefile:
            plt.savefig(
                self.path + "initialpf-%i-C.pdf" % int(self.T - 273.15), dpi=300
            )
        if display:
            plt.show()
        return plt.close()

    def driver(self, full_res=True):
        res = drivers.uniaxial_test(
            self.taylor_model(),
            erate=self.erate,
            emax=self.emax,
            sdir=self.tension,
            T=self.T,
            verbose=True,
            full_results=full_res,
        )
        return res

    def usym(self, v):
        """
        Take a Mandel symmetric vector to the full matrix.
        """
        return np.array(
            [
                [v[0], v[5] / np.sqrt(2), v[4] / np.sqrt(2)],
                [v[5] / np.sqrt(2), v[1], v[3] / np.sqrt(2)],
                [v[4] / np.sqrt(2), v[3] / np.sqrt(2), v[2]],
            ]
        )

    def hist_evolution(self, res):
        smodel = self.sxtal_model()
        lattice = self.Lattice()
        nt = len(res["history"])
        # initialize empty tensore for storing hist
        direct_from_model = np.zeros((nt, lattice.ntotal))  # slip strain
        integrated_ourselves = np.zeros((nt, lattice.ntotal))  # slip strain
        taus_from_model = np.zeros((nt, lattice.ntotal))  # crss
        slip_rates = np.zeros((nt, lattice.ntotal))  # slip rates
        resolved_shear_stresses = np.zeros((nt, lattice.ntotal))  # rss

        for i in range(1, len(res["history"])):
            hist = history.History()
            smodel.populate_history(hist)
            hist.set_data(res["history"][i])
            stress = tensors.Symmetric(self.usym(res["stress"][i]))
            T = res["temperature"][i]
            Q = hist.get_orientation("rotation")

            fixed = history.History()

            dt = res["time"][i] - res["time"][i - 1]

            for g in range(lattice.ngroup):
                for j in range(lattice.nslip(g)):
                    slip_rate = self.slip_model().slip(
                        g, j, stress, Q, hist, lattice, T, fixed
                    )
                    slip_rates[i, lattice.flat(g, j)] = slip_rate
                    resolved_shear_stresses[i, lattice.flat(g, j)] = lattice.shear(
                        g, j, Q, stress
                    )
                    integrated_ourselves[i, lattice.flat(g, j)] = (
                        integrated_ourselves[i - 1, lattice.flat(g, j)]
                        + np.abs(slip_rate) * dt
                    )
                    if lattice.flat(g, j) < 12:
                        direct_from_model[i, lattice.flat(g, j)] = hist.get_scalar(
                            "slip" + str(lattice.flat(g, j) + 24)
                        )
                    else:
                        direct_from_model[i, lattice.flat(g, j)] = hist.get_scalar(
                            "twin" + str(lattice.flat(g, j))
                        )
                    taus_from_model[
                        i, lattice.flat(g, j)
                    ] = self.hardening_model().hist_to_tau(
                        g, j, hist, lattice, T, fixed
                    )
        return (
            direct_from_model,
            integrated_ourselves,
            taus_from_model,
            slip_rates,
            resolved_shear_stresses,
        )

    def history_plot(
        self, history_var, xname, yname, fname, display=True, savefile=False
    ):
        x = np.arange(len(history_var)) + 1
        fig, ax = plt.subplots()
        ax.bar(x, history_var, color="k")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("{}".format(xname), fontsize=14)
        plt.ylabel("{}".format(yname), fontsize=14)
        # plt.legend(labels=["{}".format(yname)], prop={"size": 14}, frameon=False)
        plt.grid(False)
        plt.tight_layout()
        if savefile:
            plt.savefig(self.path + "{}-{}.pdf".format(fname, self.T), dpi=300)
        if display:
            plt.show()
        return plt.close()

    def rss_history(self, res, display=True, savefile=False):
        (
            direct_from_model,
            integrated_ourselves,
            taus_from_model,
            slip_rates,
            resolved_shear_stresses,
        ) = self.hist_evolution(res)
        lattice = self.Lattice()

        # plot rss evolution
        print("")
        print("plotting rss evolution")
        print("")
        for g in range(lattice.ngroup):
            for j in range(lattice.nslip(g)):
                i = lattice.flat(g, j)
                plt.plot(
                    resolved_shear_stresses[:, i],
                    ls="-",
                    label="rss evolution of slip {} in group {}".format(i, g),
                )
            plt.legend()
            if savefile:
                plt.savefig(self.path + "rss-group-{}.pdf".format(g), dpi=300)
            if display:
                plt.show()
            plt.close()
        # plot crss evolution
        print("")
        print("plotting crss evolution")
        print("")
        for g in range(lattice.ngroup):
            for j in range(lattice.nslip(g)):
                i = lattice.flat(g, j)
                plt.plot(
                    taus_from_model[:, i],
                    ls="-",
                    label="crss evolution of slip {} in group {}".format(i, g),
                )
            plt.legend()
            if savefile:
                plt.savefig(self.path + "crss-group-{}.pdf".format(g), dpi=300)
            if display:
                plt.show()
            plt.close()

        # plot crss evolution
        print("")
        print("plotting slip rates evolution")
        print("")
        for g in range(lattice.ngroup):
            for j in range(lattice.nslip(g)):
                i = lattice.flat(g, j)
                plt.plot(
                    slip_rates[:, i],
                    ls="-",
                    label="slip rate evolution of slip {} in group {}".format(i, g),
                )
            plt.legend()
            if savefile:
                plt.savefig(self.path + "slip-rate-group-{}.pdf".format(g), dpi=300)
            if display:
                plt.show()
            plt.close()

        # plot integrated strain evolution
        print("")
        print("plotting integrated strain evolution")
        print("")
        for g in range(lattice.ngroup):
            for j in range(lattice.nslip(g)):
                i = lattice.flat(g, j)
                plt.plot(
                    direct_from_model[:, i],
                    ls="-",
                    lw=2,
                    label="direct-slip-{}".format(i),
                )
                plt.plot(
                    integrated_ourselves[:, i],
                    ls="--",
                    lw=2,
                    label="integrate-slip-{}".format(i),
                )
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel("Step", fontsize=14)
            plt.ylabel("Strain", fontsize=14)
            # plt.legend(labels=["{}".format(yname)], prop={"size": 14}, frameon=False)
            plt.grid(False)
            plt.tight_layout()
            plt.legend()
            if savefile:
                plt.savefig(
                    self.path + "integrate-slip-strain-group-{}.pdf".format(g), dpi=300
                )
            if display:
                plt.show()
            plt.close()

        # plot accumulated slip strain evolution
        print("")
        print("plotting accumulated slip strain evolution")
        print("")
        _ = self.history_plot(
            integrated_ourselves[-1, :12],
            "Slip System",
            "Accumulated Slip Strain",
            "slip-strain",
            display=display,
            savefile=savefile,
        )

        # plot accumulated twin strain evolution
        print("")
        print("plotting accumulated twin strain evolution")
        print("")
        _ = self.history_plot(
            integrated_ourselves[-1, 12:],
            "Twin System",
            "Accumulated Twin Strain",
            "twin-strain",
            display=display,
            savefile=savefile,
        )

        # plot hardening evolution
        print("")
        print("plotting accumulated hardening evolution")
        print("")
        _ = self.history_plot(
            taus_from_model[-1, :],
            "slip/twin System",
            "hardening",
            "hardening-evolution",
            display=display,
            savefile=savefile,
        )

    def accumulate_history(self, res, start_index, end_index, accum=True):
        store_history = np.array(res["history"])
        save_index = np.arange(start_index, end_index)
        num_niternal = store_history.shape[-1]

        if accum:
            accum_hist_var = np.zeros(12)
            for j in save_index:
                for i in range(num_niternal):
                    if i % num_niternal == j:
                        accum_hist_var[j - start_index] += store_history[-1, i]

        else:
            accum_hist_var = np.zeros((store_history.shape[0], 12))
            for j in save_index:
                for i in range(num_niternal):
                    if i % num_niternal == j:
                        accum_hist_var[:, j - start_index] += store_history[:, i]

        return accum_hist_var

    def accumulated_density(self, res, accum=True):
        if accum:
            return self.accumulate_history(res, 8, 20)
        else:
            return self.accumulate_history(res, 8, 20, accum=accum)

    def accumulated_twin(self, res, accum=True):
        if accum:
            return self.accumulate_history(res, 20, 32)
        else:
            return self.accumulate_history(res, 20, 32, accum=accum)

    def accumulated_slip(self, res, accum=True):
        if accum:
            return self.accumulate_history(res, 32, 44)
        else:
            return self.accumulate_history(res, 32, 44, accum=accum)

    def save_accum_isv_dataframe(self, res, display=True, savefile=False):
        accu_density = self.accumulated_density(res)
        accu_twin = self.accumulated_twin(res)
        accu_slip = self.accumulated_slip(res)

        # plot distribution of dislocation density
        _ = self.history_plot(
            accu_density**2 * 1.0e12,
            "Slip System",
            "Accumulated Dislocation Density",
            "{}-dislocation-density".format(self.prefix),
            display=display,
            savefile=savefile,
        )

        # plot distribution of accumulated twin strain
        _ = self.history_plot(
            accu_twin,
            "Twin System",
            "Accumulated Twin Strain",
            "{}-twin-strain".format(self.prefix),
            display=display,
            savefile=savefile,
        )

        # plot distribution of accumulated slip strain
        _ = self.history_plot(
            accu_slip,
            "Slip System",
            "Accumulated Slip Strain",
            "{}-slip-strain".format(self.prefix),
            display=display,
            savefile=savefile,
        )

        data = pd.DataFrame(
            {
                "dis_density": accu_density**2 * 1.0e12,
                "accu_twin": accu_twin,
                "accu_slip": accu_slip,
            }
        )

        data.to_csv(self.path + "{}_res_{}.csv".format(self.prefix, int(self.T)))
        return data

    def save_evolve_isv_dataframe(self, res):
        evolve_density = self.accumulated_density(res, accum=False)
        evolve_twin = self.accumulated_twin(res, accum=False)
        evolve_slip = self.accumulated_slip(res, accum=False)
        data = pd.DataFrame(
            {
                "evolve_density_1": evolve_density[:, 0] ** 2 * 1.0e12,
                "evolve_density_2": evolve_density[:, 1] ** 2 * 1.0e12,
                "evolve_density_3": evolve_density[:, 2] ** 2 * 1.0e12,
                "evolve_density_4": evolve_density[:, 3] ** 2 * 1.0e12,
                "evolve_density_5": evolve_density[:, 4] ** 2 * 1.0e12,
                "evolve_density_6": evolve_density[:, 5] ** 2 * 1.0e12,
                "evolve_density_7": evolve_density[:, 6] ** 2 * 1.0e12,
                "evolve_density_8": evolve_density[:, 7] ** 2 * 1.0e12,
                "evolve_density_9": evolve_density[:, 8] ** 2 * 1.0e12,
                "evolve_density_10": evolve_density[:, 9] ** 2 * 1.0e12,
                "evolve_density_11": evolve_density[:, 10] ** 2 * 1.0e12,
                "evolve_density_12": evolve_density[:, 11] ** 2 * 1.0e12,
                "evolve_twin_1": evolve_twin[:, 0],
                "evolve_twin_2": evolve_twin[:, 1],
                "evolve_twin_3": evolve_twin[:, 2],
                "evolve_twin_4": evolve_twin[:, 3],
                "evolve_twin_5": evolve_twin[:, 4],
                "evolve_twin_6": evolve_twin[:, 5],
                "evolve_twin_7": evolve_twin[:, 6],
                "evolve_twin_8": evolve_twin[:, 7],
                "evolve_twin_9": evolve_twin[:, 8],
                "evolve_twin_10": evolve_twin[:, 9],
                "evolve_twin_11": evolve_twin[:, 10],
                "evolve_twin_12": evolve_twin[:, 11],
                "evolve_slip_1": evolve_slip[:, 0],
                "evolve_slip_2": evolve_slip[:, 1],
                "evolve_slip_3": evolve_slip[:, 2],
                "evolve_slip_4": evolve_slip[:, 3],
                "evolve_slip_5": evolve_slip[:, 4],
                "evolve_slip_6": evolve_slip[:, 5],
                "evolve_slip_7": evolve_slip[:, 6],
                "evolve_slip_8": evolve_slip[:, 7],
                "evolve_slip_9": evolve_slip[:, 8],
                "evolve_slip_10": evolve_slip[:, 9],
                "evolve_slip_11": evolve_slip[:, 10],
                "evolve_slip_12": evolve_slip[:, 11],
            }
        )

        data.to_csv(self.path + "{}_hist_{}.csv".format(self.prefix, int(self.T)))

        return data

    def deformed_texture(self, res, display=True, savefile=False):
        store_history = np.array(res["history"])
        pf = self.taylor_model().orientations(store_history[-1])
        polefigures.pole_figure_discrete(
            pf,
            [0, 0, 0, 1],
            lattice=self.Lattice(),
            x=tensors.Vector([1.0, 0, 0]),
            y=tensors.Vector([0, 1.0, 0]),
            axis_labels=["X", "Y"],
        )
        polefigures.pole_figure_discrete(
            pf.flip(),
            [0, 0, 0, 1],
            lattice=self.Lattice(),
            x=tensors.Vector([1.0, 0, 0]),
            y=tensors.Vector([0, 1.0, 0]),
            axis_labels=["X", "Y"],
        )
        plt.title("Final, <0001>")
        if savefile:
            plt.savefig(
                self.path + "deformpf-%i-C-1.pdf" % int(self.T - 273.15), dpi=300
            )
        if display:
            plt.show()
        plt.close()

        polefigures.pole_figure_discrete(pf, [1, 0, -1, 0], lattice=self.Lattice())
        plt.title("Final, <1010>")
        if savefile:
            plt.savefig(
                self.path + "deformpf-%i-C-2.pdf" % int(self.T - 273.15), dpi=300
            )
        if display:
            plt.show()
        plt.close()

        polefigures.pole_figure_discrete(pf, [1, 1, -2, 0], lattice=self.Lattice())
        plt.title("Final, <1120>")
        if savefile:
            plt.savefig(
                self.path + "deformpf-%i-C-3.pdf" % int(self.T - 273.15), dpi=300
            )
        if display:
            plt.show()
        plt.close()

        polefigures.pole_figure_discrete(pf, [1, 0, -1, 1], lattice=self.Lattice())
        plt.title("Final, <1011>")
        if savefile:
            plt.savefig(
                self.path + "deformpf-%i-C-4.pdf" % int(self.T - 273.15), dpi=300
            )
        if display:
            plt.show()
        plt.close()

        return pf

    def save_texture(self, res):
        store_history = np.array(res["history"])
        updated_quats = self.taylor_model().orientations(store_history)
        updated_euler = [
            q.to_euler(angle_type="degrees", convention="kocks") for q in updated_quats
        ]
        data = pd.DataFrame(
            {
                "ori_1": np.array(updated_euler)[:, 0],
                "ori_2": np.array(updated_euler)[:, 1],
                "ori_3": np.array(updated_euler)[:, 2],
            }
        )

        data.to_csv(self.path + "{}_texture_{}.csv".format(self.prefix, int(self.T)))
        return data


if __name__ == "__main__":

    path = "/mnt/c/Users/ladmin/Desktop/argonne/neml/neml/examples/cp/try/"
    texture_path = "/mnt/c/Users/ladmin/Desktop/argonne/neml/neml/examples/cp/"
    Q = rotations.CrystalOrientation(
        0.0, 0.0, 0.0, angle_type="degrees", convention="kocks"
    )
    c_dir = np.array([-1, 0, 0, 0, 0, 0])
    t_dir = np.array([0, 1.0, -1.0, 0, 0, 0])
    dirs = [t_dir, c_dir]
    prefixs = ["tension", "compression"]
    N, nthreads = 500, 1
    T = 298.0
    erate, emax = 8.33e-5, np.log(1 + 0.5)
    hcp_model = hcp_model(
        Q, N, nthreads, T, path, t_dir, c_dir, prefixs[0], erate, emax, texture_path
    )

    res = hcp_model.driver()
    hcp_model.plot_initial_pf(display=False, savefile=True)
    hcp_model.deformed_texture(res, display=False, savefile=True)
    hcp_model.rss_history(res, display=False, savefile=True)
    hcp_model.save_accum_isv_dataframe(res, display=False, savefile=True)
    hcp_model.save_evolve_isv_dataframe(res)
    hcp_model.save_texture(res)
