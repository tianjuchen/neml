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
    addmaf,
)
from neml.math import rotations, tensors, nemlmath, matrix
from neml import drivers

from matplotlib import RcParams
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy.linalg as la
import numpy.random as ra
import pandas as pd
import tqdm
import warnings

warnings.filterwarnings("ignore")


latex_style_times = RcParams(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
    }
)


class extrapolate:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def value(self, T):
        func = np.poly1d(np.polyfit(self.xs, self.ys, deg=1))
        return func(T).tolist()


class am_model:
    def __init__(
        self,
        Q,
        N,
        threads,
        T,
        ldir,
        prefix,
        erate,
        emax,
        M,
        kw1,
        kw2,
        ki1,
        ki2,
        ftr,
        k0,
        Qv,
        initsigma,
        g0,
        n,
        E,
        nu,
        use_ptr=False,
    ):
        self.Q = Q
        self.N = N
        self.threads = threads
        self.T = T
        self.erate = erate
        self.emax = emax
        self.ldir = ldir
        self.prefix = prefix
        self.use_ptr = use_ptr
        self.M = M
        self.kw1 = kw1
        self.kw2 = kw2
        self.ki1 = ki1
        self.ki2 = ki2
        self.ftr = ftr
        self.initsigma = initsigma
        self.g0 = g0
        self.n = n
        self.E = E
        self.nu = nu
        self.k0 = k0
        self.Qv = Qv
        self.omega = 588930.52
        self.inibvalue = 7.34e2

    def lattice(self):
        lattice = crystallography.CubicLattice(1.0)
        lattice.add_slip_system([1, 1, 0], [1, 1, 1])
        return lattice

    def singlecrystal(self, verbose=False, update_rotation=True):
        strengthmodel = addmaf.AMModel(
            self.M,
            self.kw1,
            self.kw2,
            self.ki1,
            self.ki2,
            ftr = self.ftr,
            k0 = self.k0,
            Q = self.Qv,
            omega=self.omega,
            inibvalue=self.inibvalue,
            initsigma=self.initsigma,
        )
        slipmodel = sliprules.PowerLawSlipRule(strengthmodel, self.g0, self.n)
        imodel = inelasticity.AsaroInelasticity(slipmodel)
        emodel = elasticity.IsotropicLinearElasticModel(
            self.E, "youngs", self.nu, "poissons"
        )
        kmodel = kinematics.StandardKinematicModel(emodel, imodel)
        lattice = self.lattice()

        # Sets up the single crystal model
        single_model = singlecrystal.SingleCrystalModel(
            kmodel,
            lattice,
            update_rotation=update_rotation,
            verbose=verbose,
            linesearch=True,
            initial_rotation=self.Q,
            miter=100,
            max_divide=10,
        )
        return single_model, slipmodel, strengthmodel

    def orientations(self, random=True):
        if random:
            orientations = rotations.random_orientations(self.N)
        else:
            orientations = [self.Q for _ in range(self.N)]

        return orientations

    def taylor_model(self):
        single_model, _, _ = self.singlecrystal()
        orientations = self.orientations()
        tmodel = polycrystal.TaylorModel(
            single_model, orientations, nthreads=self.threads
        )
        return tmodel

    def plot_initial_pf(self, display=True, savefile=False):
        orientations = self.orientations()
        polefigures.pole_figure_discrete(orientations, [0, 0, 1], self.lattice())
        plt.title("Initial, <001>")
        if savefile:
            plt.savefig("initialpf-%i-C.pdf" % int(self.T - 273.15), dpi=300)
        if display:
            plt.show()
        return plt.close()

    def driver(self, full_res=True, use_taylor=True):

        if use_taylor:
            res = drivers.uniaxial_test(
                self.taylor_model(),
                erate=self.erate,
                emax=self.emax,
                sdir=self.ldir,
                T=self.T,
                verbose=True,
                full_results=full_res,
            )
        else:
            single_model, _, _ = self.singlecrystal()
            res = drivers.uniaxial_test(
                single_model,
                erate=self.erate,
                emax=self.emax,
                sdir=self.ldir,
                T=self.T,
                verbose=True,
                full_results=full_res,
            )

        return res

    def plot_and_save_ss(self, use_taylor=True, display=True, savefile=False):
        res = self.driver(full_res=False, use_taylor=use_taylor)
        plt.plot(res["strain"], res["stress"], "k-", lw=2)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Strain (mm/mm)", fontsize=14)
        plt.ylabel("Stress (MPa)", fontsize=14)
        plt.grid(False)
        plt.tight_layout()
        if savefile:
            plt.savefig("Stress-strain-{}.pdf".format(self.T), dpi=300)
        if display:
            plt.show()
        plt.close()

        data = pd.DataFrame({"strain": res["strain"], "stress": res["stress"]})
        data.to_csv("res_flow_{}.csv".format(int(self.T)))

        return data

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

    def hist_test(self, res):
        smodel, _, _ = self.singlecrystal()
        lattice = self.lattice()
        nt = len(res["history"])
        # initialize empty tensore for storing hist
        direct_from_model = np.zeros((nt, lattice.ntotal + 1))  # slip strain
        integrated_ourselves = np.zeros((nt, lattice.ntotal + 1))  # slip strain
        taus_from_model = np.zeros((nt, lattice.ntotal + 1))  # crss
        slip_rates = np.zeros((nt, lattice.ntotal + 1))  # slip rates
        resolved_shear_stresses = np.zeros((nt, lattice.ntotal + 1))  # rss

        for i in range(1, len(res["history"])):
            hist = history.History()
            smodel.populate_hist(hist)
            hist.set_data(res["history"][i])
            stress = tensors.Symmetric(self.usym(res["stress"][i]))
            T = res["temperature"][i]
            Q = hist.get_orientation("rotation")

            fixed = history.History()

            dt = res["time"][i] - res["time"][i - 1]

            for g in range(lattice.ngroup):
                for j in range(lattice.nslip(g)):
                    direct_from_model[i, 0] = hist.get_scalar("wall" + str(0))
                    direct_from_model[i, lattice.flat(g, j) + 1] = hist.get_scalar(
                        "wslip" + str(lattice.flat(g, j) + 1)
                    )
                    direct_from_model[i, lattice.flat(g, j)] = hist.get_scalar(
                        "islip" + str(lattice.flat(g, j) + 13)
                    )

        return (
            direct_from_model,
            integrated_ourselves,
            taus_from_model,
            slip_rates,
            resolved_shear_stresses,
        )

    def hist_evolution(self, res):
        smodel, _, _ = self.singlecrystal()
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
                            "pslip" + str(lattice.flat(g, j) + 24)
                        )
                    else:
                        direct_from_model[i, lattice.flat(g, j)] = hist.get_scalar(
                            "slip" + str(lattice.flat(g, j))
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
            plt.savefig("{}-{}.pdf".format(fname, self.T), dpi=300)
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
                plt.savefig("rss-group-{}.pdf".format(g), dpi=300)
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
                plt.savefig("crss-group-{}.pdf".format(g), dpi=300)
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
                plt.savefig("slip-rate-group-{}.pdf".format(g), dpi=300)
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
                plt.savefig("integrate-slip-strain-group-{}.pdf".format(g), dpi=300)
            if display:
                plt.show()
            plt.close()

        # plot accumulated slip strain evolution
        print("")
        print("plotting accumulated slip strain evolution")
        print("")
        _ = self.history_plot(
            integrated_ourselves[-1, :12] / self.N,
            "Slip System",
            "Accumulated Slip Strain",
            "integrate-slip-strain",
            display=display,
            savefile=savefile,
        )

        _ = self.history_plot(
            direct_from_model[-1, :12] / self.N,
            "Slip System",
            "Accumulated Slip Strain",
            "direct-slip-strain",
            display=display,
            savefile=savefile,
        )

        # plot accumulated twin strain evolution
        print("")
        print("plotting accumulated twin strain evolution")
        print("")
        _ = self.history_plot(
            integrated_ourselves[-1, 12:] / self.N,
            "Twin System",
            "Accumulated Twin Strain",
            "integrate-twin-strain",
            display=display,
            savefile=savefile,
        )

        _ = self.history_plot(
            direct_from_model[-1, 12:] / self.N,
            "Twin System",
            "Accumulated Twin Strain",
            "direct-twin-strain",
            display=display,
            savefile=savefile,
        )

        # plot hardening evolution
        print("")
        print("plotting accumulated hardening evolution")
        print("")
        _ = self.history_plot(
            taus_from_model[-1, :] / self.N,
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
        if self.use_ptr:
            if accum:
                return self.accumulate_history(res, 21, 33)
            else:
                return self.accumulate_history(res, 21, 33, accum=accum)
        else:
            if accum:
                return self.accumulate_history(res, 8, 20)
            else:
                return self.accumulate_history(res, 8, 20, accum=accum)

    def accumulated_twin(self, res, accum=True):
        if self.use_ptr:
            if accum:
                return self.accumulate_history(res, 33, 45)
            else:
                return self.accumulate_history(res, 33, 45, accum=accum)
        else:
            if accum:
                return self.accumulate_history(res, 20, 32)
            else:
                return self.accumulate_history(res, 20, 32, accum=accum)

    def accumulated_slip(self, res, accum=True):
        if self.use_ptr:
            if accum:
                return self.accumulate_history(res, 45, 57)
            else:
                return self.accumulate_history(res, 45, 57, accum=accum)
        else:
            if accum:
                return self.accumulate_history(res, 32, 44)
            else:
                return self.accumulate_history(res, 32, 44, accum=accum)

    def accumulated_twinn_fraction(self, res, accum=True):
        if self.use_ptr:
            if accum:
                return self.accumulate_history(res, 8, 20)
            else:
                return self.accumulate_history(res, 8, 20, accum=accum)
        else:
            raise ValueError("no twinner recording the fraction!!")

    def count_twinned(self, res, accum=True, display=True, savefile=False):
        if self.use_ptr:
            if accum:
                tf = self.accumulate_history(res, 20, 21)
            else:
                tf = self.accumulate_history(res, 20, 21, accum=accum)
        else:
            raise ValueError("no twinner recording the fraction!!")

        _ = self.history_plot(
            tf,
            "Twin Counting",
            "Twin count",
            "{}-twin-count".format(self.prefix),
            display=display,
            savefile=savefile,
        )

        return tf

    def save_accum_isv_dataframe(self, res, display=True, savefile=False):
        accu_density = self.accumulated_density(res) / self.N
        accu_twin = self.accumulated_twin(res) / self.N
        accu_slip = self.accumulated_slip(res) / self.N
        accu_tf = self.accumulated_twinn_fraction(res) / self.N

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

        # plot distribution of accumulated twin fraction
        _ = self.history_plot(
            accu_tf,
            "Twin System",
            "Accumulated Twin Fraction",
            "{}-twin-fraction".format(self.prefix),
            display=display,
            savefile=savefile,
        )

        data = pd.DataFrame(
            {
                "dis_density": accu_density**2 * 1.0e12,
                "accu_twin": accu_twin,
                "accu_slip": accu_slip,
                "accu_tf": accu_tf,
            }
        )

        data.to_csv("{}_res_{}.csv".format(self.prefix, int(self.T)))
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

        data.to_csv("{}_hist_{}.csv".format(self.prefix, int(self.T)))

        return data

    def deformed_texture(self, res, display=True, savefile=False):
        store_history = np.array(res["history"])
        pf = self.taylor_model().orientations(store_history[-1])
        polefigures.pole_figure_discrete(
            pf,
            [0, 0, 1],
            lattice=self.lattice(),
            x=tensors.Vector([1.0, 0, 0]),
            y=tensors.Vector([0, 1.0, 0]),
            axis_labels=["X", "Y"],
        )
        plt.title("Final, <001>")
        if savefile:
            plt.savefig("deformpf-%i-C-1.pdf" % int(self.T - 273.15), dpi=300)
        if display:
            plt.show()
        plt.close()

        polefigures.pole_figure_discrete(pf, [1, 0, 1], lattice=self.lattice())
        plt.title("Final, <101>")
        if savefile:
            plt.savefig("deformpf-%i-C-2.pdf" % int(self.T - 273.15), dpi=300)
        if display:
            plt.show()
        plt.close()

        polefigures.pole_figure_discrete(pf, [1, 1, 1], lattice=self.lattice())
        plt.title("Final, <111>")
        if savefile:
            plt.savefig("deformpf-%i-C-3.pdf" % int(self.T - 273.15), dpi=300)
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

        data.to_csv("{}_texture_{}.csv".format(self.prefix, int(self.T)))
        return data


def wallf(d, T):
    b = 0.256
    E = 200.0e3
    nu = 0.3
    mu = E / (2 * (1 + nu))
    k = 13806.49
    omega = 421750.0
    return mu * omega * (b**3) / (k * T * d)


x_exp = np.array([0, 5*3600, 25*3600, 100*3600])
y_exp = np.array([7.34e-7, 8.14e-7, 7.7e-7, 9.25e-7]) * 1.0e9


if __name__ == "__main__":

    Q = rotations.CrystalOrientation(
        0.0, 0.0, 0.0, angle_type="degrees", convention="kocks"
    )
    ldir = np.array([1, 0, 0, 0, 0, 0])
    prefixs = ["tension", "compression"]
    N, nthreads = 100, 4
    Ts = np.array([298, 773, 873, 923, 1023, 1173])
    erate, emax = 0.25/360000, 0.25  # 8.33e-5, np.log(1 + 0.5)

    E = 200.0e3
    nu = 0.3
    mu = E / (2 * (1 + nu))
    kw1_v = 9.537e-1 #1.13
    kw2_v = 1.4931e1 #50.0
    kw1 = np.ones((12,)) * kw1_v
    kw2 = np.ones((12,)) * kw2_v
    ki1_v = 1.12e-1 #1.13e-1
    ki2_v = 9.256e1 #50.0
    ki1 = np.ones((12,)) * ki1_v
    ki2 = np.ones((12,)) * ki2_v
    ftr = 4.861e-1 #0.1
    k0 = 10.0**(-5.59)
    Qv = 1.0e4
    initsigma = 75.0
    g0 = 1.0
    n = 20.0

    for current_T in Ts[3:4]:
        print("starting to calculate T :", current_T - 273.0)
        T = float(current_T)
        lanlti_model = am_model(
            Q,
            N,
            nthreads,
            T,
            ldir,
            prefixs[0],
            erate,
            emax,
            np.ones((12,)) * mu,
            kw1,
            kw2,
            ki1,
            ki2,
            ftr,
            k0,
            Qv, 
            initsigma,
            g0,
            n,
            E,
            nu,
            use_ptr=False,
        )

        # lanlti_model.plot_and_save_ss(use_taylor=True, display=True, savefile=False)
        res = lanlti_model.driver(full_res=False, use_taylor=True)
        
        plt.style.use(latex_style_times)
        plt.plot(res["strain"], res["stress"], lw=4)
        plt.xlabel("Strain", fontsize=23)
        plt.ylabel("Stress (MPa)", fontsize=23)
        plt.tick_params(axis="both", which="major", labelsize=23)
        plt.tight_layout()
        plt.savefig("stress-strain.pdf".format(int(T - 273.0)))
        plt.show()
        plt.close()
        
        """
        # lanlti_model.plot_initial_pf(display=True, savefile=False)
        lanlti_model.deformed_texture(res, display=False, savefile=True)
        hist, _, _, _, _ = lanlti_model.hist_test(res)
        
        plt.style.use(latex_style_times)
        plt.plot(np.array(res["time"])[1:] / 3600.0, hist[1:, 0], lw=4)
        plt.plot(x_exp / 3600.0, y_exp, "o", color="k", markersize=10)
        plt.xlabel("Time (hr)", fontsize=23)
        plt.ylabel("Wall size (nm)", fontsize=23)
        plt.tick_params(axis="both", which="major", labelsize=23)
        plt.tight_layout()
        plt.savefig("wall-size-{}.pdf".format(int(current_T - 273.0)))
        plt.show()
        plt.close()
        
        plt.style.use(latex_style_times)
        plt.plot(np.array(res["time"])[1:] / 3600.0, wallf(hist[1:, 0], T) * 100.0, lw=4)
        plt.xlabel("Time (hr)", fontsize=23)
        plt.ylabel(r"Wall fraction $(\%)$", fontsize=23)
        plt.tick_params(axis="both", which="major", labelsize=23)
        plt.tight_layout()
        plt.savefig("wall-fraction-{}.pdf".format(int(current_T - 273.0)))
        plt.show()
        plt.close()
        # lanlti_model.rss_history(res, display=False, savefile=True)
        # lanlti_model.save_accum_isv_dataframe(res, display=False, savefile=True)
        # lanlti_model.save_evolve_isv_dataframe(res)
        # lanlti_model.save_texture(res)
        # lanlti_model.count_twinned(res, display=False, savefile=True)
        """