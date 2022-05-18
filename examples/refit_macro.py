#!/usr/bin/env python3
import sys

sys.path.append("..")
import os, glob
import numpy as np
import numpy.linalg as la
import numpy.random as ra
import scipy.interpolate as inter
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
from optimparallel import minimize_parallel

from neml import (
    solvers,
    models,
    elasticity,
    drivers,
    surfaces,
    hardening,
    ri_flow,
    creep,
    general_flow,
    visco_flow,
    interpolate,
)


class extrapolate:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def value(self, T):
        func = np.poly1d(np.polyfit(self.xs, self.ys, deg=1))
        return func(T).tolist()


class macro_model:
    def __init__(
        self,
        ns,
        etas,
        ks,
        Qs,
        bs,
        Ttarget,
        C11,
        C12,
        Ts,
        loading_rate,
        applied_temperature,
        nu=0.34,
        emax=0.15,
        nsteps=250,
        sdir=np.array([0, 0, -1, 0, 0, 0]),
    ):
        self.ns = ns
        self.etas = etas
        self.ks = ks
        self.Qs = Qs
        self.bs = bs
        self.Ttarget = Ttarget
        self.C11 = C11
        self.C12 = C12
        self.Ts = Ts
        self.nu = nu
        self.loading_rate = loading_rate
        self.applied_temperature = applied_temperature
        self.emax = emax
        self.nsteps = nsteps
        self.sdir = sdir

    def extrap(self, xs, ys, T):
        func = np.poly1d(np.polyfit(xs, ys, deg=1))
        return func(T)

    def isomodulus(self):
        C11_target = self.extrap(self.Ts, self.C11, self.Ttarget)
        C12_target = self.extrap(self.Ts, self.C12, self.Ttarget)
        mu = (C11_target - C12_target) / 2
        E = 2 * (1 + self.nu) * mu
        K = E / (3 * (1 - 2 * self.nu))
        return mu, K

    def make_model(
        self,
        verbose=False,
        write_out=False,
    ):

        """
        Function that returns a NEML model given the parameters
        """
        mu = interpolate.PiecewiseLinearInterpolate(
            list(self.Ttarget), list(self.isomodulus()[0])
        )
        K = interpolate.PiecewiseLinearInterpolate(
            list(self.Ttarget), list(self.isomodulus()[1])
        )
        n = interpolate.PiecewiseLinearInterpolate(list(self.Ttarget), list(self.ns))
        eta = interpolate.PiecewiseLinearInterpolate(
            list(self.Ttarget), list(self.etas)
        )
        k = interpolate.PiecewiseLinearInterpolate(list(self.Ttarget), list(self.ks))
        Q = interpolate.PiecewiseLinearInterpolate(list(self.Ttarget), list(self.Qs))
        b = interpolate.PiecewiseLinearInterpolate(list(self.Ttarget), list(self.bs))
        elastic = elasticity.IsotropicLinearElasticModel(mu, "shear", K, "bulk")
        surface = surfaces.IsoJ2()
        iso = hardening.VoceIsotropicHardeningRule(k, Q, b)
        gpower = visco_flow.GPowerLaw(n, eta)
        vflow = visco_flow.PerzynaFlowRule(surface, iso, gpower)
        flow = general_flow.TVPFlowRule(elastic, vflow)
        model = models.GeneralIntegrator(elastic, flow, verbose=False)
        if write_out:
            model.save(
                "materials_{}.xml".format(self.loading_rate),
                "ti_macro",
                )
        return drivers.uniaxial_test(
            model,
            erate=self.loading_rate,
            T=self.applied_temperature,
            emax=self.emax,
            nsteps=self.nsteps,
            sdir=self.sdir,
            verbose=verbose,
        )


# ================================================#
def load_file(path, file_name):
    # ================================================#
    fnames = glob.glob(path + file_name + "*.csv")
    for f in fnames:
        df = pd.read_csv(f)

    return df


# ================================================#
def read_params(path, T, rate):
    # ================================================#
    fnames = glob.glob(path + "*.csv")
    ns = []
    etas = []
    ks = []
    Qs = []
    bs = []
    for i in T:
        for f in fnames:
            temp = os.path.basename(f).split("_")[3]
            strain_rate = os.path.basename(f).split("_")[5].split(".csv")[0]

            if float(strain_rate) == rate and float(temp) == i:
                print(float(strain_rate), temp)
                df = pd.read_csv(f)
                ns.append(float(df["n"].values))
                etas.append(float(df["eta"].values))
                ks.append(float(df["k"].values))
                Qs.append(float(df["Q"].values))
                bs.append(float(df["b"].values))

    return ns, etas, ks, Qs, bs


if __name__ == "__main__":

    # define the hyperparameters
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
    C12 = np.array(
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
    Ttarget = np.array([600.0, 750.0, 900.0]) + 273.0
    param_path = "/mnt/c/Users/ladmin/Desktop/argonne/Ti_model_fitting/isotropic_fitting/refit_on_cp/"
    loading_rate = 1.0e-3
    applied_temperature = 750.0 + 273.0
    ns, etas, ks, Qs, bs = read_params(param_path, Ttarget, loading_rate)

    print(ns, etas, ks, Qs, bs)
    
    model = macro_model(
        ns, etas, ks, Qs, bs, Ttarget, C11, C12, Ts, loading_rate, applied_temperature
    )
    # running the model
    res = model.make_model(write_out=True)
    path = "/mnt/c/Users/ladmin/Desktop/argonne/Ti_model_fitting/huang_fitting/calibration_visualization/engss_fit/1e-3/"
    file = "res_1023"
    df = load_file(path, file)

    plt.plot(
        df["strain"], df["stress"], label="micro-{}".format(int(applied_temperature))
    )
    plt.plot(
        res["strain"], res["stress"], label="macro-{}".format(int(applied_temperature))
    )
    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")
    plt.legend()
    # plt.xlim([0.0, 0.006])
    plt.grid(True)
    # plt.savefig("isotropic_fitting_{}_of_{}.png".format(float(loading_rate), int(T)), dpi = 300)
    plt.show()
    plt.close()
    """
    data = pd.DataFrame({
        "n": n_n,
        "eta": eta_n,
        "k": k_n,
        "Q": Q_n,
        "b": b_n
        }, index=[0]) 
    data.to_csv('optim_params_of_{}_under_{}.csv'.format(float(loading_rate), int(T)))
    
    # save the stress-strain data for future use
    data = pd.DataFrame({"strain": final_res["strain"], "stress": final_res["stress"]})
    data.to_csv("macro_{}_{}.csv".format(int(T), float(rate)))
    """
