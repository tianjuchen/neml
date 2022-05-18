#!/usr/bin/env python3
import sys
import os, glob
import numpy as np
import numpy.linalg as la
import numpy.random as ra
import scipy.interpolate as inter
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
from optimparallel import minimize_parallel

from neml import solvers, models, elasticity, drivers, surfaces, hardening, ri_flow, creep, general_flow, visco_flow

temperature = 873.0
loading_rate = "1e-2"

class extrapolate:

  def __init__(self, xs, ys):
    self.xs = xs
    self.ys = ys
    
  def value(self, T):
    func = np.poly1d(np.polyfit(self.xs, self.ys, deg=1))
    return func(T).tolist()

def make_model(n, eta, k, Q, b, erate=float(loading_rate),
               T = temperature, emax = 0.15, nsteps = 250,
               sdir = np.array([0,0,-1,0,0,0]), verbose = False):
    """
      Function that returns a NEML model given the parameters
    """
    Ts = np.array([20.0, 100.0, 200.0, 300.0, 400.0, 500.0]) + 273.15
    Es = np.array([110.0, 101.0, 92.0, 85.0, 78.0, 72.0]) * 1.0e3
    E = extrapolate(Ts, Es).value(T)
    nu = 0.34
    mu = E / (2 * (1.0 + nu))
    K = E / (3 * (1 - 2 * nu))
    elastic = elasticity.IsotropicLinearElasticModel(mu, "shear", K, "bulk")
    surface = surfaces.IsoJ2()
    iso = hardening.VoceIsotropicHardeningRule(k, Q, b)
    gpower = visco_flow.GPowerLaw(n, eta)
    vflow = visco_flow.PerzynaFlowRule(surface, iso, gpower)
    flow = general_flow.TVPFlowRule(elastic, vflow)
    model = models.GeneralIntegrator(elastic, flow, verbose = False)
    return drivers.uniaxial_test(model, erate=erate, T=T, emax=emax, 
                               nsteps=nsteps, sdir=sdir, verbose = verbose)

def interpolate(strain, stress, targets):
    """
        This is just to make sure all our values line up, in case the model
        adaptively integrated or something
    """
    return inter.interp1d(strain, stress)(targets)   
  

#================================================#
def load_file(path, T, rate):
#================================================#
    fnames = glob.glob(path + "*.csv")
    for f in fnames:
        strain_rate = os.path.basename(f).split('_')[0]
        if strain_rate == rate:
            temp = os.path.basename(f).split('_')[1].split('.')[0]
        if strain_rate == rate and temp == str(int(T)) + 'k':
            df = pd.read_csv(f, usecols=[0,1], names=['True_strain', 'True_stress'], header=None)
            return df

#================================================#
def read_file(path, T, rate):
#================================================#
    fnames = glob.glob(path + "*.csv")
    for f in fnames:
        strain_rate = os.path.basename(f).split('_')[0]
        if strain_rate == rate:
            temp = os.path.basename(f).split('_')[1].split('.')[0]
        if strain_rate == rate and temp == str(int(T)) + 'k':
            df = pd.read_csv(f)
            return df

# ================================================#
def load_cp_data(path, file):
    # ================================================#
    fnames = glob.glob(path + file_name + "*.csv")
    for f in fnames:
        df = pd.read_csv(f)

    return df
    

if __name__ == "__main__":
        
    max_strain = 0.1
    ndatas = 1000
    nominal_strain = np.linspace(0.05, max_strain, ndatas)
    T = temperature
    # Load moose calculated results
    path = "/home/tianju.chen/ti_cp_fitting/engss_fit/1e-2/"
    file_name = "res_873"
    df = load_cp_data(path, file_name)    
    
    n_range = [1.0, 20.0]
    eta_range = [10.0, 500.0]
    k_range = [2.0, 10.0]
    Q_range = [22.5, 22.5]
    b_range = [1.0, 20.0]
    
    p0 = [ra.uniform(*n_range), ra.uniform(*eta_range),
          ra.uniform(*k_range), ra.uniform(*Q_range),
          ra.uniform(*b_range)]
    
    def R(params):
        res = make_model(*params)
        model_stress = interpolate(res['strain'], res['stress'], nominal_strain)
        stress = interpolate(df['strain'], df['stress'], nominal_strain)
        R = la.norm(stress - model_stress)
        print("Current residual: %e" % R)
        return R

    flag = False
    iq = 0
    while not flag:
        res = minimize_parallel(R, p0, bounds = [n_range, eta_range, 
                    k_range, Q_range, b_range])
        print(res.success)
        if res.success == True:
            flag = True
            print(res.success)
            print(res.x)
        iq += 1
        if iq >=10:
            raise ValueError("Not able to optimize the initialize")

    n_n, eta_n, k_n, Q_n, b_n = (res.x[0], res.x[1], res.x[2], res.x[3], res.x[4])


    final_res = make_model(n_n, eta_n, k_n, Q_n, b_n)
    plt.plot(df['strain'], df['stress'], label = 'Exp-{}'.format(int(T)))
    plt.plot(final_res['strain'], final_res['stress'], label = 'Model-{}'.format(int(T)))
    plt.xlabel('Strain (mm/mm)')
    plt.ylabel('Stress (MPa)')
    plt.legend()
    plt.grid(True)
    plt.savefig("isotropic_fitting_{}_of_{}.png".format(float(loading_rate), int(T)), dpi = 300)
    plt.show()
    plt.close()    

    data = pd.DataFrame({
        "n": n_n,
        "eta": eta_n,
        "k": k_n,
        "Q": Q_n,
        "b": b_n
        }, index=[0]) 
    data.to_csv('optim_params_of_{}.csv'.format(int(T)))
