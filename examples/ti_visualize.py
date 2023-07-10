#!/usr/bin/env python3

from neml import (
    solvers,
    models,
    elasticity,
    drivers,
    surfaces,
    hardening,
    visco_flow,
    general_flow,
    parse,
)
import glob, sys, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import RcParams
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches


latex_style_times = RcParams(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
    }
)


# ================================================#
def load_file(path, file):
    # ================================================#
    fnames = glob.glob(path + file_name + "*.csv")
    for f in fnames:
        df = pd.read_csv(f)

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
      df = pd.read_csv(f, usecols=[0,1], names=['True_strain', 'True_stress'], header=None)
      return df


if __name__ == "__main__":

    # Load experimental data
    path_1 = "/mnt/c/Users/ladmin/Desktop/argonne/RTRC_data_extract/Huang-2007-MSEA/"
    path_2 = "/mnt/c/Users/ladmin/Desktop/argonne/RTRC_data_extract/Yapici-2014-MD/"
    path_3 = "/mnt/c/Users/ladmin/Desktop/argonne/RTRC_data_extract/DIY-extrapolate/"
    T = 873.0
    emax = 0.4
    rate = "1e-3"
    if T <= 1200.0:
        if T < 1000.0:
            path = path_1
        else:
            path = path_2

        df = read_file(path, T, rate)
        eng_strain = np.exp(df["True_strain"]) - 1.0
        eng_stress = df["True_stress"] / (1 + eng_strain)
    else:
        path = path_3
        df = read_file(path, T, rate)
        eng_strain = df["strain"]
        eng_stress = df["stress"]

    # Load optimized xml file
    model = parse.parse_xml("mmc-ti-1e-3.xml", "ti_macro")
    res = drivers.uniaxial_test(model, erate=0.001, T=T, emax=emax)

    # Plot results against each case
    plt.style.use(latex_style_times)
    plt.plot(eng_strain[7:], eng_stress[7:], 'ko', markersize=10, markevery=1, label="experiment")
    plt.plot(res["strain"], res["stress"], lw=4, label="model")
    ax = plt.gca()
    plt.xlabel("Strain (mm/mm)", fontsize=27)
    plt.ylabel("Stress (MPa)", fontsize=27)
    plt.xlim([-0.005, 0.1])
    plt.tick_params(axis="both", which="major", labelsize=27)
    plt.locator_params(axis="both", nbins=4)
    # ax.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.legend(prop={'size': 27})
    plt.savefig("ti-flow-{}-{}.pdf".format(float(rate), int(T-273.0)))
    plt.show()
    plt.close()
