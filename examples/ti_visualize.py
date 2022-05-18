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
    rate = "1e-2"
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

    # Load moose calculated results
    path = "/mnt/c/Users/ladmin/Desktop/argonne/RVE_Ti/Ti_block/macro_test/"
    file_name = "ti_macro_out"
    df = load_file(path, file_name)

    # Load optimized xml file
    model = parse.parse_xml("ti_macro_single_1e-2_873.xml", "forest")
    res = drivers.uniaxial_test(model, erate=0.01)

    # Plot results against each case

    # convert to engineering stress-strain
    # eng_strain = np.exp(df["strain_zz"]) - 1.0
    # eng_stress = df["stress_zz"] / (1 + eng_strain)
    # plt.plot(eng_strain, eng_stress, "k-")
    input_eng_strain = df["strain"]
    input_eng_stress = df["stress"]
    plt.plot(eng_strain, eng_stress, label="experiment")
    plt.plot(input_eng_strain, input_eng_stress, label="input xml")
    plt.plot(res["strain"], res["stress"], label="fea solve")
    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")
    plt.legend()
    plt.show()
    plt.close()
