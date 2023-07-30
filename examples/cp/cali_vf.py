import sys
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib.pyplot as plt


def interp(x, y, xnew):
    return interpolate.interp1d(x, y)(xnew)


def vf(x, omega, inte):
    T = 650.0 + 273.0
    E = 200.0e3
    nu = 0.3
    mu = E / (2 * (1 + nu))
    b = 0.256
    kb = 13806.49
    return mu * omega * b**3 / (kb * T * x) + inte


if __name__ == "__main__":




    # Load results from the optimization
    time = np.loadtxt("time-history.txt")
    pred_size = np.loadtxt("wall-size-pred.txt")
    true_size = np.loadtxt("wall-size-true.txt")

    # Define font size of plot
    fs = 23

    plt.plot(time / 3600, pred_size, lw=4, label="prediction")
    plt.plot(time / 3600, true_size, "ko", markevery=20, label="actual")
    ax = plt.gca()
    plt.xlabel("Time (hr)", fontsize=fs)
    plt.ylabel("Cellular size (nm)", fontsize=fs)
    plt.legend(prop={'size':fs})
    plt.tick_params(axis="both", which="major", labelsize=fs)
    plt.locator_params(axis="both", nbins=5)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("ws.pdf")
    plt.show()
    plt.close()
    # sys.exit("stop")

    # Unit converter from actual(m) to neml unit(nm)
    uf = 1.0e9
    # Actual measured wall size
    md = np.array([7.34e-7, 7.75e-7, 6.22e-7, 6.30e-7, 8.49e-7]) * uf
    # md = np.array([7.34e-7, 8.49e-7]) * uf
    # Actual aging period
    exp_time = np.array([0, 5 * 3600, 25 * 3600, 100 * 3600, 501 * 3600])
    # exp_time = np.array([0, 501 * 3600])
    # Actual measured volume fraction f = 2 * w / d, w is the wall thickness
    mvf = np.array([67.0, 42.0, 31.0, 24.0, 15.0]) * 2 / md
    # mvf = np.array([67.0, 15.0]) * 2 / md

    true_res = interp(exp_time, mvf, time)

    popt, pcov = curve_fit(vf, pred_size, true_res)
    print("volume fraction param is:", popt)
    plt.plot(
        time / 3600,
        vf(pred_size, *popt),
        "r-",
        lw=4,
        #label="fit: w=%5.3f, intcep=%5.3f" % tuple(popt),
        label="fit",
    )
    plt.plot(exp_time / 3600, mvf, "b-", lw=4, label="data")
    ax = plt.gca()
    plt.xlabel("Time (hr)", fontsize=fs)
    plt.ylabel(r"Wall fraction ($\%$)", fontsize=fs)
    plt.legend(prop={'size': fs})
    plt.tick_params(axis="both", which="major", labelsize=fs)
    plt.locator_params(axis="both", nbins=5)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
    plt.tight_layout()
    plt.savefig("vf.pdf")
    plt.show()
    plt.close()
