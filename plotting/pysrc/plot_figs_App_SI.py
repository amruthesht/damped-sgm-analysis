'''
File: plot_figs_App_SI.py
Project: meta_paper_plot
File Created: Monday, 20th December 2021 9:37:34 pm
Author: Amruthesh T (amru@seas.upenn.edu)
-----
Last Modified: Friday, 26th August 2022 11:05:27 pm
Modified By: Amruthesh T (amru@seas.upenn.edu)
-----
Copyright (c) 2021 - 2022 Amru, University of Pennsylvania

Summary: Fill In
'''

#%%
from os import popen, makedirs, system, walk
from os.path import join, isfile, isdir, basename, dirname, exists
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import os
import random
from os.path import join
from cycler import cycler
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import LogLocator
from mpl_toolkits import mplot3d
from sklearn.metrics import pairwise_distances
from statsmodels.graphics import tsaplots
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from ast import literal_eval

#Options
from cycler import cycler
fig_width_pt = 246 #320 #/2 or 510   # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          #'axes.labelsize': 10,
          'font.size': 9,
          'font.weight': "normal",
          "font.family": "serif",
          "font.serif": ['Computer Modern Roman'],
          'legend.fontsize': 6,
          'xtick.labelsize': 7.5,
          'ytick.labelsize': 7.5,
          'xtick.major.width': 0.4,
          'xtick.minor.width': 0.3,
          'ytick.major.width': 0.4,
          'ytick.minor.width': 0.3,
          'text.usetex': True,
          'axes.linewidth': 0.5,
          'axes.prop_cycle': cycler(color='bgrcmyk'),
          'figure.figsize': fig_size
          }
plt.rcParams.update(params)

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')

def abline(slope, xi, xf, yi):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array([xi, xf])
    y_vals = yi + slope * (x_vals - xi)
    plt.plot(x_vals, y_vals, '--')

def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

def autocorrelate_graipher(data):
    data = np.array(data)
    M = np.mean(data, axis=1).reshape(data.shape[0], 1)
    S = np.sqrt(np.sum(np.var(data, axis=1)))
    data_z = (data - M) / S
    AC = np.zeros(data_z.shape[1])
    # print(data.shape, AC.shape)
    tau = np.arange(0, data_z.shape[1])
    for t in tau:
        i_s = np.arange(data_z.shape[1] - t)
        AC[t] = np.mean([np.sum(np.multiply(data_z[:,i], data_z[:,i+t]), axis=0) for i in i_s])
    return AC

def costheta(v1, v2):
    return (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def angle(v1, v2):
    return np.abs(np.arccos(np.round(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 12)))

def angle_affinity(X):
    return pairwise_distances(X, metric=angle)

def periodic_BC_norm(x1, x2, L):
    return np.linalg.norm((x1 - x2) - np.multiply(L, np.round(np.divide((x1 - x2), L))))

# ["blue", "green", "red", "cyan", "saddlebrown", "magenta", "darkviolet", "gold"]
colors = ["blue", "green", "red", "cyan", "saddlebrown", "magenta", "darkviolet", "orange", "gold", "blue", "green", "red"]
tabcolors = ["tab:blue", "tab:green", "tab:red", "tab:cyan", "tab:purple", "tab:olive", "tab:blue", "tab:green", "tab:red"]
markers = ["s", "o", "^"]
s = [14.5, 15, 15.5]

blist = [0.0, 0.0001, 0.001, 0.01, 0.04, 0.1, 0.3, 0.5, 1]

data_foldername = "Data/"
data_type_foldername = "./"

output_foldername = "output/"
graphs_foldername = "graphs/"
input_foldername = "input/"
input_filename = "init.input"
run_filename = "run_config.txt"

d = 3

########################################################################################################################

system_foldername = "./"
N_foldername = "./"

type_foldername = "Damped_ripening/"

PATH = join(data_foldername, data_type_foldername, system_foldername, N_foldername, type_foldername)

dirs = [dirs for root, dirs, files in walk(join(".", PATH))]
# print((join(".", PATH)), dirs)
dirs = np.array([dir for dir in dirs[0]])

for i, dir in np.ndenumerate(dirs):
    if isfile(join(PATH, dir, output_foldername, run_filename)):
        f = pd.read_csv(join(PATH, dir, output_foldername, run_filename), lineterminator="\n",
                        header=None, skip_blank_lines=False)
        f = f[0].str.split(r"\s{2,}", expand=True)
        f.columns = ["value", "name"]

        N_init = float(
            f.loc[f["name"] == "N_init"]["value"].values[0])

        b = float(
            f.loc[f["name"] == "b"]["value"].values[0])

        random_seed_pos_init = float(
            f.loc[f["name"] == "random_seed_pos_init"]["value"].values[0])

        dt_minimizer = float(
            f.loc[f["name"] == "dt_coarsening"]["value"].values[0])

        U_filename = str(f.loc[f["name"] == "U_filename"]["value"].values[0])

        Z_filename = str(f.loc[f["name"] == "Z_filename"]["value"].values[0])

        R_filename = str(f.loc[f["name"] == "R_filename"]["value"].values[0])

        if (b < 10):
            U_temp = pd.read_csv(join(PATH, dir, output_foldername, U_filename), sep=r"\s+",
                                    header=None, skiprows=1)
            U_temp.columns = ["system_counter", "t", "counter", "counter_time", "status_flag", "U", "dU", "dU_U", "N", "min_counter_SUCCESSFULL", "min_counter"]

            U_temp["b"] = b
            U_temp["random_seed_pos_init"] = random_seed_pos_init
            U_temp["dt_minimizer"] = dt_minimizer
            U_temp["simkey"] = i[0]

            try:
                U
            except NameError:
                U = U_temp
            else:
                U = pd.concat([U, U_temp], ignore_index=True)

            Z_temp = pd.read_csv(join(PATH, dir, output_foldername, Z_filename), sep=r"\s+", header=None, skiprows=1)
            Z_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "Nc >= 0", "Nc >= 1", "Nc >= 2",
                          "Nc >= 3", "Nc >= 4", "Nc >= 5", "Nc >= 6", "N_nr >= 0", "N_nr >= 1", "N_nr >= 2", "N_nr >= 3", "N_nr >= 4", "N_nr >= 5", "N_nr >= 6", "vol_frac", "N"]

            Z_temp["b"] = b
            Z_temp["random_seed_pos_init"] = random_seed_pos_init
            Z_temp["dt_minimizer"] = dt_minimizer
            Z_temp["simkey"] = i[0]

            try:
                Z
            except NameError:
                Z = Z_temp
            else:
                Z = pd.concat([Z, Z_temp], ignore_index=True)

            R_temp = pd.read_csv(join(PATH, dir, output_foldername, R_filename), sep=r"\s+", header=None, skiprows=1)
            R_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "N",
                          "mass_init - mass_N", "<R>", "<R>^2", "sqrt(<R^2>)", "R_1", "R_2", "R_3", "R12 < L"]

            R_temp["b"] = b
            R_temp["random_seed_pos_init"] = random_seed_pos_init
            R_temp["dt_minimizer"] = dt_minimizer
            R_temp["simkey"] = i[0]

            try:
                R
            except NameError:
                R = R_temp
            else:
                R = pd.concat([R, R_temp], ignore_index=True)

system_foldername = "./"
N_foldername = "./"

type_foldername = "Damped_IS/"

PATH = join(data_foldername, data_type_foldername, system_foldername, N_foldername, type_foldername)

dirs = [dirs for root, dirs, files in walk(join(".", PATH))]
# print((join(".", PATH)), dirs)
dirs = np.array([dir for dir in dirs[0]])

for i, dir in np.ndenumerate(dirs):
    if isfile(join(PATH, dir, output_foldername, run_filename)):
        f = pd.read_csv(join(PATH, dir, output_foldername, run_filename), lineterminator="\n",
                        header=None, skip_blank_lines=False)
        f = f[0].str.split(r"\s{2,}", expand=True)
        f.columns = ["value", "name"]

        N_init = float(
            f.loc[f["name"] == "N_init"]["value"].values[0])

        b = float(
            f.loc[f["name"] == "b"]["value"].values[0])

        random_seed_pos_init = float(
            f.loc[f["name"] == "random_seed_pos_init"]["value"].values[0])

        dt_minimizer = float(
            f.loc[f["name"] == "dt_coarsening"]["value"].values[0])

        U_filename = str(f.loc[f["name"] == "U_filename"]["value"].values[0])

        Z_filename = str(f.loc[f["name"] == "Z_filename"]["value"].values[0])

        R_filename = str(f.loc[f["name"] == "R_filename"]["value"].values[0])

        s_contour_filename = str(f.loc[f["name"] == "s_contour_filename"]["value"].values[0])

        if (b < 10):
            U_IS_temp = pd.read_csv(join(PATH, dir, output_foldername, U_filename), sep=r"\s+",
                                    header=None, skiprows=1)
            U_IS_temp.columns = ["system_counter", "t", "counter", "counter_time", "status_flag", "U", "dU", "dU_U", "N", "min_counter_SUCCESSFULL", "min_counter"]

            U_IS_temp["b"] = b
            U_IS_temp["random_seed_pos_init"] = random_seed_pos_init
            U_IS_temp["dt_minimizer"] = dt_minimizer
            U_IS_temp["simkey"] = i[0]

            try:
                U_IS
            except NameError:
                U_IS = U_IS_temp
            else:
                U_IS = pd.concat([U_IS, U_IS_temp], ignore_index=True)

            Z_IS_temp = pd.read_csv(join(PATH, dir, output_foldername, Z_filename), sep=r"\s+", header=None, skiprows=1)
            Z_IS_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "Nc >= 0", "Nc >= 1", "Nc >= 2",
                          "Nc >= 3", "Nc >= 4", "Nc >= 5", "Nc >= 6", "N_nr >= 0", "N_nr >= 1", "N_nr >= 2", "N_nr >= 3", "N_nr >= 4", "N_nr >= 5", "N_nr >= 6", "vol_frac", "N"]

            Z_IS_temp["b"] = b
            Z_IS_temp["random_seed_pos_init"] = random_seed_pos_init
            Z_IS_temp["dt_minimizer"] = dt_minimizer
            Z_IS_temp["simkey"] = i[0]

            try:
                Z_IS
            except NameError:
                Z_IS = Z_IS_temp
            else:
                Z_IS = pd.concat([Z_IS, Z_IS_temp], ignore_index=True)

            R_IS_temp = pd.read_csv(join(PATH, dir, output_foldername, R_filename), sep=r"\s+", header=None, skiprows=1)
            R_IS_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "N",
                          "mass_init - mass_N", "<R>", "<R>^2", "sqrt(<R^2>)", "R_1", "R_2", "R_3", "R12 < L"]

            R_IS_temp["b"] = b
            R_IS_temp["random_seed_pos_init"] = random_seed_pos_init
            R_IS_temp["dt_minimizer"] = dt_minimizer
            R_IS_temp["simkey"] = i[0]

            try:
                R_IS
            except NameError:
                R_IS = R_IS_temp
            else:
                R_IS = pd.concat([R_IS, R_IS_temp], ignore_index=True)

            s_IS_temp = pd.read_csv(join(PATH, dir, output_foldername, s_contour_filename), sep=r"\s+",
                                    header=None, skiprows=1)
            s_IS_temp.columns = ["system_counter", "t", "counter", "counter_time", "status_flag", "s_displacement",
                                "s", "ds", "contour", "dcontour", "N", "s_non_rattlers", "ds_non_rattlers", "N_non_rattlers"]

            s_IS_temp["b"] = b
            s_IS_temp["random_seed_pos_init"] = random_seed_pos_init
            s_IS_temp["dt_minimizer"] = dt_minimizer
            s_IS_temp["simkey"] = i[0]

            try:
                s_IS
            except NameError:
                s_IS = s_IS_temp
            else:
                s_IS = pd.concat([s_IS, s_IS_temp], ignore_index=True)

########################################################################################################################

fig = plt.figure()
fig.set_figheight(fig_height)
fig.set_figwidth(fig_width)

ax1 = plt.subplot2grid(shape=(2, 1), loc=(0, 0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid(shape=(2, 1), loc=(1, 0), rowspan=1, colspan=1)

bplot = [0.1, 1]

seed_plot = 2

for b in np.sort(bplot):
    U_plot_IS = U_IS.loc[((U_IS["status_flag"]).astype(float).astype(int) == 0) & (U_IS["b"] == b) & (U_IS["t"] >= 180) & (U_IS["t"] <= 480) & (U_IS["random_seed_pos_init"] == seed_plot)]
    U_plot_rp = U_IS.loc[((U_IS["status_flag"]).astype(float).astype(int) == 2) & (U_IS["b"] == b) & (U_IS["t"] >= 180) & (U_IS["t"] <= 480) & (U_IS["random_seed_pos_init"] == seed_plot)]

    if b == 1:
        x4, y4 = np.array(U_plot_rp["t"]), np.array(U_plot_rp["U"])
        x41, y41 = np.array(U_plot_IS["t"]), np.array(U_plot_IS["U"])
        ax1.plot(x4, y4, color=colors[blist.index((b))], zorder=-1, label=r"$\xi = {}$".format(b))
        ax1.plot(x41, y41, color="black", label=r"$U_{IS}$", alpha=0.85, linewidth=0.85)
    elif b == 0.1:
        x5, y5 = np.array(U_plot_rp["t"]), np.array(U_plot_rp["U"])
        x51, y51 = np.array(U_plot_IS["t"]), np.array(U_plot_IS["U"])
        ax2.plot(x5, y5, label=r"$\xi = {}$".format(b), color=colors[blist.index((b))], zorder=-1)
        ax2.plot(x51, y51, label=r"$U_{IS}$", color="black", alpha=0.85, linewidth=0.85)

# labels only for the last plot
ax1.legend(loc="upper right", frameon = False)
ax2.legend(loc="upper right", frameon = False)
ax2.set_xlabel(r"$t$")
plt.annotate(text=r"$U$", xy=(0.015, 0.5), xycoords='figure fraction', fontsize=9, rotation=90)

# remove x labels but not ticks for all but the last plot
ax1.tick_params(which="both", axis='x', bottom=True, top=False, labelbottom=False, direction="in")
ax2.tick_params(which="both", axis='x', bottom=True, top=False, labelbottom=True, direction="in")
# ax7.tick_params(which="both", axis='x', bottom=True, top=False, labelbottom=True, direction="in")
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
fig.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0.15)

fig.savefig(output_foldername + graphs_foldername +
            "FIG_SI_UIS.jpg", dpi=1000, bbox_inches='tight')
plt.close()

########################################################################################################################

del R

seed_plot = 4

system_foldername = "./"
N_foldername = "./"

type_foldername = "Damped_ripening/"

PATH = join(data_foldername, data_type_foldername, system_foldername, N_foldername, type_foldername)

dirs = [dirs for root, dirs, files in walk(join(".", PATH))]
# print((join(".", PATH)), dirs)
dirs = np.array([dir for dir in dirs[0]])

for i, dir in np.ndenumerate(dirs):
    if isfile(join(PATH, dir, output_foldername, run_filename)):
        f = pd.read_csv(join(PATH, dir, output_foldername, run_filename), lineterminator="\n",
                        header=None, skip_blank_lines=False)
        f = f[0].str.split(r"\s{2,}", expand=True)
        f.columns = ["value", "name"]

        N_init = float(
            f.loc[f["name"] == "N_init"]["value"].values[0])

        b = float(
            f.loc[f["name"] == "b"]["value"].values[0])

        random_seed_pos_init = float(
            f.loc[f["name"] == "random_seed_pos_init"]["value"].values[0])

        dt_minimizer = float(
            f.loc[f["name"] == "dt_coarsening"]["value"].values[0])

        R_filename = str(
            f.loc[f["name"] == "R_filename"]["value"].values[0])

        if (b < 10):
            R_temp = pd.read_csv(join(PATH, dir, output_foldername, R_filename), sep=r"\s+", header=None, skiprows=1)
            R_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "N",
                          "mass_init - mass_N", "<R>", "<R>^2", "sqrt(<R^2>)", "R_1", "R_2", "R_3", "R12 < L"]

            R_temp["b"] = b
            R_temp["random_seed_pos_init"] = random_seed_pos_init
            R_temp["dt_minimizer"] = dt_minimizer
            R_temp["simkey"] = i[0]

            try:
                R
            except NameError:
                R = R_temp
            else:
                R = pd.concat([R, R_temp], ignore_index=True)

fig = plt.figure()
fig.set_figheight(fig_height)
fig.set_figwidth(fig_width)

ax1 = plt.subplot2grid(shape=(1, 1), loc=(0, 0), colspan=1)

t_start_plot = 0
t_end = 300

left, bottom, width, height = [0.29, 0.632, 0.27, 0.43*golden_mean]
ax2 = fig.add_axes([left, bottom, width, height])

bplot = [0, 0.0001, 0.001, 0.01, 0.1, 1]
for b in np.sort(bplot):
    R_plot = R.loc[((R["status_flag"]).astype(float).astype(int) == 0) & (R["b"] == b) & (R["t"] >= t_start_plot) & (R["t"] <= t_end) & (((b != 0) & (R["random_seed_pos_init"] == seed_plot)) | ((b == 0) & (R["random_seed_pos_init"] == seed_plot)))]
    x1 = np.array(R_plot["t"])
    y1 = np.array(R_plot["<R>^2"])
    ax1.plot(x1, y1, color=colors[blist.index((b))])

    x2 = x1
    y2 = np.array(R_plot["sqrt(<R^2>)"])*np.sqrt((np.array(R_plot["N"])-1)/ np.array(R_plot["N"])) / np.array(R_plot["<R>"])
    ax2.plot(x2, y2, color=colors[blist.index((b))])

ax1.set_xlabel(r"$t$")

ax1.set_ylabel(r"${\langle a \rangle}^2$")
ax1.tick_params(which="both", axis='y', right=True, labelright=False, direction="in")
ax1.tick_params(which="both", axis='y', left=True, labelleft=True)

ax2.set_xlabel(r"$t$", labelpad=0.5, fontsize=7)
ax2.set_ylabel(r"$\sqrt{{(a - \langle a \rangle)}^2}/ \langle a \rangle$", labelpad=0.05, fontsize=6)
ax2.tick_params(axis='both', which='both', pad=0.5, labelsize=5)
ax2.tick_params(which="both", axis='y', right=True, labelright=False, direction="in")
ax2.tick_params(which="both", axis='y', left=True, labelleft=True)

# ax3.set_ylim(top=2.5)

fig.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0.25)

fig.savefig(output_foldername + graphs_foldername +
            "FIG_SI_a2.jpg", dpi=1000, bbox_inches='tight')
plt.close()

########################################################################################################################

# del dr2ds

system_foldername = "./"
N_foldername = "./"

type_foldername = "Damped_ripening/"

dr2dstau_filename = "dr2ds_configuration_path_analysis.txt"

PATH = join(data_foldername, data_type_foldername, system_foldername, N_foldername, type_foldername)

dr2ds = pd.read_csv(join(PATH, output_foldername, dr2dstau_filename), sep=r"\s+",
                                header=None, skiprows=0)

dr2ds.columns = ["dd2ds_ds", "dd2ds_dd2", "dd2ds_b", "dd2ds_ds_f", "dd2ds_dd2_f", "dd2ds_b_f", "dd2dt_dt", "dd2dt_dd2", "dd2dt_b", "dTau2udt_dt", "dTau2udt_dTau2u", "dTau2udt_b", "dd2dt_dt_f", "dd2dt_dd2_f", "dd2dt_b_f", "dTau2udt_dt_f", "dTau2udt_dTau2u_f", "dTau2udt_b_f", "dd2dt_dt_f_bin", "dd2dt_dd2_f_bin", "dd2dt_b_f_bin", "dTau2udt_dt_f_bin", "dTau2udt_dTau2u_f_bin", "dTau2udt_b_f_bin", "displacement2", "Tau2", "b_mean", "dR", "P_dR", "dR_b", "dx", "P_dx", "dx_b", "dTau2ds_ds_f", "dTau2ds_dTau2_f", "dTau2ds_b_f", "dszdt_dt_f", "dszdt_dsz_f", "b_dsz_dt_f"]

fig = plt.figure()
fig.set_figheight(fig_height)
fig.set_figwidth(fig_width)

ax2 = plt.subplot2grid(shape=(1, 1), loc=(0, 0), colspan=1)

bplot = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]
for b in np.sort(bplot):
    dr2ds_plot = dr2ds.loc[(dr2ds["b_dsz_dt_f"] == b)]
    x2, y2 = dr2ds_plot["dszdt_dt_f"], dr2ds_plot["dszdt_dsz_f"]
    if b != 0:
        ax2.plot(x2, y2, label=r"$\xi = {}$".format(b), color=colors[blist.index((b))])
    else:
        ax2.plot(x2, y2, label=r"Quasistatic", color=colors[blist.index((b))])
    ax2.scatter(x2, y2, s=4, color=colors[blist.index((b))])
    if b == 0:
        yp2 = y2[(x2 >= 1.1e1) & (x2 <= 1.1e2)]
        xp2 = x2[(x2 >= 1.1e1) & (x2 <= 1.1e2)]
       
p = np.polyfit(np.log10(xp2), np.log10(yp2), 1)
xi = 3e1
xf = 1e2
yi = 6.5e0

slope = 1.0#p[0]
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)
ax2.plot(x_vals, y_vals, color="black",linewidth=0.75)
# ax3.annotate(text='0.91', xy=(2.5e3, 4.3e3), xycoords='data', fontsize=7)
ax2.annotate(text='$' +  str(np.around(slope, 2)) +'$', xy=(4.5e1, 1.6e1), xycoords='data', fontsize=7)

ax2.legend(loc="best", frameon = False)

ax2.set_ylabel(r"$\langle {\Delta s} \rangle$", labelpad=0.5)
ax2.set_xlabel(r"$\tau$")

ax2.set_yscale("log")
ax2.set_xscale("log")

fig.tight_layout()

fig.savefig(output_foldername + graphs_foldername +
            "FIG_SI_dsdt.jpg", dpi=1000, bbox_inches='tight')
plt.close()

########################################################################################################################

system_foldername = "./"
N_foldername = "./"

type_foldername = "Damped_ripening/"

dr2dstau_a_filename = "dr2ds_a_i_configuration_path_analysis.txt"

PATH = join(data_foldername, data_type_foldername, system_foldername, N_foldername, type_foldername)

def parse_array(s: str) -> np.ndarray:
    return np.fromstring(s.strip('[]'), sep=' ')

# tell pandas which columns to run through our parser
converters = {
    'dt_a': lambda s: int(s.strip('[]')),
    'ddisplacement2dt': parse_array,
    'dTau2dt': parse_array,
    'b_a': lambda s: float(s.strip('[]')),
}

dr2ds_a = pd.read_csv(join(PATH, output_foldername, dr2dstau_a_filename), sep="\t", engine="python",
                                header=None, skiprows=1, dtype=str)
dr2ds_a.columns = ["dt_a", "ddisplacement2dt", "dTau2dt", "b_a"]
                   
# 2) strip off the [...] and convert scalars
dr2ds_a['dt_a'] = dr2ds_a['dt_a'].str.strip('[]').astype(int)
dr2ds_a['b_a']  = dr2ds_a['b_a'].str.strip('[]').astype(float)

# 3) parser for the vector columns
def parse_array(s: str) -> np.ndarray:
    return np.fromstring(s.strip('[]'), sep=' ')

# 4) turn those two columns into actual numpy arrays
dr2ds_a['ddisplacement2dt'] = dr2ds_a['ddisplacement2dt'].apply(parse_array)
dr2ds_a['dTau2dt']          = dr2ds_a['dTau2dt'].apply(parse_array)

fig = plt.figure()
fig.set_figheight(2*fig_height)
fig.set_figwidth(2*fig_width)

ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), colspan=1)
ax2 = plt.subplot2grid(shape=(2, 2), loc=(1, 0), colspan=1)
ax3 = plt.subplot2grid(shape=(2, 2), loc=(0, 1), colspan=1)
ax4 = plt.subplot2grid(shape=(2, 2), loc=(1, 1), colspan=1)

bplot = [0.0001, 0.1]
a_edges= np.array([[0.43582158, 0.9106076, 1.23856918, 6.10888579],
                   [0.39226251, 0.84794535, 1.16757403, 6.660942]]) # calculated over pool of all a_i for all seeds for given b
for b in np.sort(bplot):
    if b == 0.0001:
        ax = ax2
        cmap = plt.get_cmap("Greens")
    elif b == 0.1:
        ax = ax4
        cmap = plt.get_cmap("Purples")
    dr2ds_a_plot = dr2ds_a.loc[(dr2ds_a["b_a"] == b)]
    x, y = dr2ds_a_plot["dt_a"], np.vstack(dr2ds_a_plot["dTau2dt"].values)
    # cmap = plt.get_cmap("Greens")                # grab the colormap
    reds = [cmap(i) for i in np.linspace(0.2, 1, y.shape[1])]
    for i in range(y.shape[1]):
        if b != 0:
            ax.plot(x, y[:,i], label=r"$a_i \in [{:.2f}, {:.2f}]$".format(a_edges[bplot.index(b), i], a_edges[bplot.index(b), i+1]), color=reds[i])
        else:
            ax.plot(x, y[:,i], label=r"Quasistatic", color=reds[i])
        ax.scatter(x, y[:,i], s=5, color=reds[i])
    dr2ds_plot = dr2ds.loc[(dr2ds["dTau2udt_b_f"] == b)]
    x, y = dr2ds_plot["dTau2udt_dt_f"], dr2ds_plot["dTau2udt_dTau2u_f"]
    ax.plot(x, y, color=colors[blist.index((b))], zorder=-np.log10(b) if b != 0 else 0, label=r"$\xi = {}$ - overall average".format(b) if b != 0 else r"Quasistatic")
    ax.scatter(x, y, s=5, color=colors[blist.index((b))], zorder=-np.log10(b) if b != 0 else 0)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$")

ax2.set_ylabel(r"$\langle {\Delta \sigma}^2(\tau) \rangle$", labelpad=0.75)

for b in np.sort(bplot):
    if b == 0.0001:
        ax = ax1
        cmap = plt.get_cmap("Greens")
    elif b == 0.1:
        ax = ax3
        cmap = plt.get_cmap("Purples")
    dr2ds_a_plot = dr2ds_a.loc[(dr2ds_a["b_a"] == b)]
    x, y = dr2ds_a_plot["dt_a"], np.vstack(dr2ds_a_plot["ddisplacement2dt"].values)
    # cmap = plt.get_cmap("Greens")                # grab the colormap
    reds = [cmap(i) for i in np.linspace(0.2, 1, y.shape[1])]
    for i in range(y.shape[1]):
        ax.plot(x, y[:,i], color=reds[i], zorder=-np.log10(b) if b != 0 else 0, label=r"$a_i \in [{:.2f}, {:.2f}]$".format(a_edges[bplot.index(b), i], a_edges[bplot.index(b), i+1]))
        ax.scatter(x, y[:,i], s=5, color=reds[i], zorder=-np.log10(b) if b != 0 else 0)
    dr2ds_plot = dr2ds.loc[(dr2ds["dd2dt_b_f"] == b)]
    x, y = dr2ds_plot["dd2dt_dt_f"], dr2ds_plot["dd2dt_dd2_f"]
    ax.plot(x, y, color=colors[blist.index((b))], zorder=-np.log10(b) if b != 0 else 0, label=r"$\xi = {}$ - overall average".format(b) if b != 0 else r"Quasistatic")
    ax.scatter(x, y, s=5, color=colors[blist.index((b))], zorder=-np.log10(b) if b != 0 else 0)
    ax.legend(loc="best", frameon = False, ncols=2, columnspacing=1.4)


    ax.tick_params(axis='both', which='both') #, pad=0.5labelsize=4.3, 
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2,3,4,5,6,7,8,9], numticks=10))
    ax.set_xlabel(r"$\tau$") # , labelpad=-1fontsize=6, 
ax1.set_ylabel(r"$\langle {\Delta r}^2(\tau) \rangle$", labelpad=0.5)
ax1.set_ylim(bottom=4e-6)
ax3.set_ylim(bottom=4e-7)

# ax1.set_ylim(top=1e0)

fig.tight_layout()
plt.subplots_adjust(wspace=0.20, hspace=0.25)

fig.savefig(output_foldername + graphs_foldername +
            "FIG_SI_MSDs.jpg", dpi=1000, bbox_inches='tight')
plt.close()

########################################################################################################################

# SI figure plotting J* and G* for different b values and having beta v/s b as inset on bottom plot; legend on top one

system_foldername = "./"
N_foldername = "./"

type_foldername = "Damped_ripening/"

PATH = join(data_foldername, data_type_foldername, system_foldername, N_foldername, type_foldername)

G_filename = "G_jcc.csv"

G = pd.read_csv(join(PATH, output_foldername, G_filename), sep=';',
                                        header=None, skiprows=1)
G.columns = ["omega", "Gstar", "Gft", "Jft", "b", "beta"]

fig = plt.figure()
fig.set_figheight(2*fig_height)
fig.set_figwidth(fig_width)

ax1 = plt.subplot2grid(shape=(2, 1), loc=(0, 0), colspan=1)
ax2 = plt.subplot2grid(shape=(2, 1), loc=(1, 0), colspan=1)

bplot = [0, 0.0001, 0.001, 0.01, 0.1, 1.0]

for b in np.sort(bplot):
    G_plot = G.loc[(G["b"] == b)]
    if G_plot.size > 0:
        omega = G_plot["omega"].iloc[0]
        Gstar = G_plot["Gstar"].iloc[0]
        Jstar = G_plot["Jft"].iloc[0]
        omega = literal_eval(omega)
        Gstar = literal_eval(Gstar)
        Jstar = literal_eval(Jstar)
        ax1.plot(omega, Jstar, color=colors[blist.index((b))], ls="-", label=r"$\xi = {}$".format(b) if b != 0 else r"Quasistatic")
        ax1.scatter(omega, Jstar, color=colors[blist.index((b))], marker="o", s=5)
        ax2.plot(omega, Gstar, color=colors[blist.index((b))], ls="-")
        ax2.scatter(omega, Gstar, color=colors[blist.index((b))], marker="o", s=5)

ax1.set_xscale("log")
ax1.set_yscale("log")
# ax1.set_xlabel(r"$\omega$")
# magnitude of ~ on top J
ax1.set_ylabel(r"$|\widetilde{J}|$")
ax1.tick_params(axis='both', which='both')
ax1.legend(loc="best", frameon=False, ncol=2)

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$\omega$")
# magnitude of G*
ax2.set_ylabel(r"$|G^*|$")
ax2.tick_params(axis='both', which='both')
ax2.set_ylim(bottom=7e-4)

# inset for beta v/s b
left, bottom, width, height = [0.73, 0.1699, 0.21, 0.23*golden_mean]
ax3 = fig.add_axes([left, bottom, width, height])
left, bottom, width, height = [0.68, 0.1699, 0.04, 0.23*golden_mean]
ax3b = fig.add_axes([left, bottom, width, height], sharey=ax3)

ax3.spines.left.set_visible(False)
ax3b.spines.right.set_visible(False)

# t_relax = [0.0, np.nan, 10, 20, 120, np.nan]
for b in bplot:
    if b != 0:
        G_plot = G.loc[(G["b"] == b)]
        if G_plot.size > 0:
            alpha = G_plot["beta"].iloc[0]
            ax3.scatter(b, alpha, color=colors[blist.index((b))] , marker="o", s=3.6)

for b in bplot:
    if b == 0:
        G_plot = G.loc[(G["b"] == b)]
        if G_plot.size > 0:
            alpha = G_plot["beta"].iloc[0]
            ax3b.scatter(b, alpha, color=colors[blist.index((b))] , marker="o", s=3.6)

ax3.set_xlabel(r"$\xi$ \quad \quad", labelpad=-0.5, fontsize=7)
ax3b.set_ylabel(r"$\beta$", labelpad=0.5, fontsize=7)
ax3.tick_params(axis='both', which='both', pad=0.5, labelsize=5)
ax3.tick_params(which="both", axis='y', right=True, labelright=False, direction="in")
ax3.tick_params(which="both", axis='y', left=False, labelleft=False)
# ax3.set_xlim(left=5.5e-4, right=2e-1)
# ax3.set_ylim(top=0.22, bottom=0.101)
ax3.set_xscale("log")
# ax3.set_yscale("log")

ax3.set_xlim(left=4e-5, right=4e-0)
ax3b.set_xlim(left=-0.00013, right=0.00009)
ax3.set_ylim(top=0.32)

ax3b.tick_params(axis='both', which='both', pad=0.5, labelsize=5)
ax3b.tick_params(which="both", axis='y', right=False, labelright=False, direction="in")
ax3b.tick_params(which="both", axis='y', left=True, labelleft=True)
ax3b.tick_params(which="both", axis='x', bottom=True, labelbottom=True)
ax3b.set_xticks([0])
ax3b.set_xticklabels([0])

d = .99  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=4,
              linestyle="none", color='k', mec='k', mew=0.35, clip_on=False)
ax3.plot([0, 0], [0, 1], transform=ax3.transAxes, **kwargs)
ax3b.plot([1, 1], [0, 1], transform=ax3b.transAxes, **kwargs)

del G

fig.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0.25)

fig.savefig(output_foldername + graphs_foldername +
            "FIG_SI_rheology.jpg", dpi=1000, bbox_inches='tight')
plt.close()

########################################################################################################################

fig = plt.figure()
fig.set_figheight(2*fig_height)
fig.set_figwidth(fig_width)

ax2 = plt.subplot2grid(shape=(2, 1), loc=(0, 0), colspan=1)
ax1 = plt.subplot2grid(shape=(2, 1), loc=(1, 0), colspan=1)

system_foldername = "./"
N_foldername = "./"

type_foldername = "Damped_ripening/"

PATH = join(data_foldername, data_type_foldername, system_foldername, N_foldername, type_foldername)

dirs = [dirs for root, dirs, files in walk(join(".", PATH))]
dirs = np.array([dir for dir in dirs[0]])

dr2dstau_filename = "dr2ds_configuration_path_analysis.txt"

bplot = [0.0, 0.0001, 0.001, 0.01, 0.1,1]

for b in bplot:
    x1, y1 = np.array([]), np.array([])
    x2, y2 = np.array([]), np.array([])
    seed_array_b = np.array([])

    x10, y10 = np.array([]), np.array([])
    x20, y20 = np.array([]), np.array([])
    seed_array_b_10 = np.array([])


    for i, dir in np.ndenumerate(dirs):
        if isfile(join(PATH, dir, output_foldername, run_filename)) & isfile(join(PATH, dir, output_foldername, dr2dstau_filename)):
            f = pd.read_csv(join(PATH, dir, output_foldername, run_filename), lineterminator="\n",
                            header=None, skip_blank_lines=False)
            f = f[0].str.split(r"\s{2,}", expand=True)
            f.columns = ["value", "name"]

            bfile = float(f.loc[f["name"] == "b"]["value"].values[0])

            random_seed_pos_init = int(float(f.loc[f["name"] == "random_seed_pos_init"]["value"].values[0]))

            if (random_seed_pos_init <= 4) & (b == bfile):
                dr2ds_temp = pd.read_csv(join(PATH, dir, output_foldername, dr2dstau_filename), sep=r"\s+",
                                    header=None, usecols=[22,23,24] + [31, 32, 33], skiprows=1)

                dr2ds_temp.columns = ["dt_i", "ddisplacement2_i", "R"] + ["dt_d_i", "ddisplacement_i", "R_i"]

                dr2ds_temp["b"] = b

                dr2ds_temp["random_seed_pos_init"] = random_seed_pos_init

                dr2ds_temp["simkey"] = i[0]

                # print(b, random_seed_pos_init)
                df = dr2ds_temp.loc[np.around(dr2ds_temp["dt_d_i"]) == 1]
                dx = (df["ddisplacement_i"])
                x1 = np.append(x1, dx)

                df = dr2ds_temp.loc[np.around(dr2ds_temp["dt_d_i"]) == 10]
                dx10 = (df["ddisplacement_i"])
                x10 = np.append(x10, dx10)
                
                df = dr2ds_temp.loc[np.around(dr2ds_temp["dt_i"]) == 1]
                a = df["R"]
                dx2 = df["ddisplacement2_i"]
                dx2 = dx2*a
                y2 = np.append(y2, dx2)
                x2 = np.append(x2, a)
                # add random pos init same number of times as 'a' and 'dx2'
                seed_array_b = np.append(seed_array_b, np.ones(len(a))*random_seed_pos_init)

                df = dr2ds_temp.loc[np.around(dr2ds_temp["dt_i"]) == 10]
                a10 = df["R"]
                dx20 = df["ddisplacement2_i"]
                dx20 = dx20*a10
                y20 = np.append(y20, dx20)
                x20 = np.append(x20, a10)
                # add random pos init same number of times as 'a' and 'dx2'
                seed_array_b_10 = np.append(seed_array_b_10, np.ones(len(a10))*random_seed_pos_init)

    nbins = 50

    x1 = np.abs(x1)
    x1 = x1[x1 > 0]
    y1, bins = np.histogram(np.log10(x1), bins="auto")
    y1, x1 = np.histogram(x1, bins=10**bins, density=True)
    x1 = 10**((np.log10(x1[:-1])+np.log10(x1[1:]))/2)
    if (b == 0.0):
        yp1 = y1[(x1 >= 1e-2) & (x1 <= 8e-2)]
        xp1 = x1[(x1 >= 1e-2) & (x1 <= 8e-2)]
        p = np.polyfit(np.log10(xp1), np.log10(yp1), 1)

    _, bins = np.histogram(np.log10(x2), bins=nbins)
    bin_centers_valid = np.ones(len(bins)-1)
    y2_all = np.array([])
    for seed in np.unique(seed_array_b):
        y2_b = y2[seed_array_b == seed]
        x2_b = x2[seed_array_b == seed]

        count, _ = np.histogram((x2_b), bins=10**bins)
        bin_centers_valid *= (count >= 4)

        y2_b, x2_b, bnumbr = stats.binned_statistic(
        x2_b, y2_b, statistic=np.nanmean, bins=10**bins)

        if y2_all.shape[0] == 0:
            y2_all = np.array(y2_b)
        else:
            y2_all = np.vstack((y2_all, y2_b))

        x2_b = 10**((np.log10(x2_b[:-1])+np.log10(x2_b[1:]))/2)
    
    y2 = np.nanmean(y2_all, axis=0)
    y2 = y2[bin_centers_valid == 1]
    x2 = x2_b[bin_centers_valid == 1]
        
    ax2.scatter(x2, y2, color=colors[blist.index((b))], s=2.5)

    x10 = np.abs(x10)
    x10 = x10[x10 > 0]
    # x10 = x10[x10 > 5e-6]
    y10, bins = np.histogram(np.log10(x10), bins="auto")
    y10, x10 = np.histogram(x10, bins=10**bins, density=True)
    x10 = 10**((np.log10(x10[:-1])+np.log10(x10[1:]))/2)
    # ax1.plot(x10, y10, label=r"$\xi = $" + str(b) + ", $t = 10
    if (b == 0.0):
        yp10 = y10[(x10 >= 1e-2) & (x10 <= 8e-2)]
        xp10 = x10[(x10 >= 1e-2) & (x10 <= 8e-2)]
        p10 = np.polyfit(np.log10(xp10), np.log10(yp10), 1)

    _, bins = np.histogram(np.log10(x20), bins=nbins)
    bin_centers_valid = np.ones(len(bins)-1)
    y20_all = np.array([])
    for seed in np.unique(seed_array_b_10):
        y20_b = y20[seed_array_b_10 == seed]
        x20_b = x20[seed_array_b_10 == seed]

        count, _ = np.histogram((x20_b), bins=10**bins)
        bin_centers_valid *= (count >= 4)

        y20_b, x20_b, bnumbr = stats.binned_statistic(
        x20_b, y20_b, statistic=np.nanmean, bins=10**bins)
        # print(x2_b, y2_b)

        if y20_all.shape[0] == 0:
            y20_all = np.array(y20_b)
        else:
            y20_all = np.vstack((y20_all, y20_b))

        x20_b = 10**((np.log10(x20_b[:-1])+np.log10(x20_b[1:]))/2)

    y20 = np.nanmean(y20_all, axis=0)
    y20 = y20[bin_centers_valid == 1]
    x20 = x20_b[bin_centers_valid == 1]

    ax1.scatter(x20, y20, color=colors[blist.index((b))], s=2.5)
# fake plot to add legend
ax1.plot([], [], color="black", label=r"$\tau = 10$")
ax2.plot([], [], color="black", label=r"$\tau = 1$")

xi = 1.6e0
xf = 4e0
yi = 1.6e-3

slope = 0
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)

ax2.set_xlabel(r"$a/\bar{a}$")
ax2.set_ylabel(r"$\langle a \Delta {r}^2 / \bar{a}^3 \rangle$")
ax2.set_xscale("log")
ax2.set_yscale("log")
# ax2.set_ylim(top=3.6e-3)
# ax2.set_title(r"$\tau = 1$")
ax2.legend(loc="lower right", frameon = False)

ax1.set_xlabel(r"$a/\bar{a}$")
ax1.set_ylabel(r"$\langle a \Delta {r}^2 / \bar{a}^3 \rangle$")
ax1.set_xscale("log")
ax1.set_yscale("log")
# ax1.set_ylim(top=3.6e-3)
# ax1.set_title(r"$\tau = 10$")
ax1.legend(loc="lower right", frameon = False)

fig.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0.25)

fig.savefig(output_foldername + graphs_foldername +
            "FIG_SI_dr2a.jpg", dpi=1000, bbox_inches='tight')
plt.close()

########################################################################################################################

fig = plt.figure()
fig.set_figheight(fig_height)
fig.set_figwidth(fig_width)

ax1 = plt.subplot2grid(shape=(1, 1), loc=(0, 0), colspan=1)

system_foldername = "./"
N_foldername = "./"

type_foldername = "Damped_QR_RQ"

U_Z_filename = "s_contour_nFIRE_N_RP_RQ_hist.txt"

PATH = join(data_foldername, data_type_foldername, system_foldername, N_foldername, type_foldername)

s_FIRE = pd.read_csv(join(PATH, output_foldername, U_Z_filename), sep=r"\s+",
                                header=None, skiprows=1)
s_FIRE.columns = ["s_sq", "s_rp", "nFIRE_sq", "nFIRE_rp", "N_sq", "N_rp"]

s_rp = s_FIRE["s_rp"]
s_sq = s_FIRE["s_sq"]
N_rp = s_FIRE["N_rp"]
N_sq = s_FIRE["N_sq"]

N_rp = N_rp[s_rp > 0]
N_sq = N_sq[s_sq > 0]
s_rp = s_rp[s_rp > 0]
s_sq = s_sq[s_sq > 0]

s_rp = s_rp/N_rp
s_sq = s_sq/N_sq

# print("median s_rp, s_sq:", np.median(s_rp), np.median(s_sq))
# print("ratio of medians s_sq/s_rp:", np.median(s_sq)/np.median(s_rp))

_, bins_rp = np.histogram(np.log10(s_rp), bins="auto", density=True)
_, bins_sq = np.histogram(np.log10(s_sq), bins="auto", density=True)

ax1.hist(s_rp, bins=10**bins_rp, color="blue", alpha=0.5, label=r"Quasistatic", density=True)
ax1.hist(s_sq, bins=10**bins_sq, color="tab:blue", alpha=0.5, label=r"Random Quenched", density=True)

ax1.set_ylabel(r"$P(\Delta R/N)$")
ax1.set_xlabel(r"$\Delta R/N$")

ax1.legend(loc="upper right", frameon = False)
ax1.set_xscale("log")
ax1.set_yscale("log")

fig.tight_layout()

fig.savefig(output_foldername + graphs_foldername +
            "FIG_SI_delR.jpg", dpi=1000, bbox_inches='tight')
plt.close()

########################################################################################################################