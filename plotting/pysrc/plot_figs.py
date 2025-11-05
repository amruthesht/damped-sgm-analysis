'''
File: plot_figs.py
Project: meta_paper_plot
File Created: Monday, 20th December 2021 9:37:34 pm
Author: Amruthesh T (amru@seas.upenn.edu)
-----
Last Modified: Thursday, 17th August 2023 1:51:12 pm
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
from os.path import join
from cycler import cycler
import random
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import FixedFormatter
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import NullLocator
from mpl_toolkits import mplot3d
from sklearn.metrics import pairwise_distances
from statsmodels.graphics import tsaplots
from statsmodels.tsa.stattools import acf
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import numpy_indexed as npi

from brokenaxes import brokenaxes

import matplotlib.patches as patches

from statsmodels.graphics import tsaplots

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
# ["blue", "green", "red", "cyan", "saddlebrown", "magenta", "darkviolet", "gold"]
colors = ["blue", "green", "red", "cyan", "saddlebrown", "magenta", "darkviolet", "orange", "gold", "blue", "green", "red"]
tabcolors = ["tab:blue", "tab:green", "tab:red", "tab:cyan", "tab:purple", "tab:olive", "tab:blue", "tab:green", "tab:red"]
markers = ["s", "o", "^"]
s = [14.5, 15, 15.5]

blist = [0.0, 0.0001, 0.001, 0.01, 0.04, 0.1, 0.3, 0.5, 1]#, 0.04]

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

data_foldername = "Data/"
data_type_foldername = "./"

output_foldername = "output/"
graphs_foldername = "graphs/"
input_foldername = "input/"
input_filename = "init.input"
run_filename = "run_config.txt"

d = 3

########################################################################################################################

# Fig avalanche (a), (b) and (c)

seed_plot = 1

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

fig = plt.figure()
fig.set_figheight(2*fig_height)
fig.set_figwidth(2*fig_width)


gs0 = gridspec.GridSpec(1, 2)

gs00 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[1])
gs01 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0], hspace=0.245, wspace=0.11)
ax1 = plt.subplot(gs01[:1, :])
ax2 = plt.subplot(gs01[1:, :1])
ax4 = plt.subplot(gs00[:1, :])
ax5 = plt.subplot(gs00[1:2, :])
ax6 = plt.subplot(gs00[2:3, :])

gs_ax3 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs01[1:, 1:], wspace=0.15)
ax3b = plt.subplot(gs_ax3[0, 1:])
ax3 = plt.subplot(gs_ax3[0, :1], sharey=ax3b)
   
bplot = [0.0, 0.0001, 0.001, 0.01, 0.1, 1]
for b in np.sort(bplot):
    U_plot = U.loc[((U["status_flag"]).astype(float).astype(int) == 0) & (U["b"] == b) & (U["t"] >= 180) & (U["t"] <= 300) & (U["random_seed_pos_init"] == seed_plot)]
    x1 = np.array(U_plot["t"])[1:]
    y1 = np.array(U_plot["U"])
    if b != 0:
        y1 = (y1[:-1] - y1[1:])/ ((y1[1:] + y1[:-1])/ 2)#*b#*U_plot["dt_minimizer"].unique()[0]# np.array(U_plot["dU_U"])#/ U_plot["dt_minimizer"].unique()[0]#(y[:-1] - y[1:])/ ((y[1:] + y[:-1])/ 2) #U["dt_minimizer"].unique()[0]
    else:
        y1 = (y1[:-1] - y1[1:])/ ((y1[1:] + y1[:-1])/ 2)#*1e-5#np.array(U_plot["dU_U"])#*1e-6
    if b != 0:
        ax1.plot(x1, y1, label=r"$\xi = {}$".format(b), color=colors[blist.index((b))], zorder=np.log10(b))
    else:
        ax1.plot(x1, y1, label=r"Quasistatic", color=colors[blist.index((b))], zorder=10)

    y2 = np.array([])
    for seed in U["random_seed_pos_init"].unique():
        U_plot = U.loc[((U["status_flag"]).astype(float).astype(int) == 0) & (U["b"] == b) & (U["t"] >= 180) & (U["t"] <= 1000) & (U["random_seed_pos_init"] != -2) & (U["random_seed_pos_init"] == seed)]
        y = np.array(U_plot["U"])
        y = (y[:-1] - y[1:])/((y[1:] + y[:-1])/ 2)# np.array(U_plot["dU"])/ U_plot["dt_minimizer"].unique()[0]#(y[:-1] - y[1:])
        # y2 = y2[(y2 > 1e-6) & (y2 < 1e-3)]
        y2 = np.append(y2, y)
    y2 = y2[(y2 > 1e-3) & (y2 < 1e0)]
    _, bins = np.histogram(np.log10(y2), bins="auto")
    y2, x2 = np.histogram(y2, bins=10**bins, density=True)
    x2 = 10**((np.log10(x2[1:]) + np.log10(x2[:-1]))/ 2)

    x2 = x2[y2>0]
    y2 = y2[y2>0]

    y2 = np.append(y2, 0)
    x2 = np.append(x2, np.max(x2))
    # if b not in [0.1, 1]:
    if b != 0:
        ax2.plot(x2, y2, label=r"$\xi = {}$".format(b), color=colors[blist.index((b))], zorder=np.log10(b))
    else:
        ax2.plot(x2, y2, label=r"Quasistatic", color=colors[blist.index((b))], zorder=10)

    if b == 0:
        yp2 = y2[(x2 >= 1e-2) & (x2 <= 6e-2)]
        xp2 = x2[(x2 >= 1e-2) & (x2 <= 6e-2)]

    R_sq_0 = 0.49 
    if b != 0:
        x3 = b
    else:
        x3 = 1e-10
    y3_array = np.array([])
    # U_0_array = np.array([])
    if b == 0:
        seed_array = np.arange(1,51)
    elif b == 1:
        seed_array = np.arange(1,6)
    else:
        seed_array = np.arange(1,5)
    if b == 0:
        dynamic_scaling_t = 180
    elif b == 0.0001:
        dynamic_scaling_t = 180
    elif b == 0.001:
        dynamic_scaling_t = 180
    elif b == 0.01:
        dynamic_scaling_t = 180
    elif b == 0.1:
        dynamic_scaling_t = 180
    elif b == 1:
        dynamic_scaling_t = 180
    for seed in seed_array:#R["random_seed_pos_init"].unique():
        R_plot = R.loc[(R["random_seed_pos_init"] == seed) & (R["b"] == b) & ((R["status_flag"]).astype(float).astype(int) == 0)]
        t_sq_0 = np.min(R_plot.loc[(R_plot["<R>^2"] >= R_sq_0)]["t"])
        if t_sq_0 > dynamic_scaling_t:
            U_plot = U.loc[(U["random_seed_pos_init"] == seed) & (U["b"] == b) & ((U["status_flag"]).astype(float).astype(int) == 0)]
            if t_sq_0 > 0:
                y3_array = np.append(y3_array, U_plot.loc[(U_plot["t"] == t_sq_0)]["U"].values[0]/ U_plot.loc[(U_plot["t"] == t_sq_0)]["N"].values[0])
    y3 = np.nanmean(y3_array)
    y31 = np.nanstd(y3_array)
    if b != 0:
        ax3b.errorbar(x3, y3, yerr=y31, color=colors[blist.index((b))], fmt="o", markersize=3.5)
    else:
        ax3.errorbar(x3, y3, yerr=y31, color=colors[blist.index((b))], fmt="o", markersize=3.5)

ax1.legend(loc="upper left", frameon = False, ncol=3, fontsize=5)

ax1.set_ylabel(r"$\Delta U (\Delta t = 1) / U(t) $")
ax1.set_xlabel(r"$t$")

ax1.set_yscale("log")

ax1.set_ylim(top=1.45e-1)

ax2.set_ylabel(r"$P(\Delta U (\Delta t = 1) / U(t))$")
ax2.set_xlabel(r"$\Delta U (\Delta t = 1) / U(t)$")

ax2.set_xscale("log")
ax2.set_yscale("log")

ax2.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10, subs=[2,3,4,5,6,7,8,9], numticks=10))

p = np.polyfit(np.log10(xp2), np.log10(yp2), 1)

xi = 3e-2
xf = 1e-1
yi = 5e0

slope = p[0]
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)
ax2.plot(x_vals, y_vals, color="black",linewidth=0.75)
ax2.annotate(text='$' +  str(np.around(p[0], 1)) +'$', xy=(5e-2, 1.8e0), xycoords='data', fontsize=6.5)

ax3b.set_xlabel(r"$\xi$")

ax3b.set_yscale("log")
ax3b.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xscale("log")

ax3b.set_xlim(right=3e0)

ax3.spines['right'].set_visible(False)
ax3b.spines['left'].set_visible(False)
ax3b.yaxis.tick_right()
ax3b.tick_params(which="both", axis='y', right=True, labelright=True)
ax3b.tick_params(which="both", axis='y', left=False, labelleft=False, direction="in")
ax3b.yaxis.set_label_position("right")
#push the label inside by changing pading
ax3b.set_ylabel(r"$U/N$", labelpad=-10)
ax3.tick_params(which="both", axis='y', right=False, labelright=False)
ax3.tick_params(which="both", axis='y', left=False, labelleft=False)

d = .023  # how big to make the diagonal lines in axes coordinates
kn = 4.23
kwargs = dict(transform=ax3.transAxes, color='k', clip_on=False, linewidth=0.5)
ax3.plot((1 - kn*d, 1 + kn*d), (-d, +d), **kwargs)  # bottom-right diagonal
ax3.plot((1 - kn*d, 1 + kn*d), (1 - d, 1 + d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax3b.transAxes)  # switch to the right axes
ax3b.plot((-d, +d), (-d, +d), **kwargs)  # bottom-left diagonal
ax3b.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # top-left diagonal

ax3bticks = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
ax3btick_labels = ["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$"]

ax3b.set_xticks(ax3bticks)
ax3b.set_xticklabels(ax3btick_labels, fontsize=5.5)

ax3ticks = [1e-10]
ax3tick_labels = ["$0$"]

ax3.set_xticks(ax3ticks)
ax3.set_xticklabels(ax3tick_labels, fontsize=5.5)
#remove minor ticks on ax3
ax3.xaxis.set_minor_locator(NullLocator())


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

bplot = [0.0, 0.0001, 0.001, 0.01]

seed_plot = 1

for b in np.sort(bplot):
    U_plot_IS = U_IS.loc[((U_IS["status_flag"]).astype(float).astype(int) == 0) & (U_IS["b"] == b) & (U_IS["t"] >= 180) & (U_IS["t"] <= 480) & (U_IS["random_seed_pos_init"] == seed_plot)]
    U_plot_rp = U_IS.loc[((U_IS["status_flag"]).astype(float).astype(int) == 2) & (U_IS["b"] == b) & (U_IS["t"] >= 180) & (U_IS["t"] <= 480) & (U_IS["random_seed_pos_init"] == seed_plot)]

    if b == 0.01:
        x4, y4 = np.array(U_plot_rp["t"]), np.array(U_plot_rp["U"])
        x41, y41 = np.array(U_plot_IS["t"]), np.array(U_plot_IS["U"])
        ax4.plot(x4, y4, color=colors[blist.index((b))], zorder=-1, label=r"$\xi = {}$".format(b))
        ax4.plot(x41, y41, color="black", label=r"$U_{IS}$", alpha=0.85, linewidth=0.85)
    elif b == 0.001:
        x5, y5 = np.array(U_plot_rp["t"]), np.array(U_plot_rp["U"])
        x51, y51 = np.array(U_plot_IS["t"]), np.array(U_plot_IS["U"])
        ax5.plot(x5, y5, label=r"$\xi = {}$".format(b), color=colors[blist.index((b))], zorder=-1)
        ax5.plot(x51, y51, label=r"$U_{IS}$", color="black", alpha=0.85, linewidth=0.85)
    elif b == 0.0001:
        x6, y6 = np.array(U_plot_rp["t"]), np.array(U_plot_rp["U"])
        x61, y61 = np.array(U_plot_IS["t"]), np.array(U_plot_IS["U"])
        ax6.plot(x6, y6, label=r"$\xi = {}$".format(b), color=colors[blist.index((b))], zorder=-1)
        ax6.plot(x61, y61, label=r"$U_{IS}$", color="black", alpha=0.85, linewidth=0.85)

# labels only for the last plot
ax4.legend(loc="best", frameon = False)
ax5.legend(loc="best", frameon = False)
ax6.legend(loc="best", frameon = False)
ax6.set_xlabel(r"$t$")

# remove x labels but not ticks for all but the last plot
ax4.tick_params(which="both", axis='x', bottom=True, top=False, labelbottom=False, direction="in")
ax5.tick_params(which="both", axis='x', bottom=True, top=False, labelbottom=False, direction="in")
ax6.tick_params(which="both", axis='x', bottom=True, top=False, labelbottom=True, direction="inout")
# ax7.tick_params(which="both", axis='x', bottom=True, top=False, labelbottom=True, direction="in")
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

ax4.tick_params(which="both", axis='y', left=True, labelleft=False, direction="in")
ax5.tick_params(which="both", axis='y', left=True, labelleft=False, direction="in")
ax6.tick_params(which="both", axis='y', left=True, labelleft=False, direction="in")

ax4r = ax4.secondary_yaxis("right")
ax5r = ax5.secondary_yaxis("right")
ax6r = ax6.secondary_yaxis("right")

ax4r.tick_params(which="both", axis='y', left=False, labelleft=False, right=True, labelright=True, direction="out")
ax5r.tick_params(which="both", axis='y', left=False, labelleft=False, right=True, labelright=True, direction="out")
ax6r.tick_params(which="both", axis='y', left=False, labelleft=False, right=True, labelright=True, direction="out")

ax5r.set_ylabel(r"$U$")

fig.tight_layout()

fig.savefig(output_foldername + graphs_foldername +
            "FIG_avalanches.jpg", dpi=1000, bbox_inches='tight')
plt.close()

########################################################################################################################

# Fig MSD

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
fig.set_figheight(2*fig_height)
fig.set_figwidth(fig_width)

ax1 = plt.subplot2grid(shape=(2, 1), loc=(0, 0), colspan=1)
ax2 = plt.subplot2grid(shape=(2, 1), loc=(1, 0), colspan=1)

bplot = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]
for b in np.sort(bplot):
    dr2ds_plot = dr2ds.loc[(dr2ds["dTau2udt_b_f"] == b)]
    x, y = dr2ds_plot["dTau2udt_dt_f"], dr2ds_plot["dTau2udt_dTau2u_f"]
    if b != 0:
        ax2.plot(x, y, label=r"$\xi = {}$".format(b), color=colors[blist.index((b))])
    else:
        ax2.plot(x, y, label=r"Quasistatic", color=colors[blist.index((b))])
    ax2.scatter(x, y, s=5, color=colors[blist.index((b))])
    if b == 0.0:
        # yp1 = y[(x >= 1.3e1) & (x <= 7e1)]
        # xp1 = x[(x >= 1.3e1) & (x <= 7e1)]
        yp1 = y[(x >= 2e1) & (x <= 7e1)]
        xp1 = x[(x >= 2e1) & (x <= 7e1)]
        xp2 = x[(x >= 2e0) & (x <= 1e1)]
        yp2 = y[(x >= 2e0) & (x <= 1e1)]
    elif b == 0.1:
        yp3 = y[(x >= 2e0) & (x <= 8e0)]
        xp3 = x[(x >= 2e0) & (x <= 8e0)]
    elif b == 1.0:
        yp4 = y[(x >= 3e0) & (x <= 9e0)]
        xp4 = x[(x >= 3e0) & (x <= 9e0)]
    if b == 0.0:
        xs, yr2 = np.min(x), np.min(y)
        xs2 = np.max(x)

    # dr2ds_plot = dr2ds_IS.loc[(dr2ds_IS["dd2ds_b"] == b)]
    # if dr2ds_plot.shape[0] > 0:
    #     x, y = dr2ds_plot["dd2ds_ds"], dr2ds_plot["dd2ds_dd2"]
    #     # x, y = x[x > 1e-2], y[x > 1e-2]
    #     ax1.plot(x, y, color=colors[blist.index((b))], ls="--")
    #     ax1.scatter(x, y, s=5, color=colors[blist.index((b))])

# yr2 = xs**2
slope = 1
# ax2.plot([xs, xs2], [yr2, yr2*(xs2/xs)**slope], color="grey", linewidth=0.6, linestyle="--")

p = np.polyfit(np.log10(xp1), np.log10(yp1), 1)
xi = 2.51e1
xf = 7.9e1
yi = 2.72e-6
slope = p[0]
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)

ax2.plot(x_vals, y_vals, color="black",linewidth=0.75)
ax2.annotate(text=r"$" + str(np.around(slope, 2)) + "$", xy=(4.6e1, 2.28e-6), xycoords='data', fontsize=7)#, fontsize=5

p = np.polyfit(np.log10(xp2), np.log10(yp2), 1)
xi = 2e0
xf = 6e0
yi = 3e-7
slope = p[0]
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)

# ax1.plot(x_vals, y_vals, color="black",linewidth=0.75)
# ax1.annotate(text=r"$" + str(np.around(slope, 2)) + "$", xy=(2.3e0, 8e-7), xycoords='data', fontsize=7)#, fontsize=5

p = np.polyfit(np.log10(xp3), np.log10(yp3), 1)
xi = 2e0
xf = 6e0
yi = 1e-8
slope = p[0]
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)

# ax1.plot(x_vals, y_vals, color="black",linewidth=0.75)
# ax1.annotate(text=r"$" + str(np.around(slope, 2)) + "$", xy=(3.2e0, 1.3e-8), xycoords='data', fontsize=7)#, fontsize=5

p = np.polyfit(np.log10(xp4), np.log10(yp4), 1)
xi = 5.2e0
xf = 1.4e1
yi = 2.5e-6
slope = 2.0#p[0]
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)

ax2.plot(x_vals, y_vals, color="black",linewidth=0.75)
ax2.annotate(text=r"$" + str(np.around(slope, 2)) + "$", xy=(5.5e0, 8e-6), xycoords='data', fontsize=7)#, fontsize=5

# ax1.legend(loc="best", frameon = False, ncols=2)

ax2.set_ylabel(r"$\langle {\Delta \sigma}^2(\tau) \rangle$", labelpad=0.75)
ax2.set_xlabel(r"$\tau$")

ax2.set_yscale("log")
ax2.set_xscale("log")

bplot = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]
for b in np.sort(bplot):
    dr2ds_plot = dr2ds.loc[(dr2ds["dd2dt_b_f"] == b)]
    x, y = dr2ds_plot["dd2dt_dt_f"], dr2ds_plot["dd2dt_dd2_f"]
    # y = y[x <= 2e2]
    # x = x[x <= 2e2]
    ax1.plot(x, y, color=colors[blist.index((b))], zorder=-np.log10(b) if b != 0 else 0, label=r"$\xi = {}$".format(b) if b != 0 else r"Quasistatic")
    ax1.scatter(x, y, s=5, color=colors[blist.index((b))], zorder=-np.log10(b) if b != 0 else 0)
    if b == 0:
        yp1 = y[(x >= 2e1) & (x <= 7e1)]
        xp1 = x[(x >= 2e1) & (x <= 7e1)]
        # yp1 = y[(x >= 1.1e1) & (x <= 1.1e2)]
        # xp1 = x[(x >= 1.1e1) & (x <= 1.1e2)]
    elif b == 0.1:
        yp2 = y[(x >= 3e0) & (x <= 1.5e1)]
        xp2 = x[(x >= 3e0) & (x <= 1.5e1)]

    # dr2ds_plot = dr2ds_IS.loc[(dr2ds_IS["dd2dt_b_f"] == b)]
    # if dr2ds_plot.shape[0] > 0:
    #     x, y = dr2ds_plot["dd2dt_dt_f"], dr2ds_plot["dd2dt_dd2_f"]
    #     # x, y = x[x > 1e-2], y[x > 1e-2]
    #     ax1.plot(x, y, color=colors[blist.index((b))], ls="--")
    #     ax1.scatter(x, y, s=5, color=colors[blist.index((b))])
ax1.legend(loc="best", frameon = False, ncols=2)

p = np.polyfit(np.log10(xp1), np.log10(yp1), 1)
xi = 2.6e1
xf = 2.5e2
yi = 3.8e-2
slope = p[0]
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)
# slope=p[0]
ax1.plot(x_vals, y_vals, color="black",linewidth=0.75)
ax1.annotate(text=r"$" + str(np.around(slope, 2)) + "$", xy=(4e1, 2e-1), xycoords='data', fontsize=7)#, fontsize=5

p = np.polyfit(np.log10(xp2), np.log10(yp2), 1)
xi = 3e0
xf = 1e1
yi = 7e-6

slope = 2.0 #p[0]
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)
ax1.plot(x_vals, y_vals, color="black",linewidth=0.75)
ax1.annotate(text='$' +  str(np.around(slope, 2)) +'$', xy=(5.5e0, 7.5e-6), xycoords='data', fontsize=7)

ax1.set_xlabel(r"$\tau$") # , labelpad=-1fontsize=6, 
ax1.set_ylabel(r"$\langle {\Delta r}^2(\tau) \rangle$", labelpad=0.5) #, labelpad=0.5fontsize=6,
ax1.tick_params(axis='both', which='both') #, pad=0.5labelsize=4.3, 
ax1.set_xscale("log")
ax1.set_yscale("log")

ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=[2,3,4,5,6,7,8,9], numticks=10))

ax1.set_ylim(top=1e0)

system_foldername = "./"
N_foldername = "./"

type_foldername = "Damped_ripening/"

PATH = join(data_foldername, data_type_foldername, system_foldername, N_foldername, type_foldername)

G_filename = "G_jcc.csv"

### plot to plot G* v/s omega as inset

G = pd.read_csv(join(PATH, output_foldername, G_filename), sep=';',
                                        header=None, skiprows=1)
G.columns = ["omega", "Gstar", "Gft", "Jft", "b", "beta"]

left, bottom, width, height = [0.73, 0.1699, 0.21, 0.23*golden_mean]
ax3 = fig.add_axes([left, bottom, width, height])

for b in bplot:
    G_plot = G.loc[(G["b"] == b)]
    if G_plot.size > 0:
        omega = G_plot["omega"].iloc[0]
        Gstar = G_plot["Gstar"].iloc[0]
        omega = literal_eval(omega)
        Gstar = literal_eval(Gstar)
        ax3.plot(omega, Gstar, color=colors[blist.index((b))], ls="-")
        ax3.scatter(omega, Gstar, color=colors[blist.index((b))], marker="o", s=2)

ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel(r"$\omega$", labelpad=0.5, fontsize=7)
# magnitude of G*
ax3.set_ylabel(r"$|G^*|$", labelpad=0.5, fontsize=7)
ax3.tick_params(axis='both', which='both', pad=0.5, labelsize=5)
ax3.tick_params(which="both", axis='y', left=True, labelleft=True)

fig.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0.25)

fig.savefig(output_foldername + graphs_foldername +
            "FIG_MSDs.jpg", dpi=1000, bbox_inches='tight')
plt.close()

########################################################################################################################

# Fig stress MSD

fig = plt.figure()
fig.set_figheight(2*fig_height)
fig.set_figwidth(fig_width)

ax4 = plt.subplot2grid(shape=(2, 1), loc=(0, 0), colspan=1)
ax1 = plt.subplot2grid(shape=(2, 1), loc=(1, 0), colspan=1)

bplot = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]
for b in np.sort(bplot):
    dr2ds_plot = dr2ds.loc[(dr2ds["dTau2ds_b_f"] == b)]
    x3, y3 = dr2ds_plot["dTau2ds_ds_f"], dr2ds_plot["dTau2ds_dTau2_f"]
    if b != 0:
        ax1.plot(x3, y3, label=r"$\xi = {}$".format(b), color=colors[blist.index((b))])
        ax1.scatter(x3, y3, s=5, color=colors[blist.index((b))])
    else:
        ax1.plot(x3, y3, label=r"Quasistatic", color=colors[blist.index((b))])
        ax1.scatter(x3, y3, s=5, color=colors[blist.index((b))])
    if b == 0.0001:
        xp31 = x3[(x3 >= 7.5e0) & (x3 <= 2e1)]
        yp31 = y3[(x3 >= 7.5e0) & (x3 <= 2e1)]

ax1.set_ylabel(r"$\langle {\Delta \sigma}^2 \rangle$", labelpad=0.5)
ax1.set_xlabel(r"$\langle {\Delta s} \rangle$", labelpad=0.5)
ax1.tick_params(axis='both', which='both', pad=0.5)
ax1.tick_params(which="both", axis='y', right=True, labelright=False, direction="in")
ax1.tick_params(which="both", axis='y', left=True, labelleft=True)
ax1.set_xscale("log")
ax1.set_yscale("log")

p = np.polyfit(np.log10(xp31), np.log10(yp31), 1)
xi = 3.5e0
xf = 1.1e1
yi = 3.8e-4
slope = 0.9 #p[0]
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)

ax1.plot(x_vals, y_vals, color="black",linewidth=0.75)
ax1.annotate(text='$' +  str(np.around(slope, 2)) +'$', xy=(5.4e0, 2.48e-4), xycoords='data', fontsize=7)

bplot = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]
for b in np.sort(bplot):
    dr2ds_plot = dr2ds.loc[(dr2ds["dd2ds_b_f"] == b)]
    x2, y2 = dr2ds_plot["dd2ds_ds_f"], dr2ds_plot["dd2ds_dd2_f"]
    if b != 0:
        ax4.plot(x2, y2, label=r"$\xi = {}$".format(b), color=colors[blist.index((b))])
    else:
        ax4.plot(x2, y2, label=r"Quasistatic", color=colors[blist.index((b))])
    ax4.scatter(x2, y2, s=4, color=colors[blist.index((b))])
    if b == 0.1:
        yp1 = y2[(x2 >= 5e-2) & (x2 <= 5e-1)]
        xp1 = x2[(x2 >= 5e-2) & (x2 <= 5e-1)]
    elif b == 0:
        yp2 = y2[(x2 >= 8e0) & (x2 <= 2.5e1)]
        xp2 = x2[(x2 >= 8e0) & (x2 <= 2.5e1)]

    if b == 0.1:
        xs, yr2 = np.min(x2), np.min(y2)
    elif b == 0.01:
        xs2 = np.max(x2)
       
yr2 = xs**2
ax4.plot([xs, xs2], [yr2, yr2*(xs2/xs)**2], color="grey", linewidth=0.6, linestyle="--")

p = np.polyfit(np.log10(xp1), np.log10(yp1), 1)
xi = 5e-2
xf = 3.2e-1
yi = 1.2e-2

slope = 2.0 #p[0]
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)
ax4.plot(x_vals, y_vals, color="black",linewidth=0.75)
ax4.annotate(text='$' +  str(slope) +'$', xy=(8.5e-2, 1.19e-1), xycoords='data', fontsize=7)

p = np.polyfit(np.log10(xp2), np.log10(yp2), 1)
xi = 3.6e0
xf = 3.5e1
yi = 1.15e0

slope = 1.37 #p[0]
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)
ax4.plot(x_vals, y_vals, color="black",linewidth=0.75)
ax4.annotate(text='$' +  str(np.around(slope, 2)) +'$', xy=(1.4e1, 2.5e0), xycoords='data', fontsize=7)

ax4.legend(loc="best", frameon = False)

ax4.set_ylabel(r"$\langle {\Delta R}^2 \rangle$", labelpad=0.5)
ax4.set_xlabel(r"$\langle \Delta s \rangle$")

ax4.set_yscale("log")
ax4.set_xscale("log")

fig.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0.25)

fig.savefig(output_foldername + graphs_foldername +
            "FIG_stressMSDs.jpg", dpi=1000, bbox_inches='tight')
plt.close()

########################################################################################################################

# Fig sq rheology

del dr2ds

system_foldername = "./"
N_foldername = "./"

type_foldername = "random_quenches/FIRE_relaxations/"

dr2ds_tau_sq_filename = "dr2ds_RQ_configuration_path_analysis.txt"

PATH = join(data_foldername, data_type_foldername, system_foldername, N_foldername, type_foldername)

dr2ds = pd.read_csv(join(PATH, output_foldername, dr2ds_tau_sq_filename), sep=r"\s+",
                                header=None)

dr2ds.columns = ["dd2ds_ds", "dd2ds_dd2", "dd2ds_b", "dd2ds_ds_f", "dd2ds_dd2_f", "dd2ds_b_f", "dd2dt_dt", "dd2dt_dd2", "dd2dt_b", "dTau2udt_dt", "dTau2udt_dTau2u", "dTau2udt_b", "dd2dt_dt_f", "dd2dt_dd2_f", "dd2dt_b_f", "dTau2udt_dt_f", "dTau2udt_dTau2u_f", "dTau2udt_b_f", "dd2dt_dt_f_bin", "dd2dt_dd2_f_bin", "dd2dt_b_f_bin", "dTau2udt_dt_f_bin", "dTau2udt_dTau2u_f_bin", "dTau2udt_b_f_bin", "displacement2", "Tau2", "b_mean", "dR", "P_dR", "dR_b", "dx", "P_dx", "dx_b", "dTau2ds_ds_f", "dTau2ds_dTau2_f", "dTau2ds_b_f"]

fig = plt.figure()
fig.set_figheight(2*fig_height)
fig.set_figwidth(fig_width)

nfig = 2
ax1 = plt.subplot2grid(shape=(nfig, 2), loc=(0, 0), colspan=2)
ax2 = plt.subplot2grid(shape=(nfig, 2), loc=(1, 0), colspan=1)
ax4 = plt.subplot2grid(shape=(nfig, 2), loc=(1, 1), colspan=1)

# bplot = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]

bplot = [0.0]
b_0 = 0.0
for b in np.sort(bplot):
    dr2ds_plot = dr2ds.loc[(dr2ds["dd2ds_b_f"] == b)]
    x, y = dr2ds_plot["dd2ds_ds_f"], dr2ds_plot["dd2ds_dd2_f"]
    y = y[(x >= 3e-2)]
    x = x[(x >= 3e-2)]
    ax1.plot(x, y, color="blue", zorder=-np.log10(b) if b != 0 else 0, label=r"$\xi = {}$".format(b) if b != 0 else r"Quasistatic")
    ax1.scatter(x, y, s=5, color="blue", zorder=-np.log10(b) if b != 0 else 0)
    if b == b_0:
        yp1 = y[(x >= 2.63e0) & (x <= 2.63e1)]
        xp1 = x[(x >= 2.63e0) & (x <= 2.63e1)]
    
    yp2 = y[(x >= 7e-2) & (x <= 2e-1)]
    xp2 = x[(x >= 7e-2) & (x <= 2e-1)]

p = np.polyfit(np.log10(xp1), np.log10(yp1), 1)
xi = 3.2e0
xf = 3.2e1
yi = 1.3e0
slope = 1.37#p[0]
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)

ax1.plot(x_vals, y_vals, color="black",linewidth=0.75)
ax1.annotate(text=r"$" + str(np.around(slope, 2)) + "$", xy=(9e0, 1.3e0), xycoords='data', fontsize=7)#, fontsize=5

p = np.polyfit(np.log10(xp2), np.log10(yp2), 1)
xi = 4e-2
xf = 2e-1
yi = 3.2e-3

slope = 2.0 #p[0]
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)
ax1.plot(x_vals, y_vals, color="black",linewidth=0.75)
ax1.annotate(text='$' +  str(np.around(slope, 2)) +'$', xy=(6e-2, 4e-2), xycoords='data', fontsize=7)

ax1.set_xlabel(r"$\langle \Delta s \rangle$") # , labelpad=-1fontsize=6, 
ax1.set_ylabel(r"$\langle {\Delta R}^2 \rangle$")#, labelpad=0.5) #, labelpad=0.5fontsize=6,

ax1.set_xscale("log")
ax1.set_yscale("log")

ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=[2,3,4,5,6,7,8,9], numticks=10))

left, bottom, width, height = [0.56, 0.649, 0.29, 0.225*golden_mean]
ax3 = fig.add_axes([left, bottom, width, height])

b_0 = 0.0
for b in np.sort(bplot):
    dr2ds_plot = dr2ds.loc[(dr2ds["dTau2ds_b_f"] == b)]
    x3, y3 = dr2ds_plot["dTau2ds_ds_f"], dr2ds_plot["dTau2ds_dTau2_f"]
    y3 = y3[(x3 >= 3e-2)]
    x3 = x3[(x3 >= 3e-2)]
    if b != b_0:
        ax3.plot(x3, y3, label=r"$\xi = {}$".format(b), color=colors[blist.index((b))])
    else:
        ax3.plot(x3, y3, label=r"Quasistatic", color="tab:blue")
        ax3.scatter(x3, y3, s=1.0, color="tab:blue")
    if b == b_0:
        yp1 = y[(x >= 2.63e0) & (x <= 2.63e1)]
        xp1 = x[(x >= 2.63e0) & (x <= 2.63e1)]

ax3.set_ylabel(r"$\langle {\Delta \sigma}^2 \rangle$", labelpad=0.5, fontsize=7)
ax3.set_xlabel(r"$\langle {\Delta s} \rangle$", labelpad=0.5, fontsize=7)
ax3.tick_params(axis='both', which='both', pad=0.5, labelsize=5)
ax3.tick_params(which="both", axis='y', right=False, labelright=False, direction="in")
ax3.tick_params(which="both", axis='y', left=True, labelleft=True)
ax3.set_xscale("log")
ax3.set_yscale("log")

ax3.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
ax3.xaxis.set_minor_locator(LogLocator(base=10, subs=[2,3,4,5,6,7,8,9], numticks=10))

ax3.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
ax3.yaxis.set_minor_locator(LogLocator(base=10, subs=[2,3,4,5,6,7,8,9], numticks=10))

p = np.polyfit(np.log10(xp31), np.log10(yp31), 1)
xi = 3e0
xf = 2e1
yi = 5.7e-4
slope = 0.9#p[0]
x_vals = np.array([xi, xf])
c = np.log10(yi) - slope * np.log10(xi)
c = 10**c
y_vals = c * np.power(x_vals, slope)

ax3.plot(x_vals, y_vals, color="black",linewidth=0.75)
ax3.annotate(text='$' +  str(np.around(slope, 2)) +'$', xy=(7e0, 2.5e-4), xycoords='data', fontsize=5)

system_foldername = "./"
N_foldername = "./"

type_foldername = "Damped_QR_RQ"

U_Z_filename = "U_Z_hist.txt"

PATH = join(data_foldername, data_type_foldername, system_foldername, N_foldername, type_foldername)

U_Z = pd.read_csv(join(PATH, output_foldername, U_Z_filename), sep=r"\s+",
                                header=None, skiprows=1)
U_Z.columns = ["U_sq", "U_rp", "Z_sq", "Z_rp"]

ax2.hist(U_Z["U_rp"], bins="auto", color="blue", alpha=0.5, label=r"Quasistatic", density=True)
ax2.hist(U_Z["U_sq"], bins="auto", color="tab:blue", alpha=0.5, label=r"Random Quenched", density=True)

ax4.hist(U_Z["Z_rp"], bins="auto", color="orange", alpha=0.5, label=r"Quasistatic", density=True)
ax4.hist(U_Z["Z_sq"], bins="auto", color="tab:orange", alpha=0.5, label=r"Random Quenched", density=True)

ax2.set_ylabel(r"$P(U/N)$") #, labelpad=0.8)
ax2.set_xlabel(r"$U/N$")

ax2.annotate(text=r'$\times 10^5$', xy=(0.039, 0.405), xycoords='figure fraction', fontsize=7.5, rotation=90, va='center')
             
ax4.set_xlabel(r"$\langle z \rangle - z_c$")

ax4y = ax4.secondary_yaxis("right")
ax4y.set_ylabel(r"$P(\langle z \rangle - z_c)$")
ax4.tick_params(which="both", axis='y', right=False, labelright=False)
ax4.tick_params(which="both", axis='y', left=False, labelleft=False)
ax4y.tick_params(which="both", axis='y', right=True, labelright=True, direction="in")
ax4y.tick_params(which="both", axis='y', left=False, labelleft=False)

ax2.legend(loc="upper left", frameon = False, bbox_to_anchor=(-0.055, 1.0))
ax4.legend(loc="upper left", frameon = False, bbox_to_anchor=(-0.055, 1.0))

yticks = [0, 0.25e5, 0.5e5, 0.75e5, 1.0e5, 1.25e5]
yticklables = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
ax2.set_yticks(yticks)
ax2.set_yticklabels(yticklables)
ax2.xaxis.set_major_locator(MultipleLocator(1.0e-5))
ax2.xaxis.set_minor_locator(MultipleLocator(0.5e-5))

ax4.xaxis.set_major_locator(MultipleLocator(0.1))
ax4.xaxis.set_minor_locator(MultipleLocator(0.05))

fig.tight_layout()
plt.subplots_adjust(wspace=0.03, hspace=0.25)

fig.savefig(output_foldername + graphs_foldername +
            "FIG_sqrheology.jpg", dpi=1000, bbox_inches='tight')
plt.close()

########################################################################################################################

# Fig reset (a), (b) and (c)

del U, Z, R

seed_plot = 2
seed_plot_0 = 4

t_start = 400
t_end = 500

system_foldername = "./"
N_foldername = "./"

type_foldername = "Damped_ripening"

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

type_foldername = "Damped_reset/"

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
                U_reset
            except NameError:
                U_reset = U_temp
            else:
                U_reset = pd.concat([U_reset, U_temp], ignore_index=True)

            Z_temp = pd.read_csv(join(PATH, dir, output_foldername, Z_filename), sep=r"\s+", header=None, skiprows=1)
            Z_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "Nc >= 0", "Nc >= 1", "Nc >= 2",
                          "Nc >= 3", "Nc >= 4", "Nc >= 5", "Nc >= 6", "N_nr >= 0", "N_nr >= 1", "N_nr >= 2", "N_nr >= 3", "N_nr >= 4", "N_nr >= 5", "N_nr >= 6", "vol_frac", "N"]

            Z_temp["b"] = b
            Z_temp["random_seed_pos_init"] = random_seed_pos_init
            Z_temp["dt_minimizer"] = dt_minimizer
            Z_temp["simkey"] = i[0]

            try:
                Z_reset
            except NameError:
                Z_reset = Z_temp
            else:
                Z_reset = pd.concat([Z_reset, Z_temp], ignore_index=True)

            R_temp = pd.read_csv(join(PATH, dir, output_foldername, R_filename), sep=r"\s+", header=None, skiprows=1)
            R_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "N",
                          "mass_init - mass_N", "<R>", "<R>^2", "sqrt(<R^2>)", "R_1", "R_2", "R_3", "R12 < L"]

            R_temp["b"] = b
            R_temp["random_seed_pos_init"] = random_seed_pos_init
            R_temp["dt_minimizer"] = dt_minimizer
            R_temp["simkey"] = i[0]

            try:
                R_reset
            except NameError:
                R_reset = R_temp
            else:
                R_reset = pd.concat([R_reset, R_temp], ignore_index=True)

type_foldername = "Damped_reset_no_ripening/"

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
            f.loc[f["name"] == "dt_minimizer"]["value"].values[0])

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
                U_reset_nr
            except NameError:
                U_reset_nr = U_temp
            else:
                U_reset_nr = pd.concat([U_reset_nr, U_temp], ignore_index=True)

            Z_temp = pd.read_csv(join(PATH, dir, output_foldername, Z_filename), sep=r"\s+", header=None, skiprows=1)
            Z_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "Nc >= 0", "Nc >= 1", "Nc >= 2",
                          "Nc >= 3", "Nc >= 4", "Nc >= 5", "Nc >= 6", "N_nr >= 0", "N_nr >= 1", "N_nr >= 2", "N_nr >= 3", "N_nr >= 4", "N_nr >= 5", "N_nr >= 6", "vol_frac", "N"]

            Z_temp["b"] = b
            Z_temp["random_seed_pos_init"] = random_seed_pos_init
            Z_temp["dt_minimizer"] = dt_minimizer
            Z_temp["simkey"] = i[0]

            try:
                Z_reset_nr
            except NameError:
                Z_reset_nr = Z_temp
            else:
                Z_reset_nr = pd.concat([Z_reset_nr, Z_temp], ignore_index=True)

            R_temp = pd.read_csv(join(PATH, dir, output_foldername, R_filename), sep=r"\s+", header=None, skiprows=1)
            R_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "N",
                          "mass_init - mass_N", "<R>", "<R>^2", "sqrt(<R^2>)", "R_1", "R_2", "R_3", "R12 < L"]

            R_temp["b"] = b
            R_temp["random_seed_pos_init"] = random_seed_pos_init
            R_temp["dt_minimizer"] = dt_minimizer
            R_temp["simkey"] = i[0]

            try:
                R_reset_nr
            except NameError:
                R_reset_nr = R_temp
            else:
                R_reset_nr = pd.concat([R_reset_nr, R_temp], ignore_index=True)

fig = plt.figure()
fig.set_figheight(2*fig_height)
fig.set_figwidth(fig_width)

ax1 = plt.subplot2grid(shape=(2, 1), loc=(0, 0), colspan=1)
ax2 = plt.subplot2grid(shape=(2, 1), loc=(1, 0), colspan=1)

t_start_plot = 300

t_reset = 400

bplot = [0, 0.001, 0.01, 0.1]
for b in np.sort(bplot):
    U_plot = U.loc[((U["status_flag"]).astype(float).astype(int) == 0) & (U["b"] == b) & (U["t"] >= t_start_plot) & (U["t"] <= t_end) & ((((b != 0) & (b != 0.0001)) & (U["random_seed_pos_init"] == seed_plot)) | (((b == 0) | (b == 0.0001)) & (U["random_seed_pos_init"] == seed_plot_0)))]
    U_reset_plot = U_reset.loc[((U_reset["status_flag"]).astype(float).astype(int) == 0) & (U_reset["b"] == b) & (U_reset["t"] >= t_start_plot) & (U_reset["t"] <= t_end) & ((((b != 0) & (b != 0.0001)) & (U_reset["random_seed_pos_init"] == seed_plot)) | (((b == 0) | (b == 0.0001)) & (U_reset["random_seed_pos_init"] == seed_plot_0)))]
    U_reset_nr_plot = U_reset_nr.loc[((U_reset_nr["status_flag"]).astype(float).astype(int) == 0) & (U_reset_nr["b"] == b) & ((((b != 0) & (b != 0.0001)) & (U_reset_nr["random_seed_pos_init"] == seed_plot)) | (((b == 0) | (b == 0.0001)) & (U_reset_nr["random_seed_pos_init"] == seed_plot_0)))]
    U_reset_nr_plot = U_reset_nr_plot.loc[(U_reset_nr_plot["t"] >= t_start_plot) & (U_reset_nr_plot["t"] <= t_end)]
    x1 = np.array(U_plot["t"])
    y1 = np.array(U_plot["U"])
    x1_reset = np.array(U_reset_plot["t"])
    y1_reset = np.array(U_reset_plot["U"])
    x1_reset_nr = np.array(U_reset_nr_plot["t"])
    y1_reset_nr = np.array(U_reset_nr_plot["U"])

    Z_plot = Z.loc[((Z["status_flag"]).astype(float).astype(int) == 0) & (Z["b"] == b) & (Z["t"] >= t_start_plot) & (Z["t"] <= t_end) & ((((b != 0) & (b != 0.0001)) & (Z["random_seed_pos_init"] == seed_plot)) | (((b == 0) | (b == 0.0001)) & (Z["random_seed_pos_init"] == seed_plot_0)))]
    Z_reset_plot = Z_reset.loc[((Z_reset["status_flag"]).astype(float).astype(int) == 0) & (Z_reset["b"] == b) & (Z_reset["t"] >= t_start_plot) & (Z_reset["t"] <= t_end) & ((((b != 0) & (b != 0.0001)) & (Z_reset["random_seed_pos_init"] == seed_plot)) | (((b == 0) | (b == 0.0001)) & (Z_reset["random_seed_pos_init"] == seed_plot_0)))]
    Z_reset_nr_plot = Z_reset_nr.loc[((Z_reset_nr["status_flag"]).astype(float).astype(int) == 0) & (Z_reset_nr["b"] == b) & ((((b != 0) & (b != 0.0001)) & (Z_reset_nr["random_seed_pos_init"] == seed_plot)) | (((b == 0) | (b == 0.0001)) & (Z_reset_nr["random_seed_pos_init"] == seed_plot_0)))]
    Z_reset_nr_plot = Z_reset_nr_plot.loc[(Z_reset_nr_plot["t"] >= t_start_plot) & (Z_reset_nr_plot["t"] <= t_end)]
    x2 = np.array(Z_plot["t"])
    y2 = np.array(Z_plot["Nc >= 4"])
    x2_reset = np.array(Z_reset_plot["t"])
    y2_reset = np.array(Z_reset_plot["Nc >= 4"])
    x2_reset_nr = np.array(Z_reset_nr_plot["t"])
    y2_reset_nr = np.array(Z_reset_nr_plot["Nc >= 4"])

    argsort = np.argsort(x1)
    x1 = x1[argsort]
    y1 = y1[argsort]
    argsort = np.argsort(x2)
    x2 = x2[argsort]
    y2 = y2[argsort]
    argsort = np.argsort(x1_reset)
    x1_reset = x1_reset[argsort]
    y1_reset = y1_reset[argsort]
    argsort = np.argsort(x2_reset)
    x2_reset = x2_reset[argsort]
    y2_reset = y2_reset[argsort]
    argsort = np.argsort(x1_reset_nr)
    x1_reset_nr = x1_reset_nr[argsort]
    y1_reset_nr = y1_reset_nr[argsort]
    argsort = np.argsort(x2_reset_nr)
    x2_reset_nr = x2_reset_nr[argsort]
    y2_reset_nr = y2_reset_nr[argsort]
    if b != 0:
        ax1.plot(x1, y1, color=colors[blist.index((b))], alpha=0.5, ls="--")
        ax1.plot(x1_reset, y1_reset, color=colors[blist.index((b))], label=r"$\xi = {}$".format(b))
        if x1_reset_nr.shape[0] > 0:
            ax1.scatter(x1_reset_nr, y1_reset_nr, facecolors='none', edgecolors=colors[blist.index((b))], alpha=0.25, s=3)

        y11 = y1_reset[(x1_reset > t_start) & (x1_reset <= x1_reset[-1]) & (x1_reset <= x1[-1])]
        y12 = y1[(x1 > t_start) & (x1 <= x1_reset[-1]) & (x1 <= x1[-1])]
        x11 = x1[(x1 > t_start) & (x1 <= x1_reset[-1]) & (x1 <= x1[-1])]

        y11 = y11[1::2]

        if x11.shape[0] > 0:
            ax2.plot(x2, y2, color=colors[blist.index((b))], alpha=0.5, ls="--")
            ax2.plot(x2_reset, y2_reset, color=colors[blist.index((b))], label=r"$\xi = {}$".format(b))
            if x2_reset_nr.shape[0] > 0:
                ax2.scatter(x2_reset_nr, y2_reset_nr, facecolors='none', edgecolors=colors[blist.index((b))], alpha=0.25, s=3)
            
    else:
        ax1.plot(x1, y1, color=colors[blist.index((b))], alpha=0.5, ls="--")
        ax1.plot(x1_reset, y1_reset, color=colors[blist.index((b))], label=r"Quasistatic")
        if x1_reset_nr.shape[0] > 0:
            ax1.scatter(x1_reset_nr, y1_reset_nr, facecolors='none', edgecolors=colors[blist.index((b))], alpha=0.25, s=3)

        y11 = y1_reset[(x1_reset > t_start) & (x1_reset <= x1_reset[-1]) & (x1_reset <= x1[-1])]
        y12 = y1[(x1 > t_start) & (x1 <= x1_reset[-1]) & (x1 <= x1[-1])]
        x11 = x1[(x1 > t_start) & (x1 <= x1_reset[-1]) & (x1 <= x1[-1])]
        
        if x11.shape[0] > 0:
            ax2.plot(x2, y2, color=colors[blist.index((b))], alpha=0.5, ls="--")
            ax2.plot(x2_reset, y2_reset, color=colors[blist.index((b))], label=r"Quasistatic")
            if x2_reset_nr.shape[0] > 0:
                ax2.scatter(x2_reset_nr, y2_reset_nr, facecolors='none', edgecolors=colors[blist.index((b))], alpha=0.25, s=3)

ax1.legend(loc="best", frameon = False)

ax1.set_xlabel(r"$t$")

ax1.set_ylabel(r"$U$")
ax1.set_xlim(left=287)
ax1.tick_params(which="both", axis='y', right=True, labelright=False, direction="in")
ax1.tick_params(which="both", axis='y', left=True, labelleft=True)

ax1.set_yscale("log")
ax2.set_xlabel(r"$t$")

ax2.set_ylabel(r"$\langle z \rangle - z_c$")
ax2.set_xlim(left=287)
ax2.tick_params(which="both", axis='y', right=True, labelright=False, direction="in")
ax2.tick_params(which="both", axis='y', left=True, labelleft=True)

fig.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0.25)

fig.savefig(output_foldername + graphs_foldername +
            "FIG_reset.jpg", dpi=1000, bbox_inches='tight')
plt.close()

########################################################################################################################

# Fig xi = 0 reset (a), (b) and (c)

# del U, Z

seed_plot = 4
seed_plot_0 = 4

t_start = 400
t_end = 700

system_foldername = "./"
N_foldername = "./"

type_foldername = "random_quenches/"

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
            f.loc[f["name"] == "dt_minimizer"]["value"].values[0])

        U_filename = str(f.loc[f["name"] == "U_filename"]["value"].values[0])

        Z_filename = str(f.loc[f["name"] == "Z_filename"]["value"].values[0])

        R_filename = str(f.loc[f["name"] == "R_filename"]["value"].values[0])

        if (b < 10):
            U_temp = pd.read_csv(join(PATH, dir, output_foldername, U_filename), sep=r"\s+",
                                    header=None, skiprows=1)
            U_temp.columns = ["system_counter", "t", "counter", "counter_time", "status_flag", "U", "dU", "dU_U", "N", "min_counter_SUCCESSFULL", "min_counter", "U_min", "U_max", "F_max"]

            U_temp["b"] = b
            U_temp["random_seed_pos_init"] = random_seed_pos_init
            U_temp["dt_minimizer"] = dt_minimizer
            U_temp["simkey"] = i[0]

            try:
                U_sq
            except NameError:
                U_sq = U_temp
            else:
                U_sq = pd.concat([U_sq, U_temp], ignore_index=True)

            Z_temp = pd.read_csv(join(PATH, dir, output_foldername, Z_filename), sep=r"\s+", header=None, skiprows=1)
            Z_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "Nc >= 0", "Nc >= 1", "Nc >= 2",
                          "Nc >= 3", "Nc >= 4", "Nc >= 5", "Nc >= 6", "N_nr >= 0", "N_nr >= 1", "N_nr >= 2", "N_nr >= 3", "N_nr >= 4", "N_nr >= 5", "N_nr >= 6", "vol_frac", "p", "N"]

            Z_temp["b"] = b
            Z_temp["random_seed_pos_init"] = random_seed_pos_init
            Z_temp["dt_minimizer"] = dt_minimizer
            Z_temp["simkey"] = i[0]

            try:
                Z_sq
            except NameError:
                Z_sq = Z_temp
            else:
                Z_sq = pd.concat([Z_sq, Z_temp], ignore_index=True)

            R_temp = pd.read_csv(join(PATH, dir, output_foldername, R_filename), sep=r"\s+", header=None, skiprows=1)
            R_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "N",
                          "mass_init - mass_N", "<R>", "<R>^2", "sqrt(<R^2>)", "R_1", "R_2", "R_3", "R12 < L"]

            R_temp["b"] = b
            R_temp["random_seed_pos_init"] = random_seed_pos_init
            R_temp["dt_minimizer"] = dt_minimizer
            R_temp["simkey"] = i[0]

            try:
                R_sq
            except NameError:
                R_sq = R_temp
            else:
                R_sq = pd.concat([R_sq, R_temp], ignore_index=True)

fig = plt.figure()
fig.set_figheight(2*fig_height)
fig.set_figwidth(fig_width)

ax1 = plt.subplot2grid(shape=(2, 1), loc=(0, 0), colspan=1)
ax2 = plt.subplot2grid(shape=(2, 1), loc=(1, 0), colspan=1)

t_start_plot = 300

bplot = [0]
for b in np.sort(bplot):
    R_plot = R.loc[((R["status_flag"]).astype(float).astype(int) == 0) & (R["b"] == b) & (R["t"] >= t_start_plot) & (R["t"] <= t_end) & (((b != 0) & (R["random_seed_pos_init"] == seed_plot)) | ((b == 0) & (R["random_seed_pos_init"] == seed_plot)))]
    R_reset_plot = R_reset.loc[((R_reset["status_flag"]).astype(float).astype(int) == 0) & (R_reset["b"] == b) & (R_reset["t"] >= t_start_plot) & (R_reset["t"] <= t_end) & (((b != 0) & (R_reset["random_seed_pos_init"] == seed_plot)) | ((b == 0) & (R_reset["random_seed_pos_init"] == seed_plot)))]
    x2 = np.array(R_plot["t"])
    y2 = np.array(R_plot["<R>^2"])
    x2_reset = np.array(R_reset_plot["t"])
    y2_reset = np.array(R_reset_plot["<R>^2"])

    Z_plot = Z.loc[((Z["status_flag"]).astype(float).astype(int) == 0) & (Z["b"] == b) & (Z["t"] >= t_start_plot) & (Z["t"] <= t_end) & (((b != 0) & (Z["random_seed_pos_init"] == seed_plot)) | ((b == 0) & (Z["random_seed_pos_init"] == seed_plot)))]
    Z_reset_plot = Z_reset.loc[((Z_reset["status_flag"]).astype(float).astype(int) == 0) & (Z_reset["b"] == b) & (Z_reset["t"] >= t_start_plot) & (Z_reset["t"] <= t_end) & (((b != 0) & (Z_reset["random_seed_pos_init"] == seed_plot)) | ((b == 0) & (Z_reset["random_seed_pos_init"] == seed_plot)))]
    x1 = np.array(Z_plot["t"])
    y1 = np.array(Z_plot["Nc >= 4"])
    x1_reset = np.array(Z_reset_plot["t"])
    y1_reset = np.array(Z_reset_plot["Nc >= 4"])

    argsort = np.argsort(x1)
    x1 = x1[argsort]
    y1 = y1[argsort]
    argsort = np.argsort(x1_reset)
    x1_reset = x1_reset[argsort]
    y1_reset = y1_reset[argsort]
    argsort = np.argsort(x2_reset)
    x2_reset = x2_reset[argsort]
    y2_reset = y2_reset[argsort]
    if b != 0:
        ax1.plot(x1, y1, color=colors[blist.index((b))], alpha=0.5)
        ax1.plot(x1_reset, y1_reset, color=colors[blist.index((b))], label=r"$\xi = {}$".format(b))
    else:
        ax1.plot(x1, y1, color=colors[blist.index((b))], alpha=0.5)
        ax1.plot(x1_reset, y1_reset, color=colors[blist.index((b))], label=r"Quasistatic")

ax1.set_xlabel(r"$t$")

ax1.set_ylabel(r"$\langle z \rangle - z_c$")
ax1.tick_params(which="both", axis='y', right=True, labelright=False, direction="in")
ax1.tick_params(which="both", axis='y', left=True, labelleft=True)

left, bottom, width, height = [0.64, 0.778, 0.28, 0.28*golden_mean]
ax3 = fig.add_axes([left, bottom, width, height])

lags=250
bplot = [0]
seedplot = [4]
for b in bplot:
    for seed in seedplot:
        Z_plot = Z.loc[((Z["status_flag"]).astype(float).astype(int) == 0) & (Z["b"] == b) & (((b != 0) & (Z["random_seed_pos_init"] == seed)) | ((b == 0) & (Z["random_seed_pos_init"] == seed)))]
        Z_reset_plot = Z_reset.loc[((Z_reset["status_flag"]).astype(float).astype(int) == 0) & (Z_reset["b"] == b) & (((b != 0) & (Z_reset["random_seed_pos_init"] == seed)) | ((b == 0) & (Z_reset["random_seed_pos_init"] == seed)))]
        x1 = np.array(Z_plot["t"])
        y1 = np.array(Z_plot["Nc >= 4"])
        x1_reset = np.array(Z_reset_plot["t"])
        y1_reset = np.array(Z_reset_plot["Nc >= 4"])
        y11 = y1_reset[(x1_reset > t_start) & (x1_reset <= x1_reset[-1]) & (x1_reset <= x1[-1])]
        y12 = y1[(x1 > t_start) & (x1 <= x1_reset[-1]) & (x1 <= x1[-1])]
        x11 = x1[(x1 > t_start) & (x1 <= x1_reset[-1]) & (x1 <= x1[-1])]
        if x11.shape[0] > 0:
            try:
                acorr_11
            except NameError:
                acorr_11 = acf(y11, nlags=lags)
                acorr_11 = np.reshape(acorr_11, (1, acorr_11.shape[0]))

                acorr_12 = acf(y12, nlags=lags)
                acorr_12 = np.reshape(acorr_12, (1, acorr_12.shape[0]))
            else:
                acorr_11 = np.vstack((acorr_11, acf(y11, nlags=lags)))
                acorr_12 = np.vstack((acorr_12, acf(y12, nlags=lags)))

acorr_11 = np.mean(acorr_11, axis=0)
acorr_12 = np.mean(acorr_12, axis=0)
ax3.plot(np.arange(0, len(acorr_11)), acorr_11, color="cornflowerblue", ls="-", label=r"Quasistatic")
ax3.plot(np.arange(0, len(acorr_12)), acorr_12, color="lightsteelblue", ls="-", label=r"Quasistatic")

ax3.set_xlabel(r"$\tau$", labelpad=0.5, fontsize=7)
ax3.set_ylabel(r"$\mathcal{A}(\langle z \rangle)$", labelpad=-3.8, fontsize=7)
ax3.tick_params(axis='both', which='both', pad=0.5, labelsize=5)
ax3.tick_params(which="both", axis='y', right=True, labelright=False, direction="in")
ax3.tick_params(which="both", axis='y', left=True, labelleft=True)
# ax3.set_xlim(right=50)
ax3.set_ylim(bottom=-0.45, top=0.75)
ax3.set_xscale("log")

t_start = 400
t_end = 700

t_start_plot = 300

bplot = [0]
for b in np.sort(bplot):
    R_plot = R.loc[((R["status_flag"]).astype(float).astype(int) == 0) & (R["b"] == b) & (R["t"] >= t_start_plot) & (R["t"] <= t_end) & (((b != 0) & (R["random_seed_pos_init"] == seed_plot)) | ((b == 0) & (R["random_seed_pos_init"] == seed_plot)))]
    R_reset_plot = R_reset.loc[((R_reset["status_flag"]).astype(float).astype(int) == 0) & (R_reset["b"] == b) & (R_reset["t"] >= t_start_plot) & (R_reset["t"] <= t_end) & (((b != 0) & (R_reset["random_seed_pos_init"] == seed_plot)) | ((b == 0) & (R_reset["random_seed_pos_init"] == seed_plot)))]
    x2 = np.array(R_plot["t"])
    y2 = np.array(R_plot["<R>^2"])
    x2_reset = np.array(R_reset_plot["t"])
    y2_reset = np.array(R_reset_plot["<R>^2"])

    Z_plot = Z.loc[((Z["status_flag"]).astype(float).astype(int) == 0) & (Z["b"] == b) & (Z["t"] >= t_start_plot) & (Z["t"] <= t_end) & (((b != 0) & (Z["random_seed_pos_init"] == seed_plot)) | ((b == 0) & (Z["random_seed_pos_init"] == seed_plot)))]
    Z_reset_plot = Z_reset.loc[((Z_reset["status_flag"]).astype(float).astype(int) == 0) & (Z_reset["b"] == b) & (Z_reset["t"] >= t_start_plot) & (Z_reset["t"] <= t_end) & (((b != 0) & (Z_reset["random_seed_pos_init"] == seed_plot)) | ((b == 0) & (Z_reset["random_seed_pos_init"] == seed_plot)))]
    x1 = np.array(Z_plot["t"])
    y1 = np.array(Z_plot["Nc >= 4"])
    x1_reset = np.array(Z_reset_plot["t"])
    y1_reset = np.array(Z_reset_plot["Nc >= 4"])

    argsort = np.argsort(x1)
    x1 = x1[argsort]
    y1 = y1[argsort]
    argsort = np.argsort(x1_reset)
    x1_reset = x1_reset[argsort]
    y1_reset = y1_reset[argsort]
    argsort = np.argsort(x2_reset)
    x2_reset = x2_reset[argsort]
    y2_reset = y2_reset[argsort]
    if b != 0:
        ax2.plot(x2, y2, color="tab:blue", alpha=0.5)
        ax2.plot(x2_reset, y2_reset, color="tab:blue", label=r"$\xi = {}$".format(b))
    else:
        ax2.plot(x2, y2, color="tab:blue", alpha=0.5)
        ax2.plot(x2_reset, y2_reset, color="tab:blue", label=r"Quasistatic")

ax2.set_xlabel(r"$t$")

ax2.set_ylabel(r"${\langle a \rangle}^2$")
ax2.tick_params(which="both", axis='y', right=True, labelright=False, direction="in")
ax2.tick_params(which="both", axis='y', left=True, labelleft=True)

ax2.set_ylim(bottom=0.505)

left, bottom, width, height = [0.3, 0.316, 0.27, 0.27*golden_mean]
ax4 = fig.add_axes([left, bottom, width, height])

x4 = x2
y4 = np.array(R_plot["sqrt(<R^2>)"])*np.sqrt((np.array(R_plot["N"])-1)/ np.array(R_plot["N"])) / np.array(R_plot["<R>"])
x4_reset = x2_reset
y4_reset = np.array(R_reset_plot["sqrt(<R^2>)"])*np.sqrt((np.array(R_reset_plot["N"])-1)/ np.array(R_reset_plot["N"])) / np.array(R_reset_plot["<R>"])
ax4.plot(x4_reset, y4_reset, color="royalblue")
ax4.plot(x4, y4, color="royalblue", alpha=0.5)

ax2.annotate("", xy=(400, 0.548), xytext=(400, 0.521),arrowprops=dict(arrowstyle="->", lw=0.55), color="black")

ax4.annotate("", xy=(393, 0.6275), xytext=(330, 0.621),arrowprops=dict(arrowstyle="->", lw=0.55), color="black")
ax2.text(400, 0.518, r"scramble", fontsize=6, ha="center", va="center")

ax4.set_xlabel(r"$t$", labelpad=0.5, fontsize=7)
ax4.set_ylabel(r"$\sqrt{{(a - \langle a \rangle)}^2}/ \langle a \rangle$", labelpad=0.05, fontsize=6)
ax4.tick_params(axis='both', which='both', pad=0.5, labelsize=5)
ax4.tick_params(which="both", axis='y', right=True, labelright=False, direction="in")
ax4.tick_params(which="both", axis='y', left=True, labelleft=True)

fig.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0.25)

fig.savefig(output_foldername + graphs_foldername +
            "FIG_QSreset.jpg", dpi=1000, bbox_inches='tight')
plt.close()

########################################################################################################################