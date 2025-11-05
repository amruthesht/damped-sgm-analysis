'''
File: rheology_analysis.py
Project: Q_analysis
File Created: Monday, 20th April 2020 1:40:40 am
Author: Amruthesh T (amru@seas.upenn.edu)
-----
Last Modified: Tuesday, 25th July 2023 8:19:36 am
Modified By: Amruthesh T (amru@seas.upenn.edu)
-----
Copyright (c) 2018 - 2019 Amru, University of Pennsylvania

Summary: Fill In
'''

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from os import popen, makedirs, walk
from os.path import join, isfile, isdir, basename, dirname, exists
#from rheology import rheology
from masonft import *
# from vfit_new import *
from scipy import stats
import sys
from scipy.interpolate import interp1d

output_foldername = "output/"
graphs_foldername = "graphs/"
input_foldername = "input/"
input_filename = "init.input"
run_filename = "run_config.txt"

def jcc_rheology(dt, dr2, dsigma2, R, w, option=1, der=2):
    if (option == 1):
        print("dr2")
        # w = 4
        omega, dr2ft = masonft(dt, dr2, w, der=der)
        print("dsigma2")
        # w = 4
        omega, dsigma2ft = masonft(dt, dsigma2, w)
        Gstar = np.sqrt(np.divide(dsigma2ft,#dr2ft))
                                  3*np.pi*R*dr2ft)) 
        # #ignoring radius value for convenience; radius evolution similar across all b

        Gft = np.divide(Gstar, omega)

        Jft = np.divide(np.power(omega, -2), Gft)

        return omega, Gstar, Gft, Jft
    
    elif (option == 2): # not well tested and not robust
        # print("dr2")
        # w = 4
        omega, dr2ft = npft(dt, dr2)
        # print("dsigma2")
        # w = 4
        omega, dsigma2ft = npft(dt, dsigma2)
        Gstar = np.sqrt(np.divide(dsigma2ft,
                                  3*np.pi*R*dr2ft))

        Gft = np.divide(Gstar, omega)

        Jft = np.divide(np.power(omega, -2), Gft)

        return omega, Gstar, Gft, Jft

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

colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "blue", "green", "red"]
tabcolors = ["tab:blue", "tab:green", "tab:red", "tab:cyan", "tab:purple", "tab:olive", "tab:blue", "tab:green", "tab:red"]
markers = ["s", "o", "^"]
s = [14.5, 15, 15.5]

Ginf = np.array([])
nR = np.array([])
kD = np.array([])
m = np.array([])
tau0 = np.array([])
alpha = np.array([])
brange = np.array([])

bfit = [0, 0.0001, 0.001, 0.01, 0.1, 1]
blist = [0, 0.0001, 0.001, 0.01, 0.04, 0.1, 0.3, 1]
colorlist = ["blue", "green", "red", "cyan", "saddlebrown", "magenta", "darkviolet", "gold"]

f = pd.DataFrame(columns=["b", "Ginf", "nR", "kD", "tau0", "alpha", "omega", "G'", "G''", "G*", "t", "Jt"]) #, "w", "Gstar"])
f["b"] = blist

# Fit parameters typ
fit_method = "quad"
fit_scale = "log"
sample = "log" # "all"
fit_type = "jcc" # "amru"

if fit_type == "jcc":
    df_plot = pd.DataFrame(columns=["omega", "Gstar", "Gft", "Jft", "b", "beta"])

# Open the file in write mode
output_filename = "rheology_calculation_output_test.txt"
output_file = open(join(output_foldername, output_filename), "a")

# Redirect all print output to the file
sys.stdout = output_file
sys.stderr = output_file

def array_to_func(t, y):
    # return lambda x: np.interp1d(x, t, y)
    # return interp1d(t, y, kind='linear')#, fill_value='extrapolate')
    return lambda x: 10**(interp1d(np.log10(t), np.log10(y), kind='linear')(np.log10(x)))

# dr2ds.columns = ["dd2dc_dc", "dd2dc_dd2", "dd2dc_b", "dd2dt_dt", "dd2dt_dd2", "dd2_dt_b", "dTau2udt_dt", "dTau2udt_dTau2u", "dTau2udt_b", "dd2dt_dt_f", "dd2dt_dd2_f", "dd2_dt_b_f", "dTau2udt_dt_f", "dTau2udt_dTau2u_f", "dTau2udt_b_f", "dd2dt_dt_f_bin", "dd2dt_dd2_f_bin", "dd2_dt_b_f_bin", "dTau2udt_dt_f_bin", "dTau2udt_dTau2u_f_bin", "dTau2udt_b_f_bin"]
dr2dstau_filename = "dr2ds_configuration_path_analysis.txt"

dr2ds = pd.read_csv(join(output_foldername, dr2dstau_filename), sep=r"\s+", header=None,skiprows=0)

# dr2ds = dr2ds.astype(float)

dr2ds.columns =   ["dd2ds_ds", "dd2ds_dd2", "dd2ds_b", "dd2ds_ds_f", "dd2ds_dd2_f", "dd2ds_b_f", "dd2dt_dt", "dd2dt_dd2", "b_dd2_dt", "dTau2udt_dt", "dTau2udt_dTau2u", "dTau2udt_b", "dd2dt_dt_f", "dd2dt_dd2_f", "dd2_dt_b_f", "dTau2udt_dt_f", "dTau2udt_dTau2u_f", "dTau2udt_b_f", "dd2dt_dt_f_bin", "dd2dt_dd2_f_bin", "dd2_dt_b_f_bin", "dTau2udt_dt_f_bin", "dTau2udt_dTau2u_f_bin", "dTau2udt_b_f_bin", "dd2_mean", "dTau2_mean", "bmean", "dR", "P_dR", "dR_b", "dx", "P_dx", "dx_b", "dTau2ds_ds_f", "dTau2ds_dTau2_f", "dTau2ds_b_f", "dszdt_dt_f", "dszdt_dsz_f", "b_dsz_dt_f"]

t_end_list = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
t_start_list = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]

for b in blist:
    dr2ds_plot = dr2ds.loc[dr2ds["dd2_dt_b_f"] == b]
    if dr2ds_plot.size > 0:
        t_start = t_start_list[blist.index(b)]
        t_end = t_end_list[blist.index(b)]

        dt = dr2ds_plot["dd2dt_dt_f"]
        dr2 = dr2ds_plot["dd2dt_dd2_f"]

        dr2ds_plot = dr2ds.loc[dr2ds["dTau2udt_b_f"] == b]
        dt_other = dr2ds_plot["dTau2udt_dt_f"]
        dTau2 = dr2ds_plot["dTau2udt_dTau2u_f"]

        #log binned in log of x axis
        # nbins = 50
        nbins = np.size(dt)
        nbins = int(nbins/10)
        nbins = 10 if nbins < 10 else nbins

        # sample nbins points between min and max od dt log spaced
        if sample == "log":
            if b in []:#[0, 0.0001]:
                dt_sample = np.logspace(np.log10(np.min(dt)*5), np.log10(np.max(dt)), nbins)
            else:
                dt_sample = np.logspace(np.log10(np.min(dt)), np.log10(np.max(dt)), nbins)
            dr2_sample = array_to_func(dt, dr2)(dt_sample)
            dt_other_sample = np.logspace(np.log10(np.min(dt_other)), np.log10(np.max(dt_other)), nbins)
            dTau2_sample = array_to_func(dt_other, dTau2)(dt_other_sample)
        elif sample == "linear":
            if b in []:
                dt_sample = np.linspace(np.min(dt)*5, np.max(dt), nbins)
            else:
                dt_sample = np.linspace(np.min(dt), np.max(dt), nbins)
            dr2_sample = array_to_func(dt, dr2)(dt_sample)
            dt_other_sample = np.linspace(np.min(dt_other), np.max(dt_other), nbins)
            dTau2_sample = array_to_func(dt_other, dTau2)(dt_other_sample)
        else:
            dt_sample = dt
            dr2_sample = dr2
            dt_other_sample = dt_other
            dTau2_sample = dTau2

        if (dt_sample != dt_other_sample).all():
            raise ValueError("dt_sample != dt_other_sample")
        
        dt, dr2, dTau2 = np.array(dt), np.array(dr2), np.array(dTau2)
        dt_sample, dr2_sample, dTau2_sample = np.array(dt_sample), np.array(dr2_sample), np.array(dTau2_sample)
        
        if b in bfit:
            dt_integers = np.around(dt, decimals=0)
            dt_integers = dt_integers.astype(int)
            
            print("Fitting operation starting with new constraints and new guesses")
            
            if fit_type == "amru":
                # p, residuals, r_sqaured = vfit_inteq(b, dt_sample, dr2_sample, dt_integers, dTau2, dr2, np.min(dt_integers), np.max(dt_integers), dtau2mean, dr2mean, fit_scale, fit_method)
                
                # print("Final fitting parameters")
                # print(b, p)
                pass
            elif fit_type == "jcc":
                R = 1
                w = 2
                omega, Gstar, Gft, Jft = jcc_rheology(dt_sample, dr2_sample, dTau2_sample, R, w, 1, 1)
                # caluclate beta which is the log derivative of Gstar v/s omega at omega =  2e-2
                beta = np.gradient(np.log10(Gstar), np.log10(omega))
                # find index of omega closest to 2e-2
                beta = beta[np.argmin(np.abs(omega - 2e-2))]
                # add to dictionary to omega_plot, Gstar_plot, Gft_plot, Jft_plot with key b
                df_plot.loc[len(df_plot)] = [omega.tolist(), Gstar.tolist(), Gft.tolist(), Jft.tolist(), float(b), float(beta)]
                

# Print the results to plot if option is "jcc"
if fit_type == "jcc":
    plt.figure()
    for i in range(len(df_plot)):
        plt.plot(df_plot["omega"][i], df_plot["Gstar"][i], label=f"b = {df_plot['b'][i]}", color=colors[i])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$G^*$")
    plt.legend(loc="upper left")
    plt.title("G* vs omega")
    plt.grid()
    plt.savefig(join(output_foldername, graphs_foldername, "Gstar_vs_omega.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    # beta vs b
    for i in range(len(df_plot)):
        if df_plot["b"][i] == 0:
            plt.scatter(1e-5, df_plot["beta"][i], color=colors[i], marker="o", s=20)
        else:
            plt.scatter(df_plot["b"][i], df_plot["beta"][i], color=colors[i], marker="o", s=20)
    plt.xscale("log")
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\beta$")
    plt.title(r"$\beta$ vs $\xi$")
    plt.grid()
    plt.savefig(join(output_foldername, graphs_foldername, "beta_vs_b.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # print df_plot to file
    df_plot.to_csv(join(output_foldername, "G_jcc.csv"), index=False, sep=";")
# Reset sys.stdout to its default value (usually the console)
sys.stdout = sys.__stdout__

# Close the file after you're done
output_file.close()