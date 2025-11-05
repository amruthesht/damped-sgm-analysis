'''
File: dr2ds_analysis_ensemble.py
Project: Q_analysis
File Created: Monday, 20th April 2020 1:40:40 am
Author: Amruthesh T (amru@seas.upenn.edu)
-----
Last Modified: Friday, 21st July 2023 2:26:09 am
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
from scipy import stats
import numpy_indexed as npi

nseeds = 4#4
nbin = 50

output_foldername = "output/"
graphs_foldername = "graphs/"
input_foldername = "input/"
input_filename = "init.input"
run_filename = "run_config.txt"

dr2dstau_filename = "dr2ds_configuration_path_analysis.txt"

dr2dstau_filename_final = "dr2ds_configuration_path_analysis.txt"

dirs = [dirs for root, dirs, files in walk(".")]
dirs = np.array([dir for dir in dirs[0]])

blist = [0, 0.0001, 0.001, 0.01, 0.1, 1]

for i, dir in np.ndenumerate(dirs):
    if isfile(join(dir, output_foldername, run_filename)) & isfile(join(dir, output_foldername, dr2dstau_filename)):
        f = pd.read_csv(join(dir, output_foldername, run_filename), lineterminator="\n",
                        header=None, skip_blank_lines=False)
        f = f[0].str.split(r"\s{2,}", expand=True)
        f.columns = ["value", "name"]

        b = float(f.loc[f["name"] == "b"]["value"].values[0])

        random_seed_pos_init = int(float(f.loc[f["name"] == "random_seed_pos_init"]["value"].values[0]))

        if (random_seed_pos_init <= nseeds) & (b in blist):
            dr2ds_temp = pd.read_csv(join(dir, output_foldername, dr2dstau_filename), sep=r"\s+",
                                header=None, skiprows=1, usecols= [0,1,2,3] + [8,9] + [10,11] + list(np.arange(12,22,1))+ list(np.arange(28,37,1)))

            dr2ds_temp.columns = ["dTau2dt_dt_f",	"dTau2dt_dTau2_f",	"dTau2dt_dt",	"dTau2dt_dTau2", "dTau2ds_ds_f", "dTau2ds_dTau2_f",	"dcontourdt_dt", "dcontourdt_dcontour", "ddisplacement2dt_dt_f",	"ddisplacement2dt_ddisplacement2_f",	"ddisplacement2dt_dt",	"ddisplacement2dt_ddisplacement2",	"ddisplacement2ds_ds",	"ddisplacement2ds_ddisplacement2",	"ddisplacement2dsz_ds",	"ddisplacement2dsz_ddisplacement2",	"ddisplacement2ds_ds_f",	"ddisplacement2ds_ddisplacement2_f",	"dt_d_i",	"ddisplacement_i",	"R_i",	"dt_d_abs_i",	"ddisplacement_abs_i",	"R_abs_i",	"ddisplacment2",	"displacement2",	"Tau2"]

            dr2ds_temp["b"] = b

            dr2ds_temp["random_seed_pos_init"] = random_seed_pos_init

            dr2ds_temp["simkey"] = i[0]

            print(b, random_seed_pos_init)

            try:
                dr2ds
            except NameError:
                dr2ds = dr2ds_temp
            else:
                dr2ds = pd.concat([dr2ds, dr2ds_temp], ignore_index=True)

dr2ds_1 = pd.DataFrame()
dr2ds_2 = pd.DataFrame()
dr2ds_3 = pd.DataFrame()
dr2ds_4 = pd.DataFrame()
dr2ds_5 = pd.DataFrame()
dr2ds_6 = pd.DataFrame()
dr2ds_7 = pd.DataFrame()
dr2ds_8 = pd.DataFrame()
dr2ds_9 = pd.DataFrame()
dr2ds_10 = pd.DataFrame()
dr2ds_11 = pd.DataFrame()
dr2ds_12 = pd.DataFrame()
dr2ds_13 = pd.DataFrame()

for b in dr2ds["b"].unique():
    dr2ds_plot = dr2ds.loc[(dr2ds["b"] == b)]
    x_data, y_data = dr2ds_plot["ddisplacement2dsz_ds"], dr2ds_plot["ddisplacement2dsz_ddisplacement2"]
    x_data, y_data = x_data[~np.isnan(x_data)], y_data[~np.isnan(x_data)]
    x_data, y_data = x_data[~np.isnan(y_data)], y_data[~np.isnan(y_data)]
    _, bins = np.histogram(np.log10(x_data), bins="auto")
    y, x, bnumbr = stats.binned_statistic(
        x_data, np.log10(y_data), statistic=np.nanmean, bins=10**bins)
    y_std, x_std, bnumbr = stats.binned_statistic(
        x_data, np.log10(y_data), statistic=np.nanstd, bins=10**bins)
    y = 10**y
    x = 10**(0.5*(np.log10(x[1:])+np.log10(x[:-1])))
    y_std = 10**y_std
    #x = 10**(0.5*(np.log10(x[1:])+np.log10(x[:-1])))
    x, y, y_std = np.log10(x), np.log10(y), np.log10(y_std)

    plt.errorbar(x, y, yerr=y_std,
             label=r"$\mu$ = " + str(b) + " [simulation units]")
    plt.scatter(x, y)

    dr2ds_temp = pd.DataFrame({"dd2ds_ds": 10**x,
                            "dd2ds_dd2": 10**y, "dd2ds_b": b})
    dr2ds_1 = pd.concat([dr2ds_1, dr2ds_temp], ignore_index=True, axis=0)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta {r_c}^2$ [simulation units]")
plt.legend(loc="best")
plt.title(r"$\Delta {r_c}^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2ds_new.png")
# plt.show()
plt.close()

x_min = 3e-2
for b in dr2ds["b"].unique():
    dr2ds_plot = dr2ds.loc[(dr2ds["b"] == b)]
    x_data, y_data = dr2ds_plot["ddisplacement2ds_ds_f"], dr2ds_plot["ddisplacement2ds_ddisplacement2_f"]
    x_data, y_data = x_data[~np.isnan(x_data)], y_data[~np.isnan(x_data)]
    x_data, y_data = x_data[~np.isnan(y_data)], y_data[~np.isnan(y_data)]
    y_data = y_data[x_data >= x_min]
    x_data = x_data[x_data >= x_min]
    _, bins = np.histogram(np.log10(x_data), bins=nbin)

    bin_centers_valid = np.ones(len(bins)-1)
    y_all = np.array([])
    for random_seed_pos_init in dr2ds["random_seed_pos_init"].unique():
        dr2ds_plot = dr2ds[(dr2ds["b"] == b) & (dr2ds["random_seed_pos_init"] == random_seed_pos_init)]

        x_data, y_data = dr2ds_plot["ddisplacement2ds_ds_f"], dr2ds_plot["ddisplacement2ds_ddisplacement2_f"]
        x_data, y_data = x_data[~np.isnan(x_data)], y_data[~np.isnan(x_data)]
        x_data, y_data = x_data[~np.isnan(y_data)], y_data[~np.isnan(y_data)]

        # mak sure these bins have at least one data point from each seed
        count, _ = np.histogram((x_data), bins=10**bins)
        bin_centers_valid *= (count >= nseeds)

        y, x, bnumbr = stats.binned_statistic(
        x_data, np.log10(y_data), statistic=np.nanmean, bins=10**bins)

        if y_all.shape[0] == 0:
            y_all = np.array(y)
        else:
            y_all = np.vstack((y_all, y))

        x = 10**(0.5*(np.log10(x[1:])+np.log10(x[:-1])))

    y = np.nanmean(y_all, axis=0)
    y_std = np.nanstd(y_all, axis=0)
    x = x[bin_centers_valid == 1]
    y = y[bin_centers_valid == 1]
    y_std = y_std[bin_centers_valid == 1]

    y = 10**y
    y_std = 10**y_std

    print("y_std: ", y_std)
    x, y, y_std = np.log10(x), np.log10(y), np.log10(y_std)

    plt.errorbar(x, y, yerr=y_std,
             label=r"$\mu$ = " + str(b) + " [simulation units]")
    plt.scatter(x, y)

    if (b == 0.0):
        a, blim = 0.55, 1.55
        xp1 = x[(x >= a) & (x <= blim)]
        yp1 = y[(x >= a) & (x <= blim)]
        # yp1 = y[(x >= np.log10(5.5e0)) & (x <= np.log10(3.5e1))]
        # xp1 = x[(x >= np.log10(5.5e0)) & (x <= np.log10(3.5e1))]
        p, residuals, _, _, _ = np.polyfit(xp1, yp1, 1, full=True)

        px = np.poly1d(p)
        print("slope: dd2dsall ", p[0])
        y_pred = px(xp1)

        # Calculate the residuals
        residuals = yp1 - y_pred

        # Calculate the Total Sum of Squares (SST)
        sst = np.sum((yp1 - np.mean(yp1))**2)

        # Calculate the Residual Sum of Squares (SSR)
        ssr = np.sum(residuals**2)

        # Calculate R-squared
        r_squared = 1 - (ssr / sst)

        print(f"R-squared: {r_squared}")

        xi = 10**xp1[0]#10**a#5.8e0
        xf = 10**xp1[-1]#10**b#2.8e1
        yi = 10**(p[1] + p[0]*np.log10(xi))
        #10**(np.min(y[x >= a]))#1.1e0
        # yi = 1.1e0
        slope = p[0]
        x_vals = np.array([xi, xf])
        c = np.log10(yi) - slope * np.log10(xi)
        c = 10**c
        y_vals = c * np.power(x_vals, slope)
        print(residuals)
        plt.plot(np.log10(x_vals), np.log10(y_vals), color="black",linewidth=1, zorder=10)
        plt.annotate(text='$' +  str(np.around(slope, 2)) +'$', xy=(0.75, -1), xycoords='data', fontsize=6)

    dr2ds_temp = pd.DataFrame({"dd2ds_ds_f": 10**x,
                            "dd2ds_dd2_f": 10**y, "dd2ds_b_f": b})
    dr2ds_2 = pd.concat([dr2ds_2, dr2ds_temp], ignore_index=True, axis=0)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta {r_c}^2$ [simulation units]")
plt.legend(loc="best")
plt.title(r"$\Delta {r_c}^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2ds_all.png")
# plt.show()
plt.close()

for b in dr2ds["b"].unique():
    dr2ds_plot = dr2ds[(dr2ds["b"] == b)]

    x_data, y_data = dr2ds_plot["ddisplacement2dt_dt"], dr2ds_plot["ddisplacement2dt_ddisplacement2"]
    x_data, y_data = x_data[~np.isnan(x_data)], y_data[~np.isnan(x_data)]
    x_data, y_data = x_data[~np.isnan(y_data)], y_data[~np.isnan(y_data)]
    _, bins = np.histogram(np.log10(x_data), bins='auto')
    y, x, bnumbr = stats.binned_statistic(
        x_data, np.log10(y_data), statistic=np.nanmean, bins=10**bins)
    # x, y = x_data, y_data
    y_std, x_std, bnumbr = stats.binned_statistic(
        x_data, np.log10(y_data), statistic=np.nanstd, bins=10**bins)
    y = 10**y
    x = 10**(0.5*(np.log10(x[1:])+np.log10(x[:-1])))
    y_std = 10**y_std
    #x = 10**(0.5*(np.log10(x[1:])+np.log10(x[:-1])))
    x, y, y_std = np.log10(x), np.log10(y), np.log10(y_std)

    # print(y_std)
    plt.errorbar(x, y, yerr=y_std,
             label=r"$\mu$ = " + str(b) + " [simulation units]")
    plt.scatter(x, y)

    if (b == 0.0):
        a, blim = 1.0, 1.5
        xp1 = x[(x >= a) & (x <= blim)]
        yp1 = y[(x >= a) & (x <= blim)]
        # xp1 = x[(x >= np.log10(1.6e0)) & (x <= np.log10(1e1))]
        # yp1 = y[(x >= np.log10(1.6e0)) & (x <= np.log10(1e1))]
        p, residuals, _, _, _ = np.polyfit(xp1, yp1, 1, full=True)
        px = np.poly1d(p)
        print("slope: ", p[0])
        y_pred = px(xp1)

        # Calculate the residuals
        residuals = yp1 - y_pred

        # Calculate the Total Sum of Squares (SST)
        sst = np.sum((yp1 - np.mean(yp1))**2)

        # Calculate the Residual Sum of Squares (SSR)
        ssr = np.sum(residuals**2)

        # Calculate R-squared
        r_squared = 1 - (ssr / sst)

        print(f"R-squared: {r_squared}")

        # p = np.polyfit(np.log10(xp1), np.log10(yp1), 1)
        xi = 10**xp1[0]#10**a#5.8e0
        xf = 10**xp1[-1]#10**b#2.8e1
        yi = 10**(p[1] + p[0]*np.log10(xi))
        # yi = 10**(np.min(y[x >= a]))#2.8e-4
        # yi = 2.8e-4
        slope = p[0]
        x_vals = np.array([xi, xf])
        c = np.log10(yi) - slope * np.log10(xi)
        c = 10**c
        y_vals = c * np.power(x_vals, slope)
        print(residuals)
        plt.plot(np.log10(x_vals), np.log10(y_vals), color="black",linewidth=1, zorder=10)
        # ax3.annotate(text='0.91', xy=(2.5e3, 4.3e3), xycoords='data', fontsize=7)
        # print(slope)
        plt.annotate(text='$' +  str(np.around(slope, 2)) +'$', xy=(0.75, -4), xycoords='data', fontsize=6)

    dr2ds_temp = pd.DataFrame({"dd2dt_dt": 10**x,
                            "dd2dt_dd2": 10**y, "b_dd2_dt": b})
    dr2ds_3 = pd.concat([dr2ds_3, dr2ds_temp], ignore_index=True, axis=0)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta contour_c$ [simulation units]")
plt.legend(loc="best")
plt.title(r"$\Delta contour_c$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2dt_new.png")
# plt.show()
plt.close()

for b in dr2ds["b"].unique():
    dr2ds_plot = dr2ds[(dr2ds["b"] == b)]

    x_data, y_data = dr2ds_plot["dTau2dt_dt"], dr2ds_plot["dTau2dt_dTau2"]
    x_data, y_data = x_data[~np.isnan(x_data)], y_data[~np.isnan(x_data)]
    x_data, y_data = x_data[~np.isnan(y_data)], y_data[~np.isnan(y_data)]
    _, bins = np.histogram(np.log10(x_data), bins='auto')
    y, x, bnumbr = stats.binned_statistic(
        x_data, np.log10(y_data), statistic=np.nanmean, bins=10**bins)
    y_std, x_std, bnumbr = stats.binned_statistic(
        x_data, np.log10(y_data), statistic=np.nanstd, bins=10**bins)
    y = 10**y
    x = 10**(0.5*(np.log10(x[1:])+np.log10(x[:-1])))
    y_std = 10**y_std
    #x = 10**(0.5*(np.log10(x[1:])+np.log10(x[:-1])))
    x, y, y_std = np.log10(x), np.log10(y), np.log10(y_std)

    plt.errorbar(x, y, yerr=y_std,
             label=r"$\mu$ = " + str(b) + " [simulation units]")
    plt.scatter(x, y)

    if (b == 0.0):
        a, blim = 1.9, 2.4
        xp1 = x[(x >= a) & (x <= blim)]
        yp1 = y[(x >= a) & (x <= blim)]
        # xp1 = x[(x >= np.log10(1.6e0)) & (x <= np.log10(1e1))]
        # yp1 = y[(x >= np.log10(1.6e0)) & (x <= np.log10(1e1))]
        p, residuals, _, _, _ = np.polyfit(xp1, yp1, 1, full=True)
        px = np.poly1d(p)
        print("slope: ", p[0])
        y_pred = px(xp1)

        # Calculate the residuals
        residuals = yp1 - y_pred

        # Calculate the Total Sum of Squares (SST)
        sst = np.sum((yp1 - np.mean(yp1))**2)

        # Calculate the Residual Sum of Squares (SSR)
        ssr = np.sum(residuals**2)

        # Calculate R-squared
        r_squared = 1 - (ssr / sst)

        print(f"R-squared: {r_squared}")

        # p = np.polyfit(np.log10(xp1), np.log10(yp1), 1)
        xi = 10**xp1[0]#10**a#5.8e0
        xf = 10**xp1[-1]#10**b#2.8e1
        yi = 10**(p[1] + p[0]*np.log10(xi))
        # yi = 10**(np.min(y[x >= a]))#2.8e-4
        # yi = 2.8e-4
        slope = p[0]
        x_vals = np.array([xi, xf])
        c = np.log10(yi) - slope * np.log10(xi)
        c = 10**c
        y_vals = c * np.power(x_vals, slope)
        print(residuals)
        plt.plot(np.log10(x_vals), np.log10(y_vals), color="black",linewidth=1, zorder=10)
        # ax3.annotate(text='0.91', xy=(2.5e3, 4.3e3), xycoords='data', fontsize=7)
        # print(slope)
        plt.annotate(text='$' +  str(np.around(slope, 2)) +'$', xy=(0.75, -4), xycoords='data', fontsize=6)

    dr2ds_temp = pd.DataFrame({"dTau2udt_dt": 10**x,
                            "dTau2udt_dTau2u": 10**y, "dTau2udt_b": b})
    dr2ds_4 = pd.concat([dr2ds_4, dr2ds_temp], ignore_index=True, axis=0)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta \sigma^2$ [simulation units]")
plt.legend(loc="best")
plt.title(r"$\Delta \sigma^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dTau2udt_new.png")
# plt.show()
plt.close()

for b in dr2ds["b"].unique():
    x_d = np.array([])
    y_d = np.array([])
    for random_seed_pos_init in dr2ds["random_seed_pos_init"].unique():
        dr2ds_plot = dr2ds[(dr2ds["b"] == b) & (dr2ds["random_seed_pos_init"] == random_seed_pos_init)]

        x_data, y_data = dr2ds_plot["ddisplacement2dt_dt_f"], dr2ds_plot["ddisplacement2dt_ddisplacement2_f"]
        x_data, y_data = x_data[~np.isnan(x_data)], y_data[~np.isnan(x_data)]
        x_data, y_data = x_data[~np.isnan(y_data)], y_data[~np.isnan(y_data)]
        x_data, y_data = npi.group_by(np.around(x_data)).mean(y_data)
        x_d = np.append(x_d, x_data)
        y_d = np.append(y_d, y_data)

    _, nx = npi.count(np.around(x_d))
    _, y_std = npi.group_by(np.around(x_d)).std(y_d)
    x_d, y_d = npi.group_by(np.around(x_d)).mean(y_d)
    y_d = y_d[nx>=nseeds]
    x_d = x_d[nx>=nseeds]
    y_std = y_std[nx>=nseeds]
    # print(b, np.max(nx))
    # _, bins = np.histogram(np.log10(x_data), bins='auto')
    # y, x, bnumbr = stats.binned_statistic(
    #     x_data, np.log10(y_data), statistic=np.nanmean, bins=10**bins)
    x, y = x_d, y_d
    y_err = np.vstack((y_d - y_std, y_d + y_std))
    print(np.shape(x_d), np.shape(y_d), np.shape(y_err))
    x, y, y_err = np.log10(x), np.log10(y), np.log10(y_err)
    y_err = np.vstack((y - y_err[0], y_err[1] - y))

    plt.errorbar(x, y, yerr=y_err,
            label=r"$\mu$ = " + str(b) + " [simulation units]")
    plt.scatter(x, y)

    if (b == 0.0):
        a, blim = np.log10(25), np.log10(250)#1.4, 2.4
        xp1 = x[(x >= a) & (x <= blim)]
        yp1 = y[(x >= a) & (x <= blim)]
        # xp1 = x[(x >= np.log10(1.6e0)) & (x <= np.log10(1e1))]
        # yp1 = y[(x >= np.log10(1.6e0)) & (x <= np.log10(1e1))]
        p, residuals, _, _, _ = np.polyfit(xp1, yp1, 1, full=True)
        px = np.poly1d(p)
        print("slope: dd2dtf ", p[0])
        y_pred = px(xp1)

        # Calculate the residuals
        residuals = yp1 - y_pred

        # Calculate the Total Sum of Squares (SST)
        sst = np.sum((yp1 - np.mean(yp1))**2)

        # Calculate the Residual Sum of Squares (SSR)
        ssr = np.sum(residuals**2)

        # Calculate R-squared
        r_squared = 1 - (ssr / sst)

        print(f"R-squared: {r_squared}")

        n = len(yp1)                          # number of points
        sigma2 = ssr / (n - 2)                # residual variance estimate
        Sxx = np.sum((xp1 - np.mean(xp1))**2) # sum of squared x‑deviations

        slope_error = np.sqrt(sigma2 / Sxx)
        print(f"Slope error: {slope_error}")

        # p = np.polyfit(np.log10(xp1), np.log10(yp1), 1)
        xi = 10**xp1[0]#10**a#5.8e0
        xf = 10**xp1[-1]#10**b#2.8e1
        yi = 10**(p[1] + p[0]*np.log10(xi))
        # yi = 10**(np.min(y[x >= a]))#2.8e-4
        # yi = 2.8e-4
        slope = p[0]
        x_vals = np.array([xi, xf])
        c = np.log10(yi) - slope * np.log10(xi)
        c = 10**c
        y_vals = c * np.power(x_vals, slope)
        print(residuals)
        plt.plot(np.log10(x_vals), np.log10(y_vals), color="black",linewidth=1, zorder=10)
        # ax3.annotate(text='0.91', xy=(2.5e3, 4.3e3), xycoords='data', fontsize=7)
        # print(slope)
        plt.annotate(text='$' +  str(np.around(slope, 2)) +'$', xy=(0.75, -2), xycoords='data', fontsize=6)

    dr2ds_temp = pd.DataFrame({"dd2dt_dt_f": 10**x,
                            "dd2dt_dd2_f": 10**y, "b_dd2_dt_f": b})
    dr2ds_5 = pd.concat([dr2ds_5, dr2ds_temp], ignore_index=True, axis=0)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta contour_c$ [simulation units]")
plt.legend(loc="best")
plt.title(r"$\Delta contour_c$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2dt_f_new.png")
# # plt.show()
plt.close()

for b in dr2ds["b"].unique():
    x_d = np.array([])
    y_d = np.array([])
    for random_seed_pos_init in dr2ds["random_seed_pos_init"].unique():
        dr2ds_plot = dr2ds[(dr2ds["b"] == b) & (dr2ds["random_seed_pos_init"] == random_seed_pos_init)]

        x_data, y_data = dr2ds_plot["dTau2dt_dt_f"], dr2ds_plot["dTau2dt_dTau2_f"]
        x_data, y_data = x_data[~np.isnan(x_data)], y_data[~np.isnan(x_data)]
        x_data, y_data = x_data[~np.isnan(y_data)], y_data[~np.isnan(y_data)]
        x_data, y_data = npi.group_by(np.around(x_data)).mean(y_data)
        x_d = np.append(x_d, x_data)
        y_d = np.append(y_d, y_data)

    _, nx = npi.count(np.around(x_d))
    _, y_std = npi.group_by(np.around(x_d)).std(y_d)
    x_d, y_d = npi.group_by(np.around(x_d)).mean(y_d)
    y_d = y_d[nx>=nseeds]
    x_d = x_d[nx>=nseeds]
    y_std = y_std[nx>=nseeds]
    print(b, np.sum(nx>=nseeds))
    x, y = x_d, y_d
    y_err = np.vstack((y_d - y_std, y_d + y_std))
    x, y, y_err = np.log10(x), np.log10(y), np.log10(y_err)
    y_err = np.vstack((y - y_err[0], y_err[1] - y))

    plt.errorbar(x, y, yerr=y_err,
             label=r"$\mu$ = " + str(b) + " [simulation units]")
    plt.scatter(x, y)

    if (b == 0.0):
        a, blim = np.log10(25), np.log10(79)#1.4, 1.9
        xp1 = x[(x >= a) & (x <= blim)]
        yp1 = y[(x >= a) & (x <= blim)]
        # xp1 = x[(x >= np.log10(1.6e0)) & (x <= np.log10(1e1))]
        # yp1 = y[(x >= np.log10(1.6e0)) & (x <= np.log10(1e1))]
        p, residuals, _, _, _ = np.polyfit(xp1, yp1, 1, full=True)
        px = np.poly1d(p)
        print("slope: dTaudtf ", p[0])
        y_pred = px(xp1)

        # Calculate the residuals
        residuals = yp1 - y_pred

        # Calculate the Total Sum of Squares (SST)
        sst = np.sum((yp1 - np.mean(yp1))**2)

        # Calculate the Residual Sum of Squares (SSR)
        ssr = np.sum(residuals**2)

        # Calculate R-squared
        r_squared = 1 - (ssr / sst)

        print(f"R-squared: {r_squared}")

        n = len(yp1)                          # number of points
        sigma2 = ssr / (n - 2)                # residual variance estimate
        Sxx = np.sum((xp1 - np.mean(xp1))**2) # sum of squared x‑deviations

        slope_error = np.sqrt(sigma2 / Sxx)
        print(f"Slope error: {slope_error}")

        # p = np.polyfit(np.log10(xp1), np.log10(yp1), 1)
        xi = 10**xp1[0]#10**a#5.8e0
        xf = 10**xp1[-1]#10**b#2.8e1
        yi = 10**(p[1] + p[0]*np.log10(xi))
        # yi = 10**(np.min(y[x >= a]))#2.8e-4
        # yi = 2.8e-4
        slope = p[0]
        x_vals = np.array([xi, xf])
        c = np.log10(yi) - slope * np.log10(xi)
        c = 10**c
        y_vals = c * np.power(x_vals, slope)
        print(residuals)
        plt.plot(np.log10(x_vals), np.log10(y_vals), color="black",linewidth=1, zorder=10)
        # ax3.annotate(text='0.91', xy=(2.5e3, 4.3e3), xycoords='data', fontsize=7)
        # print(slope)
        plt.annotate(text='$' +  str(np.around(slope, 2)) +'$', xy=(0.75, -6), xycoords='data', fontsize=6)


    dr2ds_temp = pd.DataFrame({"dTau2udt_dt_f": 10**x,
                            "dTau2udt_dTau2u_f": 10**y, "dTau2udt_b_f": b})
    dr2ds_6 = pd.concat([dr2ds_6, dr2ds_temp], ignore_index=True, axis=0)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta \sigma^2$ [simulation units]")
plt.legend(loc="best")
plt.title(r"$\Delta \sigma^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dTau2udt_f_new.png")
# # plt.show()
plt.close()

for b in dr2ds["b"].unique():
    x_d = np.array([])
    y_d = np.array([])
    for random_seed_pos_init in dr2ds["random_seed_pos_init"].unique():
        dr2ds_plot = dr2ds[(dr2ds["b"] == b) & (dr2ds["random_seed_pos_init"] == random_seed_pos_init)]

        x_data, y_data = dr2ds_plot["ddisplacement2dt_dt_f"], dr2ds_plot["ddisplacement2dt_ddisplacement2_f"]
        x_data, y_data = x_data[~np.isnan(x_data)], y_data[~np.isnan(x_data)]
        x_data, y_data = x_data[~np.isnan(y_data)], y_data[~np.isnan(y_data)]
        x_data, y_data = npi.group_by(np.around(x_data)).mean(y_data)
        x_d = np.append(x_d, x_data)
        y_d = np.append(y_d, y_data)

    _, nx = npi.count(np.around(x_d))
    x_d, y_d = npi.group_by(np.around(x_d)).mean(y_d)
    y_d = y_d[nx>=nseeds]
    x_d = x_d[nx>=nseeds]
    x, y = x_d, y_d
    _, bins = np.histogram(np.log10(x_d), bins='auto')
    if y_d.shape[0] > 0:
        y, x, bnumbr = stats.binned_statistic(
            x_d, np.log10(y_d), statistic=np.nanmean, bins=10**bins)
        y_std, x_std, bnumbr = stats.binned_statistic(
            x_d, np.log10(y_d), statistic=np.nanstd, bins=10**bins)

        x = (0.5*(np.log10(x[1:])+np.log10(x[:-1])))
            
        x, y, y_std = x, y, y_std#np.log10(x), np.log10(y), np.log10(y_std)

        plt.errorbar(x, y, #yerr=y_std,
                label=r"$\mu$ = " + str(b) + " [simulation units]")
        plt.scatter(x, y)

        dr2ds_temp = pd.DataFrame({"dd2dt_dt_f_bin": 10**x,
                                "dd2dt_dd2_f_bin": 10**y, "b_dd2_dt_f_bin": b})
        dr2ds_7 = pd.concat([dr2ds_7, dr2ds_temp], ignore_index=True, axis=0)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta contour_c$ [simulation units]")
plt.legend(loc="best")
plt.title(r"$\Delta contour_c$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2dt_f_new_bin.png")
# # plt.show()
plt.close()

for b in dr2ds["b"].unique():
    x_d = np.array([])
    y_d = np.array([])
    for random_seed_pos_init in dr2ds["random_seed_pos_init"].unique():
        dr2ds_plot = dr2ds[(dr2ds["b"] == b) & (dr2ds["random_seed_pos_init"] == random_seed_pos_init)]

        x_data, y_data = dr2ds_plot["dTau2dt_dt_f"], dr2ds_plot["dTau2dt_dTau2_f"]
        x_data, y_data = x_data[~np.isnan(x_data)], y_data[~np.isnan(x_data)]
        x_data, y_data = x_data[~np.isnan(y_data)], y_data[~np.isnan(y_data)]
        x_data, y_data = npi.group_by(np.around(x_data)).mean(y_data)
        x_d = np.append(x_d, x_data)
        y_d = np.append(y_d, y_data)

    _, nx = npi.count(np.around(x_d))
    x_d, y_d = npi.group_by(np.around(x_d)).mean(y_d)
    y_d = y_d[nx>=nseeds]
    x_d = x_d[nx>=nseeds]
    x, y = x_d, y_d
    _, bins = np.histogram(np.log10(x_d), bins='auto')
    if y_d.shape[0] > 0:
        print(b)
        y, x, bnumbr = stats.binned_statistic(
            x_d, np.log10(y_d), statistic=np.nanmean, bins=10**bins)
        y_std, x_std, bnumbr = stats.binned_statistic(
            x_d, np.log10(y_d), statistic=np.nanstd, bins=10**bins)

        x = (0.5*(np.log10(x[1:])+np.log10(x[:-1])))
            
        x, y, y_std = x, y, y_std#np.log10(x), np.log10(y), np.log10(y_std)

        plt.errorbar(x, y, #yerr=y_std,
                label=r"$\mu$ = " + str(b) + " [simulation units]")
        plt.scatter(x, y)

        dr2ds_temp = pd.DataFrame({"dTau2udt_dt_f_bin": 10**x,
                                "dTau2udt_dTau2u_f_bin": 10**y, "dTau2udt_b_f_bin": b})
        dr2ds_8 = pd.concat([dr2ds_8, dr2ds_temp], ignore_index=True, axis=0)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta \sigma^2$ [simulation units]")
plt.legend(loc="best")
plt.title(r"$\Delta \sigma^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dTau2udt_f_new_bin.png")
# # plt.show()
plt.close()

for b in dr2ds["b"].unique():
    x_d = np.array([])
    y_d = np.array([])
    for random_seed_pos_init in dr2ds["random_seed_pos_init"].unique():
        dr2ds_plot = dr2ds[(dr2ds["b"] == b) & (dr2ds["random_seed_pos_init"] == random_seed_pos_init)]

        x_data, y_data = dr2ds_plot["displacement2"], dr2ds_plot["Tau2"]
        # print(b, x_data, y_data)
        x_data, y_data = x_data[~np.isnan(x_data)], y_data[~np.isnan(x_data)]
        x_data, y_data = x_data[~np.isnan(y_data)], y_data[~np.isnan(y_data)]
        x_data, y_data = np.mean(x_data), np.mean(y_data)
        x_d = np.append(x_d, x_data)
        y_d = np.append(y_d, y_data)

    x_d, y_d = np.mean(x_d), np.mean(y_d)
    print(b, x_d, y_d)

    dr2ds_temp = pd.DataFrame({"displacement2": np.array([x_d]),
                        "Tau2": np.array([y_d]), "b_mean": b})
    dr2ds_9 = pd.concat([dr2ds_9, dr2ds_temp], ignore_index=True, axis=0)

    plt.scatter(x_d, y_d, label=r"$\mu$ = " + str(b) + " [simulation units]")
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta \sigma^2$ [simulation units]")
plt.legend(loc="best")
plt.title(r"$\Delta \sigma^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "Tau2_v_displacement2.png")
# # plt.show()

for b in dr2ds["b"].unique():
    x_d = np.array([])
    y_d = np.array([])
    for random_seed_pos_init in dr2ds["random_seed_pos_init"].unique():
        dr2ds_plot = dr2ds[(dr2ds["b"] == b) & (dr2ds["random_seed_pos_init"] == random_seed_pos_init)]

        x_data = (dr2ds_plot["ddisplacment2"])
        x_data = x_data[~np.isnan(x_data)]
        x_data = np.sqrt(x_data)
        x_d = np.append(x_d, x_data)

    x_d = np.abs(x_d)
    x_d = x_d[x_d > 0]
    x_d = x_d[x_d > 5e-6]
    y_d, bins = np.histogram(np.log10(x_d), bins="auto")
    y_d, x_d = np.histogram(x_d, bins=10**bins, density=True)
    x_d = 10**((np.log10(x_d[:-1])+np.log10(x_d[1:]))/2)
    x, y = x_d, y_d
    if y_d.shape[0] > 0:
        print(b)
        plt.errorbar(np.log10(x), np.log10(y), #yerr=y_std,
            label=r"$\mu$ = " + str(b) + " [simulation units]")
        plt.scatter(np.log10(x), np.log10(y))

        dr2ds_temp = pd.DataFrame({"dR": x,
                                "P_dR": y, "dR_b": b})
        dr2ds_10 = pd.concat([dr2ds_10, dr2ds_temp], ignore_index=True, axis=0)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta \sigma^2$ [simulation units]")
plt.legend(loc="best")
plt.title(r"$\Delta \sigma^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "P(dR).png")
# # plt.show()
plt.close()

for b in dr2ds["b"].unique():
    x_d = np.array([])
    y_d = np.array([])
    for random_seed_pos_init in dr2ds["random_seed_pos_init"].unique():
        dr2ds_plot = dr2ds[(dr2ds["b"] == b) & (dr2ds["random_seed_pos_init"] == random_seed_pos_init)]
        
        x_data = (dr2ds_plot.loc[np.around(dr2ds_plot["dt_d_i"]) == 1]["ddisplacement_abs_i"])
        x_data = x_data[~np.isnan(x_data)]
        x_d = np.append(x_d, x_data)

    x_d = np.abs(x_d)
    x_d = x_d[x_d > 0]
    x_d = x_d[x_d > 5e-6]
    y_d, bins = np.histogram(np.log10(x_d), bins="auto")
    y_d, x_d = np.histogram(x_d, bins=10**bins, density=True)
    x_d = 10**((np.log10(x_d[:-1])+np.log10(x_d[1:]))/2)
    x, y = x_d, y_d
    if y_d.shape[0] > 0:
        print(b)
        plt.errorbar(np.log10(x), np.log10(y), #yerr=y_std,
            label=r"$\mu$ = " + str(b) + " [simulation units]")
        plt.scatter(np.log10(x), np.log10(y))

        dr2ds_temp = pd.DataFrame({"dx": x,
                                "P_dx": y, "dx_b": b})
        dr2ds_11 = pd.concat([dr2ds_11, dr2ds_temp], ignore_index=True, axis=0)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta \sigma^2$ [simulation units]")
plt.legend(loc="best")
plt.title(r"$\Delta \sigma^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "P(dx).png")
# # plt.show()
plt.close()

for b in dr2ds["b"].unique():
    dr2ds_plot = dr2ds.loc[(dr2ds["b"] == b)]
    x_data, y_data = dr2ds_plot["dTau2ds_ds_f"], dr2ds_plot["dTau2ds_dTau2_f"]
    x_data, y_data = x_data[~np.isnan(x_data)], y_data[~np.isnan(x_data)]
    x_data, y_data = x_data[~np.isnan(y_data)], y_data[~np.isnan(y_data)]
    y_data = y_data[x_data >= x_min]
    x_data = x_data[x_data >= x_min]
    _, bins = np.histogram(np.log10(x_data), bins=nbin)

    bin_centers_valid = np.ones(len(bins)-1)
    y_all = np.array([])
    for random_seed_pos_init in dr2ds["random_seed_pos_init"].unique():
        dr2ds_plot = dr2ds[(dr2ds["b"] == b) & (dr2ds["random_seed_pos_init"] == random_seed_pos_init)]

        x_data, y_data = dr2ds_plot["dTau2ds_ds_f"], dr2ds_plot["dTau2ds_dTau2_f"]
        x_data, y_data = x_data[~np.isnan(x_data)], y_data[~np.isnan(x_data)]
        x_data, y_data = x_data[~np.isnan(y_data)], y_data[~np.isnan(y_data)]

        # mak sure these bins have at least one data point from each seed
        count, _ = np.histogram((x_data), bins=10**bins)
        bin_centers_valid *= (count >= nseeds)

        y, x, bnumbr = stats.binned_statistic(
        x_data, np.log10(y_data), statistic=np.nanmean, bins=10**bins)

        if y_all.shape[0] == 0:
            y_all = np.array(y)
        else:
            y_all = np.vstack((y_all, y))

        x = 10**(0.5*(np.log10(x[1:])+np.log10(x[:-1])))

    y = np.nanmean(y_all, axis=0)
    y_std = np.nanstd(y_all, axis=0)
    x = x[bin_centers_valid == 1]
    y = y[bin_centers_valid == 1]
    y_std = y_std[bin_centers_valid == 1]

    y = 10**y
    y_std = 10**y_std

    x, y, y_std = np.log10(x), np.log10(y), np.log10(y_std)

    plt.errorbar(x, y, yerr=y_std,
             label=r"$\mu$ = " + str(b) + " [simulation units]")
    plt.scatter(x, y)

    if (b == 0.0):
        a, blim = 0.55, 1.05
        xp1 = x[(x >= a) & (x <= blim)]
        yp1 = y[(x >= a) & (x <= blim)]
        p, residuals, _, _, _ = np.polyfit(xp1, yp1, 1, full=True)
        px = np.poly1d(p)
        print("slope: dTau2dsall", p[0])
        y_pred = px(xp1)

        # Calculate the residuals
        residuals = yp1 - y_pred

        # Calculate the Total Sum of Squares (SST)
        sst = np.sum((yp1 - np.mean(yp1))**2)

        # Calculate the Residual Sum of Squares (SSR)
        ssr = np.sum(residuals**2)

        # Calculate R-squared
        r_squared = 1 - (ssr / sst)

        print(f"R-squared: {r_squared}")

        # p = np.polyfit(np.log10(xp1), np.log10(yp1), 1)
        xi = 10**xp1[0]#10**a#5.8e0
        xf = 10**xp1[-1]#10**b#2.8e1
        yi = 10**(p[1] + p[0]*np.log10(xi))
        slope = p[0]
        x_vals = np.array([xi, xf])
        c = np.log10(yi) - slope * np.log10(xi)
        c = 10**c
        y_vals = c * np.power(x_vals, slope)
        print(residuals)
        plt.plot(np.log10(x_vals), np.log10(y_vals), color="black",linewidth=1, zorder=10)
        plt.annotate(text='$' +  str(np.around(slope, 2)) +'$', xy=(0.75, -4), xycoords='data', fontsize=6)

    dr2ds_temp = pd.DataFrame({"dTau2ds_ds_f": 10**x,
                            "dTau2ds_dTau2_f": 10**y, "dTau2ds_b_f": b})
    dr2ds_12 = pd.concat([dr2ds_12, dr2ds_temp], ignore_index=True, axis=0)
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta {Tau}^2$ [simulation units]")
plt.legend(loc="best")
plt.title(r"$\Delta {Tau}^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dTau2ds_all.png")
# plt.show()
plt.close()

for b in dr2ds["b"].unique():
    x_d = np.array([])
    y_d = np.array([])
    for random_seed_pos_init in dr2ds["random_seed_pos_init"].unique():
        dr2ds_plot = dr2ds[(dr2ds["b"] == b) & (dr2ds["random_seed_pos_init"] == random_seed_pos_init)]

        x_data, y_data = dr2ds_plot["dcontourdt_dt"], dr2ds_plot["dcontourdt_dcontour"]
        x_data, y_data = x_data[~np.isnan(x_data)], y_data[~np.isnan(x_data)]
        x_data, y_data = x_data[~np.isnan(y_data)], y_data[~np.isnan(y_data)]
        x_data, y_data = npi.group_by(np.around(x_data)).mean(y_data)
        x_d = np.append(x_d, x_data)
        y_d = np.append(y_d, y_data)

    _, nx = npi.count(np.around(x_d))
    x_d, y_d = npi.group_by(np.around(x_d)).mean(y_d)
    y_d = y_d[nx>=nseeds]
    x_d = x_d[nx>=nseeds]
    x, y = x_d, y_d
    x, y, y_std = np.log10(x), np.log10(y), np.log10(y_std)

    plt.errorbar(x, y, #yerr=y_std,
            label=r"$\mu$ = " + str(b) + " [simulation units]")
    plt.scatter(x, y)

    dr2ds_temp = pd.DataFrame({"dszdt_dt_f": 10**x,
                            "dszdt_dsz_f": 10**y, "b_dsz_dt_f": b})
    dr2ds_13 = pd.concat([dr2ds_13, dr2ds_temp], ignore_index=True, axis=0)
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta contour_c$ [simulation units]")
plt.legend(loc="best")
plt.title(r"$\Delta contour_c$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddszdt_f_new.png")
# # plt.show()
plt.close()

dr2ds_mean = pd.concat([dr2ds_1, dr2ds_2, dr2ds_3, dr2ds_4, dr2ds_5, dr2ds_6, dr2ds_7, dr2ds_8, dr2ds_9, dr2ds_10, dr2ds_11, dr2ds_12, dr2ds_13], axis=1)

dr2ds_mean.to_csv(join(output_foldername,
                  dr2dstau_filename_final), index=False, sep='\t', na_rep='nan', mode='a', header=False)