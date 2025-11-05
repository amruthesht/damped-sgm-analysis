'''
File: dr2ds_a_i_analysis_ensemble.py
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
nbin = 3

output_foldername = "output/"
graphs_foldername = "graphs/"
input_foldername = "input/"
input_filename = "init.input"
run_filename = "run_config.txt"

dr2dstau_filename = "dr2ds_a_i_configuration_path_analysis.txt"

dr2dstau_filename_final = "dr2ds_a_i_configuration_path_analysis.txt"

dr2dstau_filename_overall = "dr2ds_configuration_path_analysis.txt"

dirs = [dirs for root, dirs, files in walk(".")]
dirs = np.array([dir for dir in dirs[0]])

def parse_array(s: str) -> np.ndarray:
    return np.fromstring(s.strip('[]'), sep=' ')

# tell pandas which columns to run through our parser
converters = {
    'dTau2dt_dTau2': parse_array,
    'ddisplacement2dt_ddisplacement2': parse_array,
    'Ali': parse_array,
}

colnames = ["dTau2dt_dt", "dTau2dt_dTau2", "ddisplacement2dt_dt", "ddisplacement2dt_ddisplacement2", "Ali"]

nseeds = 4  # Number of seeds to consider

for i, dir in np.ndenumerate(dirs):
    if isfile(join(dir, output_foldername, run_filename)) & isfile(join(dir, output_foldername, dr2dstau_filename)):
        f = pd.read_csv(join(dir, output_foldername, run_filename), lineterminator="\n",
                        header=None, skip_blank_lines=False)
        f = f[0].str.split(r"\s{2,}", expand=True)
        f.columns = ["value", "name"]

        b = float(f.loc[f["name"] == "b"]["value"].values[0])

        random_seed_pos_init = int(float(f.loc[f["name"] == "random_seed_pos_init"]["value"].values[0]))

        if (random_seed_pos_init <= nseeds) & (b >= 0.0):
            dr2ds_temp = pd.read_csv(join(dir, output_foldername, dr2dstau_filename), sep=r"\s+",
                                header=None, skiprows=1, names=colnames, converters=converters)

            dr2ds_temp.columns = ["dTau2dt_dt",	"dTau2dt_dTau2", "ddisplacement2dt_dt", "ddisplacement2dt_ddisplacement2", "Ali"]

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

dr2ds["dTau2dt_dt"] = np.round(dr2ds["dTau2dt_dt"]).astype(int)
dr2ds["ddisplacement2dt_dt"] = np.round(dr2ds["ddisplacement2dt_dt"]).astype(int)

Ali = np.array([])
A_edges = np.array([])
b_val_array = np.array([])

for b_val in dr2ds["b"].unique():
    seeds = dr2ds.loc[dr2ds["b"] == b_val, "random_seed_pos_init"].unique()
    Ali = np.array([])
    for seed in seeds:
        sub = dr2ds[
            (dr2ds["b"] == b_val) &
            (dr2ds["random_seed_pos_init"] == seed) &
            (dr2ds["dTau2dt_dt"] == 1)
        ]
        # extend the list by all matching Ali arrays
        Ali = np.append(Ali, np.hstack(sub["Ali"].values))

    a_edges = np.quantile(Ali, np.linspace(0, 1, nbin + 1))
    a_centers = (a_edges[:-1] + a_edges[1:]) / 2
    if A_edges.size == 0:
        A_edges = a_edges
        b_val_array = np.array([b_val])
    else:
        A_edges = np.vstack((A_edges, a_edges))
        b_val_array = np.append(b_val_array, b_val)

print(A_edges[b_val_array == 0.0001][0])
print(A_edges[b_val_array == 0.1][0])

for b_val in dr2ds["b"].unique():
    seeds = dr2ds.loc[dr2ds["b"] == b_val, "random_seed_pos_init"].unique()
    for seed in seeds:
        sub = dr2ds[
            (dr2ds["b"] == b_val) &
            (dr2ds["random_seed_pos_init"] == seed) &
            (dr2ds["dTau2dt_dt"] == 1)
        ]
        # print the number of ais in each bin
        # print(np.hstack(sub["Ali"].values))
        counts, _ = np.histogram(np.hstack(sub["Ali"].values), bins=A_edges[b_val_array == b_val][0])
        print(f"b: {b_val}, seed: {seed}, counts: {counts}")

# For each seed and b value, calculate the mean of Tau and displacement2 binning by Ali bin
dt_array = np.array([])
dx2_array = np.array([])
dTau2_array = np.array([])
b_array = np.array([])

for b_val in dr2ds["b"].unique():
    seeds = dr2ds.loc[dr2ds["b"] == b_val, "random_seed_pos_init"].unique()
    dt_dt = np.array([])
    dTau2dt_y_dt = np.array([])
    ddisplacement2dt_y_dt = np.array([])
    for seed in seeds:
        for dt in dr2ds[(dr2ds["b"] == b_val) & (dr2ds["random_seed_pos_init"] == seed)]["dTau2dt_dt"].unique():
            dr2ds_temp = dr2ds[(dr2ds["b"] == b_val) & (dr2ds["dTau2dt_dt"] == dt) & (dr2ds["random_seed_pos_init"] == seed)]
            dTau2dt_y, _, _ = stats.binned_statistic(
                np.hstack(dr2ds_temp["Ali"].values), np.hstack(dr2ds_temp["dTau2dt_dTau2"].values), statistic='mean', bins=A_edges[b_val_array == b_val][0])
            if dt_dt.size == 0:
                dt_dt = np.array([dt])
            else:
                dt_dt = np.hstack((dt_dt, np.array([dt])))
            if dTau2dt_y_dt.size == 0:
                dTau2dt_y_dt = dTau2dt_y
            else:
                dTau2dt_y_dt = np.vstack((dTau2dt_y_dt, dTau2dt_y))
            ddisplacement2dt_y, _, _ = stats.binned_statistic(
                np.hstack(dr2ds_temp["Ali"].values), np.hstack(dr2ds_temp["ddisplacement2dt_ddisplacement2"].values), statistic='mean', bins=A_edges[b_val_array == b_val][0])
            if ddisplacement2dt_y_dt.size == 0:
                ddisplacement2dt_y_dt = ddisplacement2dt_y
            else:
                ddisplacement2dt_y_dt = np.vstack((ddisplacement2dt_y_dt, ddisplacement2dt_y))

    _, nx = npi.count(np.around(dt_dt))
    dt_dt, ddisplacement2dt_y_dt = npi.group_by(np.around(dt_dt)).mean(ddisplacement2dt_y_dt, axis=0)
    _, dTau2dt_y_dt = npi.group_by(np.around(dt_dt)).mean(dTau2dt_y_dt, axis=0)

    dt_dt = dt_dt.reshape(-1, 1)
    dt_dt = dt_dt[nx>=nseeds]
    ddisplacement2dt_y_dt = ddisplacement2dt_y_dt[nx>=nseeds]
    dTau2dt_y_dt = dTau2dt_y_dt[nx>=nseeds]

    if dt_array.size == 0:
        dt_array = dt_dt
        dx2_array = ddisplacement2dt_y_dt
        dTau2_array = dTau2dt_y_dt
        # and b_val same number of times as the length of dt_dt and shape
        b_array = b_val * np.ones_like(dt_dt)
    else:
        dt_array = np.vstack((dt_array, dt_dt))
        dx2_array = np.vstack((dx2_array, ddisplacement2dt_y_dt))
        dTau2_array = np.vstack((dTau2_array, dTau2dt_y_dt))
        b_array = np.vstack((b_array, b_val * np.ones_like(dt_dt)))

dt_reg_array = np.array([])
dx2_reg_array = np.array([])
dTau2_reg_array = np.array([])
b_reg_array = np.array([])

for b_val in dr2ds["b"].unique():
    seeds = dr2ds.loc[dr2ds["b"] == b_val, "random_seed_pos_init"].unique()
    dt_dt = np.array([])
    dTau2dt_y_dt = np.array([])
    ddisplacement2dt_y_dt = np.array([])
    for seed in seeds:
        dr2ds_temp = dr2ds[(dr2ds["b"] == b_val) & (dr2ds["random_seed_pos_init"] == seed)]
        dt = np.vstack(dr2ds_temp["dTau2dt_dt"].values)
        dTau2dt_y = np.vstack(dr2ds_temp["dTau2dt_dTau2"].values)
        ddisplacement2dt_y = np.vstack(dr2ds_temp["ddisplacement2dt_ddisplacement2"].values)

        # dt = np.mean(dt, axis=1)
        dTau2dt_y = np.mean(dTau2dt_y, axis=1).reshape(-1, 1)
        ddisplacement2dt_y = np.mean(ddisplacement2dt_y, axis=1).reshape(-1, 1)

        if dt_dt.size == 0:
            dt_dt = dt
            dTau2dt_y_dt = dTau2dt_y
            ddisplacement2dt_y_dt = ddisplacement2dt_y
        else:
            dt_dt = np.vstack((dt_dt, dt))
            dTau2dt_y_dt = np.vstack((dTau2dt_y_dt, dTau2dt_y))
            ddisplacement2dt_y_dt = np.vstack((ddisplacement2dt_y_dt, ddisplacement2dt_y))

    _, nx = npi.count(np.around(dt_dt))
    _, ddisplacement2dt_y_dt = npi.group_by(np.around(dt_dt)).mean(ddisplacement2dt_y_dt, axis=0)
    dt_dt, dTau2dt_y_dt = npi.group_by(np.around(dt_dt)).mean(dTau2dt_y_dt, axis=0)

    dt_dt = dt_dt[nx>=nseeds]
    ddisplacement2dt_y_dt = ddisplacement2dt_y_dt[nx>=nseeds]
    dTau2dt_y_dt = dTau2dt_y_dt[nx>=nseeds]

    if dt_reg_array.size == 0:
        dt_reg_array = dt_dt
        dx2_reg_array = ddisplacement2dt_y_dt
        dTau2_reg_array = dTau2dt_y_dt
        # and b_val same number of times as the length of dt_dt and shape
        b_reg_array = b_val * np.ones_like(dt_dt)
    else:
        dt_reg_array = np.vstack((dt_reg_array, dt_dt))
        dx2_reg_array = np.vstack((dx2_reg_array, ddisplacement2dt_y_dt))
        dTau2_reg_array = np.vstack((dTau2_reg_array, dTau2dt_y_dt))
        b_reg_array = np.vstack((b_reg_array, b_val * np.ones_like(dt_dt)))

# Save the data
f1 = pd.DataFrame({
    "dt": list(dt_array),
    "ddisplacement2dt": list(dx2_array),
    "dTau2dt": list(dTau2_array),
    "b": list(b_array)
})

f2 = pd.DataFrame({
    "dt": list(dt_reg_array),
    "ddisplacement2dt": list(dx2_reg_array),
    "dTau2dt": list(dTau2_reg_array),
    "b": list(b_reg_array)
})

f1.to_csv(join(output_foldername, dr2dstau_filename_final), sep="\t", index=False)
f2.to_csv(join(output_foldername, "dr2dstau_master_z_ds_t_trim_trim_new_lengths_zero_mean_a_i_final_reg.txt"), sep="\t", index=False)

dr2dstau_filename = "dr2dstau_master_z_ds_t_trim_trim_new_lengths_zero_mean_final.txt"

dr2ds_overall = pd.read_csv(join(output_foldername, dr2dstau_filename), sep=r"\s+",
                                header=None, skiprows=0)

dr2ds_overall.columns = ["dd2ds_ds", "dd2ds_dd2", "dd2ds_b", "dd2ds_ds_f", "dd2ds_dd2_f", "dd2ds_b_f", "dd2dt_dt", "dd2dt_dd2", "dd2dt_b", "dTau2udt_dt", "dTau2udt_dTau2u", "dTau2udt_b", "dd2dt_dt_f", "dd2dt_dd2_f", "dd2dt_b_f", "dTau2udt_dt_f", "dTau2udt_dTau2u_f", "dTau2udt_b_f", "dd2dt_dt_f_bin", "dd2dt_dd2_f_bin", "dd2dt_b_f_bin", "dTau2udt_dt_f_bin", "dTau2udt_dTau2u_f_bin", "dTau2udt_b_f_bin", "displacement2", "Tau2", "b_mean", "dR", "P_dR", "dR_b", "dx", "P_dx", "dx_b", "dTau2ds_ds_f", "dTau2ds_dTau2_f", "dTau2ds_b_f", "dszdt_dt_f", "dszdt_dsz_f", "b_dsz_dt_f"]
# Plotting
for b_val in dr2ds["b"].unique():
    dr2ds_plot = dr2ds_overall.loc[(dr2ds_overall["dd2dt_b_f"] == b_val)]
    plt.figure(figsize=(10, 6))
    for i in range(0, nbin):
        # print(dt_array.shape, b_array.shape, dx2_array.shape)
        plt.plot(dt_array[b_array == b_val], dx2_array[b_array.flatten() == b_val][:, i], 'o', label=f'A_i bin {i+1} ({A_edges[b_val_array == b_val][0][i]} - {A_edges[b_val_array == b_val][0][i+1]})')
    x, y = dt_reg_array[b_reg_array == b_val], dx2_reg_array[b_reg_array.flatten() == b_val]
    # sort x and and y by x
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    plt.plot(x, y, 'k-', label='Mean')
    plt.plot(dr2ds_plot["dd2dt_dt_f"], dr2ds_plot["dd2dt_dd2_f"], 'r-', label='Overall Mean')
    plt.xlabel('dt')
    plt.ylabel('ddisplacement2dt')
    plt.title(f'b = {b_val}')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(join(output_foldername, graphs_foldername, f'displacement2dt_vs_dt_b_{b_val}.png'))
    plt.close()

for b_val in dr2ds["b"].unique():
    plt.figure(figsize=(10, 6))
    dr2ds_plot = dr2ds_overall.loc[(dr2ds_overall["dTau2udt_b_f"] == b_val)]
    for i in range(0, nbin):
        plt.plot(dt_array[b_array == b_val], dTau2_array[b_array.flatten() == b_val][:, i], 'o', label=f'A_i bin {i+1} ({A_edges[b_val_array == b_val][0][i]} - {A_edges[b_val_array == b_val][0][i+1]})')
    x, y = dt_reg_array[b_reg_array == b_val], dTau2_reg_array[b_reg_array.flatten() == b_val]
    # sort x and and y by x
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    plt.plot(x, y, 'k-', label='Mean')
    plt.plot(dr2ds_plot["dTau2udt_dt_f"], dr2ds_plot["dTau2udt_dTau2u_f"], 'r-', label='Overall Mean')
    plt.xlabel('dt')
    plt.ylabel('dTau2dt')
    plt.title(f'b = {b_val}')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(join(output_foldername, graphs_foldername, f'dTau2dt_vs_dt_b_{b_val}.png'))
    plt.close()