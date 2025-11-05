'''
File: QR_RQ_analysis.py
Project: QRP_analysis
File Created: Monday, 5th August 2019 4:12:29 pm
Author: Amruthesh T (amru@seas.upenn.edu)
-----
Last Modified: Wednesday, 18th August 2021 1:00:24 am
Modified By: Amruthesh T (amru@seas.upenn.edu)
-----
Copyright (c) 2018 - 2019 Amru, University of Pennsylvania

Summary: Fill In
'''

#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from os import popen, makedirs, walk
from os.path import join, isfile, isdir, basename, dirname, exists

# rp - Quasistaitic ripening (QR) simulations
# rq - Random quench (RQ) simulations 

output_foldername = "output/"
graphs_foldername = "graphs/"
input_foldername = "input/"
input_filename = "init.input"
run_filename = "run_config_rp.txt"

dirs = [dirs for root, dirs, files in walk(".")]
dirs = np.array([dir for dir in dirs[0]])

for i, dir in np.ndenumerate(dirs):
    if isfile(join(dir, output_foldername, run_filename)):
        if isfile(join(dir, output_foldername, "U_RP_RQ.txt")):
            f = pd.read_csv(join(dir, output_foldername, run_filename), sep="\n",
                            header=None, skip_blank_lines=False)
            f = f[0].str.split(r"\s{2,}", expand=True)
            f.columns = ["value", "name"]

            b = str(f.loc[f["name"] == "b"]["value"].values[0])

            type_external_force = float(f.loc[f["name"] == "type_external_force"]["value"].values[0])

            type_relaxation = str(
                f.loc[f["name"] == "type_relaxation"]["value"].values[0])

            type_dynamics = str(
                f.loc[f["name"] == "type_dynamics"]["value"].values[0])

            type_MD_MC = str(f.loc[f["name"] == "type_MD_MC"]["value"].values[0])

            U_filename = str(f.loc[f["name"] == "U_filename"]["value"].values[0])

            Z_filename = str(f.loc[f["name"] == "Z_filename"]["value"].values[0])

            s_contour_filename = str(
                f.loc[f["name"] == "s_contour_filename"]["value"].values[0])

            R_filename = str(f.loc[f["name"] == "R_filename"]["value"].values[0])

            order_parameter_filename = str(
                f.loc[f["name"] == "order_parameter_filename"]["value"].values[0])

            r_ij_parameter_filename = str(
                f.loc[f["name"] == "r_ij_parameter_filename"]["value"].values[0])

            N_init = float(f.loc[f["name"] == "N_init"]["value"].values[0])

            dt_coarsening = float(
                f.loc[f["name"] == "dt_coarsening"]["value"].values[0])

            random_seed_pos_init = float(
                f.loc[f["name"] == "random_seed_pos_init"]["value"].values[0])

            type_distribution = int(float(
                f.loc[f["name"] == "type_distribution"]["value"].values[0]))

            U_temp = pd.read_csv(join(dir, output_foldername, U_filename), sep=r"\s+",
                                header=None, skiprows=1)
            U_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "U", "dU", "dU/U",
                            "N", "min_counter_SUCCESSFUL", "min_counter"] + (["F_ext", "r_i_0_j_0"] if (type_external_force > 0) else [])

            U_sq_temp = pd.read_csv(join(dir, output_foldername, "U_RP_RQ.txt"), sep=r"\s+",
                                    header=None, skiprows=1)
            U_sq_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "U", "dU", "dU/U",
                                "N", "min_counter_SUCCESSFUL", "min_counter", "U_min", "U_max", "F_max"]
                                
            Z_temp = pd.read_csv(join(dir, output_foldername, Z_filename),
                                sep=r"\s+", header=None, skiprows=1)
            Z_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "Nc >= 0", "Nc >= 1", "Nc >= 2",
                            "Nc >= 3", "Nc >= 4", "Nc >= 5", "Nc >= 6", "N_nr >= 0", "N_nr >= 1", "N_nr >= 2", "N_nr >= 3", "N_nr >= 4", "N_nr >= 5", "N_nr >= 6", "vol_frac", "N"]

            Z_sq_temp = pd.read_csv(join(dir, output_foldername, "Z_RP_RQ.txt"),
                                    sep=r"\s+", header=None, skiprows=1)
            Z_sq_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "Nc >= 0", "Nc >= 1", "Nc >= 2",
                                "Nc >= 3", "Nc >= 4", "Nc >= 5", "Nc >= 6", "N_nr >= 0", "N_nr >= 1", "N_nr >= 2", "N_nr >= 3", "N_nr >= 4", "N_nr >= 5", "N_nr >= 6", "vol_frac", "p", "N"]

            s_temp = pd.read_csv(join(dir, output_foldername, s_contour_filename),
                                 sep=r"\s+", header=None, skiprows=1)
            s_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "s_displacement", "s", "ds", "contour",
                              "dcontour", "N", "s_nr", "ds_nr", "N_nr"]
            
            s_sq_temp = pd.read_csv(join(dir, output_foldername, "s_RP_RQ.txt"),
                                    sep=r"\s+", header=None, skiprows=1)
            s_sq_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "s_displacement", "s", "ds", "contour",
                                 "dcontour", "N", "s_nr", "ds_nr", "N_nr"]

            R_temp = pd.read_csv(join(dir, output_foldername, R_filename),
                                sep=r"\s+", header=None, skiprows=1)
            R_temp.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "N",
                            "mass_init - mass_N", "<R>", "<R>^2", "sqrt(<R^2>)", "R_1", "R_2", "R_3", "R_12 < L"]

            U_temp["N_init"] = N_init
            U_temp["b"] = b
            U_temp["dt_coarsening"] = dt_coarsening
            U_temp["random_seed_pos_init"] = random_seed_pos_init
            U_temp["type_distribution"] = type_distribution
            U_temp["type_relaxation"] = type_relaxation
            U_temp["type_dynamics"] = type_dynamics
            U_temp["type_MD_MC"] = type_MD_MC
            U_temp["simkey"] = i[0]
            U_temp["U_N"] = U_temp["U"] / U_temp["N"]

            U_sq_temp["N_init"] = N_init
            U_sq_temp["b"] = b
            U_sq_temp["dt_coarsening"] = dt_coarsening
            U_sq_temp["random_seed_pos_init"] = random_seed_pos_init
            U_sq_temp["type_distribution"] = type_distribution
            U_sq_temp["type_relaxation"] = type_relaxation
            U_sq_temp["type_dynamics"] = type_dynamics
            U_sq_temp["type_MD_MC"] = type_MD_MC
            U_sq_temp["simkey"] = i[0]
            U_sq_temp["U_N"] = U_sq_temp["U"] / U_sq_temp["N"]

            Z_temp["N_init"] = N_init
            Z_temp["b"] = b
            Z_temp["dt_coarsening"] = dt_coarsening
            Z_temp["random_seed_pos_init"] = random_seed_pos_init
            Z_temp["type_distribution"] = type_distribution
            Z_temp["type_relaxation"] = type_relaxation
            Z_temp["type_dynamics"] = type_dynamics
            Z_temp["type_MD_MC"] = type_MD_MC
            Z_temp["simkey"] = i[0]

            Z_sq_temp["N_init"] = N_init
            Z_sq_temp["b"] = b
            Z_sq_temp["dt_coarsening"] = dt_coarsening
            Z_sq_temp["random_seed_pos_init"] = random_seed_pos_init
            Z_sq_temp["type_distribution"] = type_distribution
            Z_sq_temp["type_relaxation"] = type_relaxation
            Z_sq_temp["type_dynamics"] = type_dynamics
            Z_sq_temp["type_MD_MC"] = type_MD_MC
            Z_sq_temp["simkey"] = i[0]

            s_temp["N_init"] = N_init
            s_temp["b"] = b
            s_temp["dt_coarsening"] = dt_coarsening
            s_temp["random_seed_pos_init"] = random_seed_pos_init
            s_temp["type_distribution"] = type_distribution
            s_temp["type_relaxation"] = type_relaxation
            s_temp["type_dynamics"] = type_dynamics
            s_temp["type_MD_MC"] = type_MD_MC
            s_temp["simkey"] = i[0]

            s_sq_temp["N_init"] = N_init
            s_sq_temp["b"] = b
            s_sq_temp["dt_coarsening"] = dt_coarsening
            s_sq_temp["random_seed_pos_init"] = random_seed_pos_init
            s_sq_temp["type_distribution"] = type_distribution
            s_sq_temp["type_relaxation"] = type_relaxation
            s_sq_temp["type_dynamics"] = type_dynamics
            s_sq_temp["type_MD_MC"] = type_MD_MC
            s_sq_temp["simkey"] = i[0]

            R_temp["N_init"] = N_init
            R_temp["b"] = b
            R_temp["dt_coarsening"] = dt_coarsening
            R_temp["random_seed_pos_init"] = random_seed_pos_init
            R_temp["type_distribution"] = type_distribution
            R_temp["type_relaxation"] = type_relaxation
            R_temp["type_dynamics"] = type_dynamics
            R_temp["type_MD_MC"] = type_MD_MC
            R_temp["simkey"] = i[0]

            try:
                U
            except NameError:
                U = U_temp
            else:
                U = pd.concat([U, U_temp], ignore_index=True)

            try:
                U_sq
            except NameError:
                U_sq = U_sq_temp
            else:
                U_sq = pd.concat([U_sq, U_sq_temp], ignore_index=True)

            try:
                Z
            except NameError:
                Z = Z_temp
            else:
                Z = pd.concat([Z, Z_temp], ignore_index=True)

            try:
                Z_sq
            except NameError:
                Z_sq = Z_sq_temp
            else:
                Z_sq = pd.concat([Z_sq, Z_sq_temp], ignore_index=True)

            try:
                s
            except NameError:
                s = s_temp
            else:
                s = pd.concat([s, s_temp], ignore_index=True)

            try:
                s_sq
            except NameError:
                s_sq = s_sq_temp
            else:
                s_sq = pd.concat([s_sq, s_sq_temp], ignore_index=True)

            try:
                R
            except NameError:
                R = R_temp
            else:
                R = pd.concat([R, R_temp], ignore_index=True)

n = 2

for item in U[(U["status_flag"]).astype(int) == 0]["simkey"].unique():
    U_plot = U[(U["simkey"] == item) & (
        (U["status_flag"]).astype(int) == 0)]
    plt.plot(U_plot["t"], U_plot["N"],
             label="simkey = " + str(item) + " [simulation units]")
plt.xlabel("t [simulation units]")
plt.ylabel("N")
plt.title("N v/s time")
plt.legend(loc='best')
plt.savefig(output_foldername + graphs_foldername +
            "N_i_v_st" + ".png")
plt.show()
plt.close()

for item in U[(U["status_flag"]).astype(int) == 0]["simkey"].unique():
    U_plot = U[(U["simkey"] == item) & (
        (U["status_flag"]).astype(int) == 0)]
    plt.plot(U_plot["t"], U_plot["U"],
             label="simkey = " + str(item) + " [simulation units]")
plt.xlabel("t [simulation units]")
plt.ylabel("U [simulation units]")
plt.title("U v/s time")
plt.legend(loc='best')
plt.savefig(output_foldername + graphs_foldername +
            "U_i_v_st" + ".png")
plt.show()
plt.close()

for item in U[(U["status_flag"]).astype(int) == 0]["simkey"].unique():
    U_plot = U[(U["simkey"] == item) & (
        (U["status_flag"]).astype(int) == 0)]
    plt.semilogy(U_plot["t"], U_plot["dU/U"],
                 label="simkey = " + str(item) + " [simulation units]")
plt.xlabel("t [simulation units]")
plt.ylabel(r"$\frac{\Delta U}{U}$ [simulation units]")
plt.title(r"$\frac{\Delta U}{U}$ v/s time")
plt.legend(loc='best')
plt.savefig(output_foldername + graphs_foldername +
            "dU_U_i_v_st" + ".png")
plt.show()
plt.close()

for item in R[(R["status_flag"]).astype(int) == 0]["simkey"].unique():
    R_plot = R[(R["simkey"] == item) & (
        (R["status_flag"]).astype(int) == 0)]
    plt.plot(R_plot["t"], np.power(
        R_plot["<R>"], n), label="simkey = " + str(item) + " [simulation units]")
plt.xlabel("t [simulation units]")
plt.ylabel(r"<R>$^{2}$ [simulation units]")
plt.title(r"<R>$^{2}$ v/s time")
# plt.legend(loc='best')
plt.savefig(output_foldername + graphs_foldername +
            "R2_i_v_st" + ".png")
plt.show()
plt.close()

R2_1 = 0.49
R2_2 = 0.55
R2_3 = 0.6

random_seed_pos_init_1 = np.array([])
random_seed_pos_init_2 = np.array([])
random_seed_pos_init_3 = np.array([])
t_sq_1 = np.array([])
t_sq_2 = np.array([])
t_sq_3 = np.array([])
U_rp = np.array([])
U_rq = np.array([])
Z_rp = np.array([])
Z_rq = np.array([])
s_rp = np.array([])
s_rq = np.array([])
N_rp = np.array([])
N_rq = np.array([])
# contour_rp = np.array([])
# contour_rq = np.array([])
nFIRE_rp = np.array([])
nFIRE_rq = np.array([])

for item in R[(R["status_flag"]).astype(int) == 0]["simkey"].unique():
    R_plot = R[(R["simkey"] == item) & (
        (R["status_flag"]).astype(int) == 0)]
    random_seed_pos_init_1 = np.append(random_seed_pos_init_1, R_plot["random_seed_pos_init"].values[0])
    random_seed_pos_init_2 = np.append(random_seed_pos_init_2, R_plot["random_seed_pos_init"].values[0])
    random_seed_pos_init_3 = np.append(random_seed_pos_init_3, R_plot["random_seed_pos_init"].values[0])
    t_sq_1 = np.append(t_sq_1, (np.min(R_plot.loc[(R_plot["<R>^2"] >= R2_1)]["t"]) if R_plot.loc[(R_plot["<R>^2"] >= R2_1)].size > 0 else np.nan))
    t_sq_2 = np.append(t_sq_2, (np.min(R_plot.loc[(R_plot["<R>^2"] >= R2_2)]["t"]) if R_plot.loc[(R_plot["<R>^2"] >= R2_2)].size > 0 else np.nan))
    t_sq_3 = np.append(t_sq_3, (np.min(R_plot.loc[(R_plot["<R>^2"] >= R2_3)]["t"]) if R_plot.loc[(R_plot["<R>^2"] >= R2_3)].size > 0 else np.nan))

    print(R_plot["random_seed_pos_init"].values[0], t_sq_1[-1], t_sq_2[-1], t_sq_3[-1])
    # U at t_sq_1
    U_plot = U[(U["simkey"] == item) & (U["t"] == t_sq_1[-1]) & (U["status_flag"].astype(int) == 0)]
    print(R_plot["random_seed_pos_init"].values[0], U_plot["U"].values[0], U_plot["N"].values[0])
    U_rp = np.append(U_rp, U_plot["U"]/U_plot["N"])
    # the top first 20 of U/N
    U_rq_plot = U_sq[(U_sq["simkey"] == item) & (U_sq["status_flag"].astype(int) == 0)][0:19]
    U_rq = np.append(U_rq, U_rq_plot["U"]/U_rq_plot["N"])

    # same with Z
    Z_plot = Z[(Z["simkey"] == item) & (Z["t"] == t_sq_1[-1]) & (Z["status_flag"].astype(int) == 0)]
    Z_rp = np.append(Z_rp, Z_plot["Nc >= 4"].values[0])
    Z_rq_plot = Z_sq[(Z_sq["simkey"] == item) & (Z_sq["status_flag"] == 0)][0:19]
    Z_rq = np.append(Z_rq, Z_rq_plot["Nc >= 4"].values)

    # same with s
    s_plot = s[(s["simkey"] == item) & (s["t"] == t_sq_1[-1]) & (s["status_flag"].astype(int) == 0)]
    s_rp = np.append(s_rp, s_plot["ds"].values[0])
    s_rq_plot = s_sq[(s_sq["simkey"] == item) & (s_sq["status_flag"].astype(int) == 0)][0:19]
    s_rq = np.append(s_rq, s_rq_plot["ds"].values)

    N_rp = np.append(N_rp, s_plot["N"].values[0])
    N_rq = np.append(N_rq, s_rq_plot["N"].values)

    # same with nFIRE
    nFIRE_rp = np.append(nFIRE_rp, U_plot["min_counter"].values[0])
    nFIRE_rq = np.append(nFIRE_rq, U_rq_plot["min_counter"].values)

# output to file
df = pd.DataFrame()
df["random_seed_pos_init_1"] = random_seed_pos_init_1
df["t_sq_1"] = t_sq_1
df["random_seed_pos_init_2"] = random_seed_pos_init_2
df["t_sq_2"] = t_sq_2
df["random_seed_pos_init_3"] = random_seed_pos_init_3
df["t_sq_3"] = t_sq_3
# output to file with nan
df.to_csv(output_foldername + "t_sq.txt", index=False, sep="\t", na_rep="nan")

#histograms of U_sq vs U_rp
# U_rq -= 0.95e-5
plt.hist(U_rq, bins="auto", alpha=0.5, label="U_RQ", density=True)
plt.hist(U_rp, bins="auto", alpha=0.5, label="U_QR", density=True)
# annotate mean and std for both
plt.axvline(np.mean(U_rq), color='b', linestyle='dashed', linewidth=1)
plt.axvline(np.mean(U_rp), color='r', linestyle='dashed', linewidth=1)
plt.legend(loc='best')
plt.xlabel("U/N")
plt.ylabel("P(U/N)")
plt.savefig(output_foldername + graphs_foldername + "P_U_hist_RP_RQ.png")
plt.close()

# output mean and sts of above
print("U_sq mean: ", np.mean(U_rq), np.size(U_rq))
print("U_sq std: ", np.std(U_rq), np.size(U_rq))
print("U_rp mean: ", np.mean(U_rp), np.size(U_rp))
print("U_rp std: ", np.std(U_rp), np.size(U_rp))

# histograms of Z_sq vs Z_rp
plt.hist(Z_rq, bins="auto", alpha=0.5, label="Z_RQ", density=True)
plt.hist(Z_rp, bins="auto", alpha=0.5, label="Z_QR", density=True)
# annotate mean and std for both
plt.axvline(np.mean(Z_rq), color='b', linestyle='dashed', linewidth=1)
plt.axvline(np.mean(Z_rp), color='r', linestyle='dashed', linewidth=1)
plt.legend(loc='best')
plt.xlabel("Z")
plt.ylabel("P(Z)")
plt.savefig(output_foldername + graphs_foldername + "P_Z_hist_RP_RQ.png")
plt.close()

# output mean and sts of above
print("Z_sq mean: ", np.mean(Z_rq))
print("Z_sq std: ", np.std(Z_rq))
print("Z_rp mean: ", np.mean(Z_rp))
print("Z_rp std: ", np.std(Z_rp))

# print U and Z histogram data to text file

df = pd.DataFrame()
# add nans at the end of shroter array
df["U_sq"] = U_rq
# remodel U_rp to length of U_rq with nans
U_rp = np.append(U_rp, np.full(U_rq.size - U_rp.size, np.nan))
df["U_rp"] = U_rp
df["Z_sq"] = Z_rq
Z_rp = np.append(Z_rp, np.full(Z_rq.size - Z_rp.size, np.nan))
df["Z_rp"] = Z_rp
# add to file with nans
df.to_csv(output_foldername + "U_Z_RP_RQ_hist_new.txt", index=False, sep="\t", na_rep="nan")

_, bins_rq = np.histogram(np.log10(s_rq), bins="auto", density=True)
_, bins_rp = np.histogram(np.log10(s_rp), bins="auto", density=True)
# plot both histograms on same plot
plt.hist(s_rq, bins=10**bins_rq, alpha=0.5, label="s_RQ", density=True)
plt.hist(s_rp, bins=10**bins_rp, alpha=0.5, label="s_QR", density=True)
# annotate mean and std for both
plt.axvline(10**np.mean(np.log10(s_rq)), color='b', linestyle='dashed', linewidth=1)
plt.axvline(10**np.mean(np.log10(s_rp)), color='r', linestyle='dashed', linewidth=1)
plt.legend(loc='best')
plt.xlabel(r"$\Delta r$")
plt.ylabel(r"P($\Delta r$)")
plt.xscale("log")
plt.yscale("log")
plt.savefig(output_foldername + graphs_foldername + "P_s_hist_RP_RQ.png")
plt.close()

# output mean and sts of above
print("s_sq mean: ", np.mean(s_rq))
print("s_sq std: ", np.std(s_rq))
print("s_sq median: ", np.median(s_rq))
print("s_rp mean: ", np.mean(s_rp))
print("s_rp std: ", np.std(s_rp))
print("s_rp median: ", np.median(s_rp))

plt.hist(nFIRE_rq, bins="auto", alpha=0.5, label="nFIRE_RQ", density=True)
plt.hist(nFIRE_rp, bins="auto", alpha=0.5, label="nFIRE_QR", density=True)
# annotate mean and std for both
plt.axvline(10**np.mean(np.log10(nFIRE_rq)), color='b', linestyle='dashed', linewidth=1)
plt.axvline(10**np.mean(np.log10(nFIRE_rp)), color='r', linestyle='dashed', linewidth=1)
plt.legend(loc='best')
plt.xlabel(r"$n_{FIRE}$")
plt.ylabel(r"P($n_{FIRE}$)")
plt.savefig(output_foldername + graphs_foldername + "P_nFIRE_hist_RP_RQ.png")
plt.close()

# output mean and sts of above
print("nFIRE_rq mean: ", np.mean(nFIRE_rq))
print("nFIRE_rq std: ", np.std(nFIRE_rq))
print("nFIRE_rq median: ", np.median(nFIRE_rq))
print("nFIRE_rp mean: ", np.mean(nFIRE_rp))
print("nFIRE_rp std: ", np.std(nFIRE_rp))
print("nFIRE_rp median: ", np.median(nFIRE_rp))

# print info to file

df = pd.DataFrame()
df["s_sq"] = s_rq
s_rp = np.append(s_rp, np.full(s_rq.size - s_rp.size, np.nan))
df["s_rp"] = s_rp
df["nFIRE_sq"] = nFIRE_rq
nFIRE_rp = np.append(nFIRE_rp, np.full(nFIRE_rq.size - nFIRE_rp.size, np.nan))
df["nFIRE_rp"] = nFIRE_rp

df["N_sq"] = N_rq
N_rp = np.append(N_rp, np.full(N_rq.size - N_rp.size, np.nan))
df["N_rp"] = N_rp
# add to file with nans
df.to_csv(output_foldername + "s_contour_nFIRE_RP_RQ_hist.txt", index=False, sep="\t", na_rep="nan")