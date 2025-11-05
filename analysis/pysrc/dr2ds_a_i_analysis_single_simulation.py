'''
File: dr2ds_a_i_analysis_single_simulation.py
Project: QRP_analysis
File Created: Friday, 3rd May 2019 4:28:03 pm
Author: Amruthesh T (amru@seas.upenn.edu)
-----
Last Modified: Thursday, 13th July 2023 5:54:57 pm
Modified By: Amruthesh T (amru@seas.upenn.edu)
-----
Copyright (c) 2018 - 2019 Amru, University of Pennsylvania

Summary: Fill In
'''
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join, isfile
from os import popen, makedirs, walk
import os
import itertools as it
from scipy import stats
import numpy_indexed as npi

output_foldername = "output/"
graphs_foldername = "graphs/"
input_foldername = "input/"
input_filename = "init.input"
run_filename = "run_config.txt"

d = 3
n_flag = 1
neighbor_update_flag = 1

N_particles = 0

def periodic_BC(x1, x2, L):
    return (x1 - x2) - np.multiply(L, np.round(np.divide((x1 - x2), L)))


def Fij(ri, rj, Ri, Rj, L):
    epsilon = 2
    alpha = 2
    r = periodic_BC(ri, rj, L)
    if np.linalg.norm(r) < (Ri + Rj):
        return epsilon / (Ri + Rj) * np.power(1.0 - np.linalg.norm(r) / (Ri + Rj), alpha - 1) * np.divide(r, np.linalg.norm(r))
    else:
        return np.zeros(d)


def persistent_ensemble():
    global N_particles
    data = utils_collection[i_max]
    n_all = np.max(data["p_idx"].values)
    flag_all = np.ones(n_all)
    for i_file_counter_all in utils_collection.keys():
        data = utils_collection[i_file_counter_all]
        for i in range(n_all):
            flag_all[i] *= data["z_flag"].values[i+1]
    
    for i_file_counter_all in utils_collection.keys():
        data = utils_collection[i_file_counter_all]
        for i in range(n_all):
            if (data["z_flag"].values[i+1] == 1):
                data["z_flag"].values[i+1] = flag_all[i]

    # print("N = " + str(np.sum(data["z_flag"] == 1)))
    N_particles = np.sum(data["z_flag"] == 1)
    # print("Persistent ensemble done with N = " + str(np.sum(flag_all == 1)))


def z_ensemble():
    update_neighbors(neighbor_update_flag)
    data = utils_collection[i_max]
    n_all = np.max(data["p_idx"].values)
    flag_all = np.ones(n_all)
    for i_file_counter_all in utils_collection.keys():
        data = utils_collection[i_file_counter_all]
        for i in range(n_all):
            if (data["Z"].values[i+1] < d+1) and (data["z_flag"].values[i+1] == 1):
                # data.loc[mask_i, "flag"] = 0
                data["z_flag"].values[i+1] = 0
            flag_all[i] *= data["z_flag"].values[i+1]
    
    for i_file_counter_all in utils_collection.keys():
        data = utils_collection[i_file_counter_all]
        for i in range(n_all):
            data["z_flag"].values[i+1] = flag_all[i]

    # print("z ensemble done with N = " + str(np.sum(flag_all == 1)))


def z_trimmed_ensemble():
    # persistent_ensemble()

    # z_ensemble()

    update_neighbors(neighbor_update_flag)

    for i_file_counter in utils_collection.keys():
        data = utils_collection[i_file_counter]
        N = np.sum(data["z_flag"] == 1)
        # print("Z trimming started with N = " + str(N))
        recurrsive_z_trimming(i_file_counter)
        N = np.sum(data["z_flag"] == 1)
        # print("Z trimming done with N = " + str(N))


#@jit
def recurrsive_z_trimming(i_file_counter):
    data = utils_collection[i_file_counter]
    if (len(data)):
        N_i = np.sum(data["z_flag"] == 1)
        # x = data["Z"].values
        # print(x[np.nonzero(x)])
        update_z(i_file_counter)
        N_f = np.sum(data["z_flag"] == 1)
        # x = data["Z"].values
        # print(x[np.nonzero(x)])
        print("N_i = " + str(N_i) + ", N_f = " + str(N_f) + ", with z = " + str((data["Z"].values[0])))
        if (N_f != N_i):
            recurrsive_z_trimming(i_file_counter)


def update_neighbors(neighbor_update_flag):
    for i_file_counter in utils_collection.keys():
        data = utils_collection[i_file_counter]
        data["z_flag"] = data["flag"]
        
    global n_flag
    if (n_flag == 0 or n_flag == 1) and neighbor_update_flag == 2:
        for i_file_counter in utils_collection.keys():
            data = utils_collection[i_file_counter]
            data["neighbors"] = data.apply(lambda x: [], axis=1)
            n = np.max(data["p_idx"].values)

            for i in range(n):
                if (data["flag"].values[i+1] == 1):
                    data["Z"].values[i+1] = 0
                    R_i = data["R"].values[i+1]
                    for j in range(i+1, n):
                        if (data["flag"].values[j+1] == 1):
                            R_j = data["R"].values[j+1]
                            dx = 0
                            for k in range(d):
                                ri = data["x_" + str(k+1)].values[i+1]
                                rj = data["x_" + str(k+1)].values[j+1]
                                L_box = (data["L_box_" + str(k+1)].values[i+1] + data["L_box_" + str(k+1)].values[j+1]) / 2.0
                                dx += periodic_BC(ri, rj, L_box)**2
                                
                            dx /= (R_i + R_j)**2
                            # print(dx)
                            if dx <= 1:
                                data["neighbors"].values[i+1].append(j+1)

        n_flag = 2


def update_z(i_file_counter):
    data = utils_collection[i_file_counter]
    n = np.max(data["p_idx"].values)

    for i in range(n):
        data["Z"].values[i+1] = 0
        if (data["p_idx"].values[i+1] != i+1):
            print("Error in indexing")
    data["Z"].values[0] = 0

    for i in range(n):
        if (data["z_flag"].values[i+1] == 1):
            neighbors = data["neighbors"].values[i+1]
            for j in neighbors:
                if (data["z_flag"].values[j] == 1):
                    data["Z"].values[i+1] += 1
                    data["Z"].values[j] += 1
    
    for i in range(n):
        if (data["Z"].values[i+1] < d+1) and (data["z_flag"].values[i+1] == 1):
            # data.loc[mask_i, "flag"] = 0
            data["z_flag"].values[i+1] = 0
        elif (data["z_flag"].values[i+1] == 1):
            data["Z"].values[0] += data["Z"].values[i+1]
    data["Z"].values[0] /= np.sum(data["z_flag"].values == 1)


def update_z_all():
    if n_flag == 2:
        for i_file_counter in utils_collection.keys():
            data = utils_collection[i_file_counter]
            n = np.max(data["p_idx"].values)

            for i in range(n):
                data["Z"].values[i+1] = 0
                if (data["p_idx"].values[i+1] != i+1):
                    print("Error in indexing")

            for i in range(n):
                if (data["z_flag"].values[i+1] == 1):
                    neighbors = data["neighbors"].values[i+1]
                    for j in neighbors:
                        if (data["z_flag"].values[j] == 1):
                            data["Z"].values[i+1] += 1
                            data["Z"].values[j] += 1
    else:
        print("No neighbors found")


def update_tau():
    """ update_neighbors(neighbor_update_flag) """
    for i_file_counter in utils_collection.keys():
        data = utils_collection[i_file_counter]
        n = np.max(data["p_idx"].values)

        k = 0
        for l in range(d):
            for m in range(d):
                # data.loc[mask_i, "Tau_" + str(l+m+1)] = 0
                data["Tau_z_" + str(k+1)] = 0.0
                k += 1


def update_s():
    for index, i_file_counter in np.ndenumerate(utils_collection_keys):
        data = utils_collection[i_file_counter]
        data["s_z"] = 0.0
        n = np.max(data["p_idx"].values)

        if index[0] > 0:
            s = 0.0
            data = utils_collection[i_file_counter]
            data_prev = utils_collection[utils_collection_keys[index[0]-1]]
            for i in range(n):
                # R_j = data.loc[mask_j, "R"].values[0]
                if (data["z_flag"].values[i+1] == 1):
                    for k in range(d):
                        dx = (data["dx_" + str(k+1)].values[i+1] - data_prev["dx_" + str(k+1)].values[i+1])
                        data["s_z"].values[i+1] = np.abs(dx) + data_prev["s_z"].values[i+1]
                        s += dx**2

            s = np.sqrt(s)
            data["s_z"].values[0] = s + data_prev["s_z"].values[0]

    # print("s updated")

f = pd.read_csv(join(output_foldername, run_filename), lineterminator="\n",
                header=None, skip_blank_lines=False)
f = f[0].str.split(r"\s{2,}", expand=True)
f.columns = ["value", "name"]

utils_filename = str(f.loc[f["name"] == "utils_filename"]["value"].values[0])

utils_extension_filename = ".dat"

U_filename = str(f.loc[f["name"] == "U_filename"]["value"].values[0])

type_relaxation = int(f.loc[f["name"] == "type_relaxation"]["value"].values[0])

output_mode = float(
    f.loc[f["name"] == "output_mode"]["value"].values[0])

dt_coarsening = float(
    f.loc[f["name"] == "dt_coarsening"]["value"].values[0])

COARSEN_STEP_MAX = int(
    f.loc[f["name"] == "COARSEN_STEP_MAX"]["value"].values[0])
utils_print_frequency = int(
    f.loc[f["name"] == "utils_print_frequency"]["value"].values[0])

U = pd.read_csv(join(output_foldername, U_filename), lineterminator="\n",
                header=None, skip_blank_lines=False, skiprows=1)
U = U[0].str.split("\s+", expand=True)

NUMBER_OF_COUNTERS = 1

if (type_relaxation == 3):
    if (output_mode == 5):
        U.columns = ["counter_system", "t", "counter", "counter_time", "state_flag", "U", "dU", "dU/U",
                     "N", "min_counter_SUCCESSFUL", "min_counter", "kbT"]
    else:
        U.columns = ["counter_system", "t", "counter", "counter_time", "state_flag", "U", "dU", "dU/U",
                     "N", "min_counter_SUCCESSFUL", "min_counter"]
elif (type_relaxation == 4):
    if (output_mode == 5):
        U.columns = ["counter_system", "t", "counter", "counter_time", "state_flag", "U", "dU",
                     "dU/U", "N", "min_counter_SUCCESSFUL", "min_counter",  "bias_U", "number of biases"]
    else:
        U.columns = ["counter_system", "t", "counter", "counter_time", "state_flag", "U", "dU",
                     "dU/U", "N", "min_counter_SUCCESSFUL", "min_counter",  "bias_U", "number of biases"]
else:
    if (output_mode == 5):
        U.columns = ["counter_system", "t", "counter", "counter_time", "state_flag", "U", "dU", "dU/U",
                     "N", "min_counter_SUCCESSFUL", "min_counter"]
    else:
        U.columns = ["counter_system", "t", "counter", "counter_time", "state_flag", "U", "dU", "dU/U",
                     "N", "min_counter_SUCCESSFUL", "min_counter"]

NUMBER_OF_COUNTERS = int(np.max(U["counter"]))

utils_collection = {}

i_file_counter = 0

dynamical_scaling_time = 180#180 #150

dynamical_scaling_time_max = 656#656 b == 0;657 b=0.0001; 625 b == 0.001; 483 b == 0.01; 361 b == 0.1; 233 b == 1#150

b = float(f.loc[f["name"] == "b"]["value"].values[0])

if b == 0:
    dynamical_scaling_time_max = 656
elif b == 0.0001:
    dynamical_scaling_time_max = 657
elif b == 0.001:
    dynamical_scaling_time_max = 625
elif b == 0.01:
    dynamical_scaling_time_max = 483
elif b == 0.1:
    dynamical_scaling_time_max = 361
elif b == 1:
    dynamical_scaling_time_max = 233

files = [files for root, folders, files in walk(output_foldername)]

for filename in files[0]:
    if utils_filename and utils_extension_filename in filename:

        i_file_counter += 1

        dtype = np.dtype([("system_counter", 'i4'), ("t", 'f8'), ("counter", 'i4'), ("counter_time", 'f8'), ("state_flag", 'f8'), ("N", 'i4'), ("vol_frac", 'f8')] + [("L_box_" + str(k+1), 'f8') for k in range(d)] + [("p_idx", 'i4'), ("r_idx", 'i4'), ("flag", 'i4'), ("R", 'f8')] + [("x_" + str(k+1), 'f8') for k in range(d)] + [("dx_" + str(k+1), 'f8') for k in range(d)] + [("deltax_" + str(k+1), 'f8') for k in range(d)] +
                         [("s", 'f8'), ("ds", 'f8'), ("contour", 'f8'), ("dcontour", 'f8'), ("U", 'f8'), ("dU", 'f8'), ("dU_U", 'f8'), ("Z", 'f8')] + [("Tau_" + str(k+1), 'f8') for k in range(d * d)])

        f = open(join(output_foldername, filename), "rb")
        f.seek(0)

        data = np.fromfile(f, dtype=dtype)
        # data["z_flag"] = 

        t = np.around(data["t"][0])
        state_flag = data["state_flag"][0]

        if (t > dynamical_scaling_time) and (t < dynamical_scaling_time_max) and (state_flag == 0):
            # print(t, state_flag)
            utils_collection[i_file_counter] = pd.DataFrame(data)

        f.close()

del data

# print("read data")

dt = np.array([])
dt_z = np.array([])
dcounter = np.array([])
dS = np.array([])
dS_z = np.array([])
dcontour = np.array([])
dDisplacement2 = np.array([])

dt_i = np.array([])
ddisplacement2_i = np.array([])
R = np.array([])

ddisplacement2_abs_i = np.array([])
R_abs = np.array([])

dt_d_i = np.array([])
ddisplacement_i = np.array([])
R_i = np.array([])

ddisplacement_abs_i = np.array([])
R_abs_i = np.array([])

dt_z_i = np.array([])
ddisplacement2_z_i = np.array([])
R_z = np.array([])

ddisplacement2 = np.array([])
dr2 = np.array([])
dTau2 = np.array([])
dTau2_tmin = np.array([])
dTau2_z = np.array([])

D2li = np.array([])
dTau2li = np.array([])
Ali = np.array([])

displacement2 = np.array([])
Tau2 = np.array([])
Tau2_z = np.array([])
Tau = np.array([])
Tau_z = np.array([])
Ai = np.array([])

dt_range = [1,10, 100]

t_min = 0.0
t_max = 0.0
i_min = 0
i_max = 0

times = np.array([])
utils_collection_keys = np.array([])

for i in utils_collection.keys():
    if (float(utils_collection[i]["t"][0]) > dynamical_scaling_time) & (float(utils_collection[i]["t"][0]) < dynamical_scaling_time_max):
        time = float(utils_collection[i]["t"][0])
        if t_min == 0.0:
            t_min = time
            t_max = time
            i_min = i
            i_max = i
        else:
            if time < t_min:
                t_min = time
                i_min = i
            if time > t_max:
                t_max = time
                i_max = i

        times = np.append(times, time)
        utils_collection_keys = np.append(utils_collection_keys, i)

utils_collection_keys = [x for _,x in sorted(zip(times, utils_collection_keys))]

z_ensemble()
# z_trimmed_ensemble()
persistent_ensemble()

update_tau()

update_s()

k_delta_t = 100
n_deta_t = 3.0
delta_t_min = 1.0
delta_t_max = (dynamical_scaling_time_max - dynamical_scaling_time + 1.0) / (k_delta_t / n_deta_t / N_particles + 1.0)
print("Delta t max = " + str(delta_t_max))
n_keys = len(utils_collection_keys)
# for index, (i, j) in enumerate(list(it.combinations(utils_collection.keys(), 2))):
# for index, (i, j) in enumerate(list(it.combinations(utils_collection_keys, 2))):

for i in utils_collection.keys():
    if (float(utils_collection[i]["t"][0]) > dynamical_scaling_time) & (float(utils_collection[i]["t"][0]) < dynamical_scaling_time_max):
        if (output_mode == 1):
            if (float(utils_collection[i]["state_flag"][0]) == 0):
                flag_check = (utils_collection[i]["flag"] != -1)
                flag_live_check = (utils_collection[i]["flag"] == 1)
                flag_z_check = (utils_collection[i]["z_flag"] == 1)

                ai = np.array(utils_collection[i]["R"])
                ai = ai[flag_z_check == 1]

                if np.size(ai) > 0:
                    Ai = ai.reshape((1, np.size(ai)))
                else:
                    Ai = np.concatenate((Ai, ai.reshape((1, np.size(ai)))), axis=0)

Ai = np.mean(Ai, axis=0)

for index_i in range(n_keys):
    for index_j in range(index_i+1, n_keys):
        i = utils_collection_keys[index_i]
        j = utils_collection_keys[index_j]
        if (float(utils_collection[i]["t"][0]) > dynamical_scaling_time) & (float(utils_collection[j]["t"][0]) > dynamical_scaling_time) & (float(utils_collection[i]["t"][0]) < dynamical_scaling_time_max) & (float(utils_collection[j]["t"][0]) < dynamical_scaling_time_max):
            delt = abs(float(utils_collection[j]["t"][0]) -
                    float(utils_collection[i]["t"][0]))

            delc = abs(float(utils_collection[j]["counter"][0]) -
                    float(utils_collection[i]["counter"][0]))

            if (output_mode == 1):
                if (float(utils_collection[i]["state_flag"][0]) == 0) & (float(utils_collection[j]["state_flag"][0]) == 0):

                    if (np.around(delt) < delta_t_max):
                        dt = np.append(dt, delt)

                        dcounter = np.append(dcounter, abs(float(utils_collection[j]["counter"][0]) -
                                                        float(utils_collection[i]["counter"][0])))

                        dcontour = np.append(dcontour, abs(float(utils_collection[j]["contour"][0]) -
                                                        float(utils_collection[i]["contour"][0])))

                    if (np.around(delt) > delta_t_min):
                        dt_z = np.append(dt_z, delt)
                        
                        dS = np.append(dS, abs(float(utils_collection[j]["s"][0]) -
                                            float(utils_collection[i]["s"][0])))
                        
                        dS_z = np.append(dS_z, abs(float(utils_collection[j]["s_z"][0]) -
                                            float(utils_collection[i]["s_z"][0])))
                    
                    flag_check = np.array(np.multiply(utils_collection[i]["flag"] != -1, utils_collection[j]["flag"] != -1))
                    flag_live_check = np.array(np.multiply(utils_collection[i]["flag"] == 1, utils_collection[j]["flag"] == 1))
                    flag_z_check = np.array(np.multiply(utils_collection[i]["z_flag"] == 1, utils_collection[j]["z_flag"] == 1))

                    flag_live_i_check = np.array(utils_collection[i]["flag"] == 1)
                    flag_live_j_check = np.array(utils_collection[j]["flag"] == 1)
                    # flag_live_max_check_d = np.repeat(np.reshape(flag_live_max_check, (1, np.size(flag_live_max_check))), d, axis=0)

                    d_i = np.array([(utils_collection[j]["dx_"+str(k+1)] - utils_collection[i]["dx_"+str(k+1)]) for k in range(d)])
                    d_l_i = (d_i[:, flag_z_check == 1])
                    # print(d_i.shape, d_l_i.shape, flag_live_max_check_d.shape)
                    d_l_i_m = np.mean(d_i[:, flag_check == 1], axis=1)
                    d_l_i_m = np.reshape(d_l_i_m, (d, 1))
                    d_l_i_nm = d_l_i # - d_l_i_m
                    d2_l_i = np.sum(np.power(d_l_i_nm, 2), axis=0)
                    # d2_i = d2_i[flag_check == 1]
                    # d2_l_i = d2_i[flag_live_max_check == 1]
                    radius_j, radius_i = np.array(utils_collection[j]["R"]), np.array(utils_collection[i]["R"])
                    a_i = (radius_i + radius_j)/2.0
                    a_l_i = a_i[flag_z_check == 1]
                    a_l_i_d = np.repeat(np.reshape(a_l_i, (1, np.size(a_l_i))), d, axis=0)
                    a_l_mean = (np.mean(radius_i[flag_live_i_check == 1]) + np.mean(radius_j[flag_live_j_check == 1]))/2.0 #np.mean(a_i[flag_live_check == 1])#(a_l_i)
                    # a_mean_z = np.mean(a2_i[z_check == 1])

                    a_l_ii = radius_i[flag_z_check == 1]
                    a_l_jj = radius_j[flag_z_check == 1]

                    dTau_i = np.array([(utils_collection[j]["Tau_"+str(k+1)] - utils_collection[i]["Tau_"+str(k+1)]) for k in [d, 2*d, 2*d+1]])
                    dTau_l_i = (dTau_i[:, flag_z_check == 1])
                    dTau_l_i_m = np.mean(dTau_i[:, flag_check == 1], axis=1)
                    dTau_l_i_m = np.reshape(dTau_l_i_m, (d, 1))
                    dTau_l_i_nm = dTau_l_i# - dTau_l_i_m
                    dTau2_l_i = np.sum(np.power(dTau_l_i_nm,2), axis=0)

                    dTau_z_i = np.array([(utils_collection[j]["Tau_z_"+str(k+1)] - utils_collection[i]["Tau_z_"+str(k+1)]) for k in [d, 2*d, 2*d+1]])
                    dTau_z_l_i = (dTau_z_i[:, flag_z_check == 1])
                    dTau_z_l_i_m = np.mean(dTau_z_i[:, flag_check == 1], axis=1)
                    dTau_z_l_i_m = np.reshape(dTau_z_l_i_m, (d, 1))
                    dTau_z_l_i_nm = dTau_z_l_i# - dTau_z_l_i_m
                    dTau2_z_l_i = np.sum(np.power(dTau_z_l_i_nm,2), axis=0)

                    if np.around(delt) in dt_range:
                        ddisplacement2_i = np.append(ddisplacement2_i, d2_l_i/a_l_mean**2)
                        R = np.append(R, a_l_i/a_l_mean)
                        delt_array = np.full_like(d2_l_i, delt)
                        dt_i = np.append(dt_i, delt_array)

                        # d2_l_i = (d_i.reshape(int(np.around(np.size(d_i)/d)), d)[flag_live_max_check == 1]).flatten()
                        ddisplacement2_abs_i = np.append(ddisplacement2_abs_i, d2_l_i)
                        R_abs = np.append(R_abs, a_l_i)
                        # delt_array = np.full_like(d_l_i, delt)
                        # dt_d_abs_i = np.append(dt_d_i, delt_array)

                        ddisplacement_i = np.append(ddisplacement_i, d_l_i.flatten()/a_l_mean)
                        R_i = np.append(R_i, a_l_i_d.flatten()/a_l_mean)
                        delt_array = np.full_like(d_l_i.flatten(), delt)
                        dt_d_i = np.append(dt_d_i, delt_array)

                        # print(dt_d_i.shape, ddisplacement_i.shape, R_i.shape, a_l_mean)

                        # d_l_i = (d_i.reshape(int(np.around(np.size(d_i)/d)), d)[flag_live_max_check == 1]).flatten()
                        ddisplacement_abs_i = np.append(ddisplacement_abs_i, d_l_i.flatten())
                        R_abs_i = np.append(R_abs_i, a_l_i_d.flatten())
                        # delt_array = np.full_like(d_l_i, delt)
                        # dt_d_abs_i = np.append(dt_d_abs_i, delt_array)

                    if (np.around(delt) < delta_t_max): 
                        ddisplacement2 = np.append(ddisplacement2, np.mean(d2_l_i))

                        dTau2 = np.append(dTau2, np.mean(dTau2_l_i))
                        dTau2_z = np.append(dTau2_z, np.mean(dTau2_z_l_i))

                    if (np.around(delt) > delta_t_min):
                        d2_i = d2_l_i #np.sum(np.power(d_l_i, 2), axis=0)
                        dDisplacement2 = np.append(dDisplacement2, np.sum(d2_i))
                        dTau2_tmin = np.append(dTau2_tmin, np.sum(dTau2_l_i))

                    # dr2 = np.append(dr2, np.mean(sum([np.power(np.multiply(periodic_BC(utils_collection[j]["x_"+str(k+1)], utils_collection[i]["x_"+str(k+1)], np.mean([utils_collection[j]["L_box"+str(k+1)][0], utils_collection[i]["L_box"+str(k+1)][0]])),
                    #                                                    flag_check), 2) for k in range(d)])))
                    dr2 = np.append(dr2, np.nan)

                    # d2_l_i, dTau2_l_i, a_l_ii, a_l_jj, a_l_i
                    if (np.around(delt) < delta_t_max):
                        if np.size(D2li) == 0:
                            D2li = d2_l_i.reshape((1, np.size(d2_l_i)))
                            dTau2li = dTau2_l_i.reshape((1, np.size(dTau2_l_i)))
                            Ali = Ai.reshape((1, np.size(Ai)))
                        else:
                            D2li = np.concatenate((D2li, d2_l_i.reshape((1, np.size(d2_l_i)))), axis=0)
                            dTau2li = np.concatenate((dTau2li, dTau2_l_i.reshape((1, np.size(dTau2_l_i)))), axis=0)
                            Ali = np.concatenate((Ali, Ai.reshape((1, np.size(Ai)))), axis=0)

dTau2dt_x, dTau2dt_y = npi.group_by(np.around(dt)).mean(dTau2li)  # dt, dTau2
ddisplacement2dt_x, ddisplacement2dt_y = npi.group_by(np.around(dt)).mean(D2li)  # dt, dDisplacement2
_, Ali = npi.group_by(np.around(dt)).mean(Ali)  # dt, A

print("dTau2dt_x shape: ", dTau2dt_x.shape, "dTau2dt_y shape: ", dTau2dt_y.shape, "Ali shape: ", Ali.shape)
f = pd.DataFrame({"dTau2dt_dt": list(dTau2dt_x),
                  "dTau2dt_dTau2": list(dTau2dt_y),
                  "ddisplacement2dt_dt": list(ddisplacement2dt_x),
                  "ddisplacement2dt_ddisplacement2": list(ddisplacement2dt_y),
                  "Ali": list(Ali)})

f.to_csv(join(output_foldername,
              "dr2ds_a_i_configuration_path_analysis.txt"), index=False, sep='\t', na_rep='nan')