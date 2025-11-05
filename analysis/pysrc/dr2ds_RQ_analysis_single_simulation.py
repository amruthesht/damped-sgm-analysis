'''
File: dr2ds_RQ_analysis_single_simulation.py
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
    for i_file_counter_all in utils_collection_keys:
        data = utils_collection[i_file_counter_all]
        for i in range(n_all):
            flag_all[i] *= data["z_flag"].values[i+1]
    
    for i_file_counter_all in utils_collection_keys:
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
    for i_file_counter_all in utils_collection_keys:
        data = utils_collection[i_file_counter_all]
        for i in range(n_all):
            if (data["Z"].values[i+1] < d+1) and (data["z_flag"].values[i+1] == 1):
                # data.loc[mask_i, "flag"] = 0
                data["z_flag"].values[i+1] = 0
            flag_all[i] *= data["z_flag"].values[i+1]
    
    for i_file_counter_all in utils_collection_keys:
        data = utils_collection[i_file_counter_all]
        for i in range(n_all):
            data["z_flag"].values[i+1] = flag_all[i]


def z_min_ensemble():
    update_neighbors(neighbor_update_flag)
    data = utils_collection[i_max]
    n_all = np.max(data["p_idx"].values)
    flag_all = np.ones(n_all)
    for i_file_counter_all in utils_collection_keys:
        if np.around(utils_collection[i_file_counter_all]["t"].values[0]) >= z_min_time:
            data = utils_collection[i_file_counter_all]
            for i in range(n_all):
                if (data["Z"].values[i+1] < d+1) and (data["z_flag"].values[i+1] == 1):
                    data["z_flag"].values[i+1] = 0
                flag_all[i] *= data["z_flag"].values[i+1]
    
    for i_file_counter_all in utils_collection_keys:
        data = utils_collection[i_file_counter_all]
        for i in range(n_all):
            data["z_flag"].values[i+1] = flag_all[i]


def z_trimmed_ensemble():
    update_neighbors(neighbor_update_flag)

    for i_file_counter in utils_collection_keys:
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
    for i_file_counter in utils_collection_keys:
        data = utils_collection[i_file_counter]
        data["z_flag"] = data["flag"]
        
    global n_flag
    if (n_flag == 0 or n_flag == 1) and neighbor_update_flag == 2:
        for i_file_counter in utils_collection_keys:
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
            data["z_flag"].values[i+1] = 0
        elif (data["z_flag"].values[i+1] == 1):
            data["Z"].values[0] += data["Z"].values[i+1]
    data["Z"].values[0] /= np.sum(data["z_flag"].values == 1)


def update_z_all():
    if n_flag == 2:
        for i_file_counter in utils_collection_keys:
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
    for i_file_counter in utils_collection_keys:
        data = utils_collection[i_file_counter]
        n = np.max(data["p_idx"].values)

        k = 0
        for l in range(d):
            for m in range(d):
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
                if (data["z_flag"].values[i+1] == 1):
                    for k in range(d):
                        dx = (data["dx_" + str(k+1)].values[i+1] - data_prev["dx_" + str(k+1)].values[i+1])
                        data["s_z"].values[i+1] = np.abs(dx) + data_prev["s_z"].values[i+1]
                        s += dx**2

            s = np.sqrt(s)
            data["s_z"].values[0] = s + data_prev["s_z"].values[0]


f = pd.read_csv(join(output_foldername, run_filename), lineterminator="\n",
                header=None, skip_blank_lines=False)
f = f[0].str.split(r"\s{2,}", expand=True)
f.columns = ["value", "name"]

utils_filename = str(f.loc[f["name"] == "utils_filename"]["value"].values[0])

utils_extension_filename = ".dat"

U_filename = str(f.loc[f["name"] == "U_filename"]["value"].values[0])

type_relaxation = int(f.loc[f["name"] == "type_relaxation"]["value"].values[0])

random_seed_pos_init = int(np.around(float(f.loc[f["name"] == "random_seed_pos_init"]["value"].values[0])))

print("Random seed = " + str(random_seed_pos_init))

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

if random_seed_pos_init == 1:
    dynamical_scaling_time = 20000#180#180 #150
elif random_seed_pos_init == 2:
    dynamical_scaling_time = 90000#180#180 #150
elif random_seed_pos_init == 3:
    dynamical_scaling_time = 60000#180#180 #150
elif random_seed_pos_init == 4:
    dynamical_scaling_time = 90000#180#180 #150
elif random_seed_pos_init == 5:
    dynamical_scaling_time = 20000#180#180 #150
elif random_seed_pos_init == 6:
    dynamical_scaling_time = 30000#180#180 #150
else:
    dynamical_scaling_time = 60000

dynamical_scaling_time_max = np.inf #152000#656#656 b == 0;657 b=0.0001; 625 b == 0.001; 483 b == 0.01; 361 b == 0.1; 233 b == 1#150

z_min_length = 0

state_flag_value = 0
if (output_mode == 1):
    state_flag_value = 0
elif (output_mode == 5):
    state_flag_value = 1

files = [files for root, folders, files in walk(output_foldername)]

for filename in files[0]:
    if utils_filename and utils_extension_filename in filename:

        i_file_counter += 1

        dtype = np.dtype([("system_counter", 'i4'), ("t", 'f8'), ("counter", 'i4'), ("counter_time", 'f8'), ("state_flag", 'f8'), ("N", 'i4'), ("vol_frac", 'f8')] + [("L_box_" + str(k+1), 'f8') for k in range(d)] + [("p_idx", 'i4'), ("r_idx", 'i4'), ("flag", 'i4'), ("R", 'f8')] + [("x_" + str(k+1), 'f8') for k in range(d)] + [("dx_" + str(k+1), 'f8') for k in range(d)] + [("deltax_" + str(k+1), 'f8') for k in range(d)] +
                         [("s", 'f8'), ("ds", 'f8'), ("contour", 'f8'), ("dcontour", 'f8'), ("U", 'f8'), ("dU", 'f8'), ("dU_U", 'f8'), ("Z", 'f8')] + [("Tau_" + str(k+1), 'f8') for k in range(d * d)])

        if (output_mode == 1):
            dtype = np.dtype([("system_counter", 'i4'), ("t", 'f8'), ("counter", 'i4'), ("counter_time", 'f8'), ("state_flag", 'f8'), ("N", 'i4'), ("vol_frac", 'f8')] + [("L_box_" + str(k+1), 'f8') for k in range(d)] + [("p_idx", 'i4'), ("r_idx", 'i4'), ("flag", 'i4'), ("R", 'f8')] + [("x_" + str(k+1), 'f8') for k in range(d)] + [("dx_" + str(k+1), 'f8') for k in range(d)] + [("deltax_" + str(k+1), 'f8') for k in range(d)] +
                            [("s", 'f8'), ("ds", 'f8'), ("contour", 'f8'), ("dcontour", 'f8'), ("U", 'f8'), ("dU", 'f8'), ("dU_U", 'f8'), ("Z", 'f8')] + [("Tau_" + str(k+1), 'f8') for k in range(d * d)])
        elif (output_mode == 5):
            dtype = np.dtype([("system_counter", 'i4'), ("counter_time", 'f8'), ("t", 'i4'), ("counter", 'f8'), ("state_flag", 'f8'), ("N", 'i4'), ("vol_frac", 'f8')] + [("L_box_" + str(k+1), 'f8') for k in range(d)] + [("p_idx", 'i4'), ("r_idx", 'i4'), ("flag", 'i4'), ("R", 'f8')] + [("x_" + str(k+1), 'f8') for k in range(d)] + [("dx_" + str(k+1), 'f8') for k in range(d)] + [("deltax_" + str(k+1), 'f8') for k in range(d)] +
                            [("s", 'f8'), ("ds", 'f8'), ("contour", 'f8'), ("dcontour", 'f8'), ("U", 'f8'), ("dU", 'f8'), ("dU_U", 'f8'), ("Z", 'f8')] + [("Tau_" + str(k+1), 'f8') for k in range(d * d)])

        f = open(join(output_foldername, filename), "rb")
        f.seek(0)

        data = np.fromfile(f, dtype=dtype)
        # data["z_flag"] = 

        t = np.around(data["t"][0])
        state_flag = data["state_flag"][0]

        if (t > dynamical_scaling_time) and (t < dynamical_scaling_time_max) and (state_flag == state_flag_value):
            # print(t, state_flag)
            utils_collection[i_file_counter] = pd.DataFrame(data)

        f.close()

del data

# print("read data")

dt = np.array([])
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

displacement2 = np.array([])
Tau2 = np.array([])
Tau2_z = np.array([])

dt_range = [1,10, 100]

t_min = 0.0
t_max = 0.0
i_min = 0
i_max = 0

times = np.array([])
utils_collection_keys = np.array([])

for i in utils_collection.keys():
    if (float(utils_collection[i]["t"][0]) < dynamical_scaling_time_max):
        time = float(utils_collection[i]["t"][0])
        times = np.append(times, time)

# dynamical_scaling_time = np.max(times) - dynamical_scaling_length
z_min_time = np.max(times) - z_min_length

# copy of all util keys
utils_collection_keys_all = np.array(list(utils_collection.keys()))

for i in utils_collection_keys_all:
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

        utils_collection_keys = np.append(utils_collection_keys, i)
    else:
        del utils_collection[i]

utils_collection_keys = [x for _,x in sorted(zip(times, utils_collection_keys))]

z_min_ensemble()
# z_trimmed_ensemble()
persistent_ensemble()

update_tau()

update_s()

k_delta_t = 100
n_deta_t = 3.0
delta_t_min = utils_print_frequency
delta_c_min = utils_print_frequency
time_max = np.max(times)
time_min = np.min(times)
delta_t_max = np.inf #(time_max - time_min + 1.0) / (k_delta_t * utils_print_frequency / (n_deta_t * N_particles) + 1.0)
print("Delta t max = " + str(delta_t_max))
n_keys = len(utils_collection_keys)
for index_i in range(n_keys):
    for index_j in range(index_i+1, n_keys):
        i = utils_collection_keys[index_i]
        j = utils_collection_keys[index_j]
        if (float(utils_collection[i]["t"][0]) > dynamical_scaling_time) & (float(utils_collection[j]["t"][0]) > dynamical_scaling_time) & (float(utils_collection[i]["t"][0]) < dynamical_scaling_time_max) & (float(utils_collection[j]["t"][0]) < dynamical_scaling_time_max):
            delt = abs(float(utils_collection[j]["t"][0]) -
                    float(utils_collection[i]["t"][0]))

            delc = abs(float(utils_collection[j]["counter"][0]) -
                    float(utils_collection[i]["counter"][0]))

            if (output_mode == 1) or (output_mode == 5):
                if (float(utils_collection[i]["state_flag"][0]) == state_flag_value) & (float(utils_collection[j]["state_flag"][0]) == state_flag_value):

                    if (np.around(delt) > delta_c_min) and (np.around(delt) < delta_t_max):
                        dt = np.append(dt, delt)

                        dcounter = np.append(dcounter, abs(float(utils_collection[j]["counter"][0]) -
                                                        float(utils_collection[i]["counter"][0])))

                        dcontour = np.append(dcontour, abs(float(utils_collection[j]["contour"][0]) -
                                                        float(utils_collection[i]["contour"][0])))

                    if (np.around(delt) > delta_c_min):
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

                        ddisplacement_i = np.append(ddisplacement_i, d_l_i.flatten()/a_l_mean)
                        R_i = np.append(R_i, a_l_i_d.flatten()/a_l_mean)
                        delt_array = np.full_like(d_l_i.flatten(), delt)
                        dt_d_i = np.append(dt_d_i, delt_array)

                        ddisplacement_abs_i = np.append(ddisplacement_abs_i, d_l_i.flatten())
                        R_abs_i = np.append(R_abs_i, a_l_i_d.flatten())
                    
                    if (np.around(delt) > delta_c_min) and (np.around(delt) < delta_t_max):
                        ddisplacement2 = np.append(ddisplacement2, np.mean(d2_l_i))

                        dTau2 = np.append(dTau2, np.mean(dTau2_l_i))
                        dTau2_z = np.append(dTau2_z, np.mean(dTau2_z_l_i))

                    if (np.around(delt) > delta_c_min):
                        d2_i = d2_l_i
                        dDisplacement2 = np.append(dDisplacement2, np.sum(d2_i))
                        dTau2_tmin = np.append(dTau2_tmin, np.sum(dTau2_l_i))

                    dr2 = np.append(dr2, np.nan)

for i in utils_collection_keys:
    if (float(utils_collection[i]["t"][0]) > dynamical_scaling_time) & (float(utils_collection[i]["t"][0]) < dynamical_scaling_time_max):
        if (output_mode == 1):
            if (float(utils_collection[i]["state_flag"][0]) == state_flag_value):
                flag_check = (utils_collection[i]["flag"] != -1)
                flag_live_check = (utils_collection[i]["flag"] == 1)
                flag_z_check = (utils_collection[i]["z_flag"] == 1)
                di = np.array([(utils_collection[i]["dx_"+str(k+1)] - utils_collection[i_min]["dx_"+str(k+1)]) for k in range(d)])
                d2_i = np.sum(np.power(di,2), axis=0)
                displacement2 = np.append(displacement2, np.mean(d2_i[flag_z_check == 1]))
                taui = np.array([(utils_collection[i]["Tau_"+str(k+1)]) for k in [1, d-1, 2*d-1]])
                taui_z = np.array([(utils_collection[i]["Tau_z_"+str(k+1)]) for k in [1, d-1, 2*d-1]])
                tau2_i = np.sum(np.power(taui,2), axis=0)
                tau2_z_i = np.sum(np.power(taui_z,2), axis=0)
                Tau2 = np.append(Tau2, np.mean(tau2_i[flag_z_check == 1]))
                Tau2_z = np.append(Tau2_z, np.mean(tau2_z_i[flag_z_check == 1]))

displacement2[:] = np.mean(displacement2)
Tau2[:] = np.mean(Tau2)
Tau2_z[:] = np.mean(Tau2_z)

if np.size(dTau2) > 0:
    dTau2dt_x, dTau2dt_y = npi.group_by(np.around(dt)).mean(dTau2)  # dt, dTau2
    dTau2dt_x_f, dTau2dt_y_f = dTau2dt_x, dTau2dt_y
    _, bins = np.histogram(np.log10(dTau2dt_x), bins='auto')
    dTau2dt_y, dTau2dt_x, bnumbr = stats.binned_statistic(
        dTau2dt_x, np.log10(dTau2dt_y), statistic='mean', bins=10**bins)
    dTau2dt_y = 10**dTau2dt_y
    dTau2dt_x = 10**(0.5*(np.log10(dTau2dt_x[1:])+np.log10(dTau2dt_x[:-1])))

if np.size(dTau2_z) > 0:
    dTau2dt_z_x, dTau2dt_z_y = npi.group_by(np.around(dt)).mean(dTau2_z)  # dt, dTau2
    dTau2dt_z_x_f, dTau2dt_z_y_f = dTau2dt_z_x, dTau2dt_z_y
    _, bins = np.histogram(np.log10(dTau2dt_z_x), bins='auto')
    dTau2dt_z_y, dTau2dt_z_x, bnumbr = stats.binned_statistic(
        dTau2dt_z_x, np.log10(dTau2dt_z_y), statistic='mean', bins=10**bins)
    dTau2dt_z_y = 10**dTau2dt_z_y
    dTau2dt_z_x = 10**(0.5*(np.log10(dTau2dt_z_x[1:])+np.log10(dTau2dt_z_x[:-1])))

dTau2ds_z_x, dTau2ds_z_y = dS_z, dTau2_tmin  # dt, dTau2
_, bins = np.histogram(np.log10(dTau2ds_z_x), bins='auto')
dTau2ds_z_y, dTau2ds_z_x, bnumbr = stats.binned_statistic(
    dTau2ds_z_x, np.log10(dTau2ds_z_y), statistic='mean', bins=10**bins)
dTau2ds_z_y = 10**dTau2ds_z_y
dTau2ds_z_x = 10**(0.5*(np.log10(dTau2ds_z_x[1:])+np.log10(dTau2ds_z_x[:-1])))
# print(dTau2ds_z_x, dTau2ds_z_y)

if np.size(dcounter) > 0:
    dcontourdt_x, dcontourdt_y = npi.group_by(np.around(dt)).mean(dcontour)
    dcontourdt_x, dcontourdt_y = dcontourdt_x, dcontourdt_y
    _, bins = np.histogram(np.log10(dcontourdt_x), bins='auto')
    dcontourdt_y, dcontourdt_x, bnumbr = stats.binned_statistic(
        dcontourdt_x, np.log10(dcontourdt_y), statistic='mean', bins=10**bins)
    dcontourdt_y = 10**dcontourdt_y
    dcontourdt_x = 10**(0.5 *
                        (np.log10(dcontourdt_x[1:])+np.log10(dcontourdt_x[:-1])))

if np.size(ddisplacement2) > 0:
    ddisplacement2dt_x, ddisplacement2dt_y = npi.group_by(np.around(dt)).mean(ddisplacement2)
    # dt, ddisplacement2
    ddisplacement2dt_x_f, ddisplacement2dt_y_f = ddisplacement2dt_x, ddisplacement2dt_y
    _, bins = np.histogram(np.log10(ddisplacement2dt_x), bins='auto')
    ddisplacement2dt_y, ddisplacement2dt_x, bnumbr = stats.binned_statistic(
        ddisplacement2dt_x, np.log10(ddisplacement2dt_y), statistic='mean', bins=10**bins)
    ddisplacement2dt_y = 10**ddisplacement2dt_y
    ddisplacement2dt_x = 10**(0.5 *
                            (np.log10(ddisplacement2dt_x[1:])+np.log10(ddisplacement2dt_x[:-1])))

# npi.group_by(dS).mean(dDisplacement2)
ddisplacement2ds_x, ddisplacement2ds_y = dS, dDisplacement2
_, bins = np.histogram(np.log10(ddisplacement2ds_x), bins='auto')
ddisplacement2ds_y, ddisplacement2ds_x, bnumbr = stats.binned_statistic(
    ddisplacement2ds_x, np.log10(ddisplacement2ds_y), statistic='mean', bins=10**bins)
ddisplacement2ds_y = 10**ddisplacement2ds_y
ddisplacement2ds_x = 10**(0.5 *
                          (np.log10(ddisplacement2ds_x[1:])+np.log10(ddisplacement2ds_x[:-1])))

# npi.group_by(dTau2).mean(dDisplacement2)
ddisplacement2dsz_x, ddisplacement2dsz_y = dS_z, dDisplacement2
# print(ddisplacement2dsz_x, ddisplacement2dsz_y)
_, bins = np.histogram(np.log10(ddisplacement2dsz_x), bins='auto')
ddisplacement2dsz_y, ddisplacement2dsz_x, bnumbr = stats.binned_statistic(
    ddisplacement2dsz_x, np.log10(ddisplacement2dsz_y), statistic='mean', bins=10**bins)
ddisplacement2dsz_y = 10**ddisplacement2dsz_y
ddisplacement2dsz_x = 10**(0.5 *
                            (np.log10(ddisplacement2dsz_x[1:])+np.log10(ddisplacement2dsz_x[:-1])))

f = pd.DataFrame({"dTau2dt_dt_f": dTau2dt_x_f,
                    "dTau2dt_dTau2_f": dTau2dt_y_f})
f_1 = pd.DataFrame({"dTau2dt_dt": dTau2dt_x, "dTau2dt_dTau2": dTau2dt_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dTau2_z_dt_f": dTau2dt_z_x_f,
                    "dTau2_z_dTau2_f": dTau2dt_z_y_f})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dTau2_z_dt": dTau2dt_z_x, "dTau2_z_dTau2": dTau2dt_z_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dTau2ds_ds_f": dS_z,
                    "dTau2ds_dTau2_f": dTau2_tmin})
f = pd.concat([f, f_1], axis=1)
# f_1 = pd.DataFrame({"dsdt_dt": dsdt_x, "dsdt_ds": dsdt_y})
# f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dcontourdt_dt": dcontourdt_x,
                    "dcontourdt_dcontour": dcontourdt_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2dt_dt_f": ddisplacement2dt_x_f,
                    "ddisplacement2dt_ddisplacement2_f": ddisplacement2dt_y_f})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2dt_dt": ddisplacement2dt_x,
                    "ddisplacement2dt_ddisplacement2": ddisplacement2dt_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2ds_ds": ddisplacement2ds_x,
                    "ddisplacement2ds_ddisplacement2": ddisplacement2ds_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2dsz_ds": ddisplacement2dsz_x,
                    "ddisplacement2dsz_ddisplacement2": ddisplacement2dsz_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2ds_ds_f": dS_z,
                    "ddisplacement2ds_ddisplacement2_f": dDisplacement2})
f = pd.concat([f, f_1], axis=1)
# f_1 = pd.DataFrame({"ddisplacement2dcontour_dcontour": ddisplacement2dcontour_x,
#                     "ddisplacement2dcontour_ddisplacement2": ddisplacement2dcontour_y})
# f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dt_i": dt_i, "ddisplacement2_i": ddisplacement2_i,
                    "R": R})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dt_abs_i": dt_i, "ddisplacement2_abs_i": ddisplacement2_abs_i,
                    "R_abs": R_abs})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dt_d_i": dt_d_i,
                    "ddisplacement_i": ddisplacement_i, "R_i": R_i})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dt_d_abs_i": dt_d_i,
                    "ddisplacement_abs_i": ddisplacement_abs_i, "R_abs_i": R_abs_i})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacment2": ddisplacement2})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"displacement2": displacement2,
                    "Tau2": Tau2})
f = pd.concat([f, f_1], axis=1)

f.to_csv(join(output_foldername,
              "dr2ds_RQ_configuration_path_analysis.txt"), index=False, sep='\t', na_rep='nan')