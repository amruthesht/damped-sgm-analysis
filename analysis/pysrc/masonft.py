'''
File: masonft.py
Project: QRP_analysis
File Created: Monday, 11th November 2019 9:05:45 pm
Author: Amruthesh T (amru@seas.upenn.edu)
-----
Last Modified: Sunday, 24th July 2022 9:38:16 pm
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
import math
from scipy.special import gamma

def masonft(x, y, w, der=2):
    x = np.array(x)
    y = np.array(y)

    q = np.power(x, -1)

    m, d, dd = logderive(x, y, w)

    if der == 2:
        Fs = np.power(q, -1)*m*gamma(1+d)*(1+(2*dd/np.pi))

        if max(abs(dd)) > 0.15:
            print('Warning, high curvature in data, moduli may be unreliable!')
    elif der == 1:
        Fs = np.power(q, -1)*y*gamma(1+d)

    return q, Fs

def masonift(q, F, w, der=1):
    q = np.array(q)
    F = np.array(F)

    x = np.power(q, -1)

    if der == 2:
        raise ValueError('Cannot perform inverse Fourier transform of 2nd derivative')
    elif der == 1:
        # solve for y in eq:F*q = y*gamma(1+x/y*y')
        # y' is the derivative of y with respect to x
        # code below is a solution to the above equation
        y = None



    

def npft(x, y):
    x = np.array(x)
    y = np.array(y)

    q = np.power(x, -1)

    Fs = (np.fft.fft(y))

    return q, Fs


def logderive(x, y, w):
    N = x.size
    df = np.zeros(N)
    ddf = np.zeros(N)
    f2 = np.zeros(N)
    lx = np.log(x)
    ly = np.log(y)

    for i in range(N):
        ww = np.exp(-np.power((lx-lx[i]), 2) / (2*w**2))
        res = polyfitw(lx, ly, ww, 2)
        f2[i] = np.exp(res[2] + res[1]*lx[i] + res[0]*(lx[i]**2))
        df[i] = res[1]+(2*res[0]*lx[i])
        ddf[i] = 2*res[0]

    return f2, df, ddf


def polyfitw(x, y, w, n):
    N = x.size
    x = np.array(x)
    y = np.array(y)
    w = np.array(w)
    W = np.diag(w)

    V = np.zeros((N, n+1))
    V[:, n] = np.ones(N)
    for i in reversed(range(n)):
        V[:, i] = x * V[:, i+1]

    coorm = np.matmul(np.matmul(np.transpose(V), W), V)

    b = np.matmul(np.matmul(np.transpose(V), W), y)

    p = np.matmul(np.linalg.inv(coorm), b)

    return p