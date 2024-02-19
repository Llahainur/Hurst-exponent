#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def __to_inc(x):
    incs = x[1:] - x[:-1]
    return incs


def __to_pct(x):
    pcts = x[1:] / x[:-1] - 1.
    return pcts


def __get_simplified_RS(series, kind):
    incs = __to_inc(series)
    R = max(series) - min(series)  # range in absolute values
    S = np.std(incs, ddof=1)

    if R == 0 or S == 0:
        return 0  # return 0 to skip this interval due the undefined R/S ratio
    return R / S


def Hurst(ser):
    # # Evaluate Hurst equation
    H, c, data = compute_Hc(ser, kind='price', simplified=True)

    f, ax = plt.subplots()
    ax.plot(data[0], c * data[0] ** H, color="deepskyblue")
    ax.scatter(data[0], data[1], color="purple")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time')
    ax.set_xlabel("H={:.4f}".format(H))
    ax.grid(True)
    plt.show()
    print("H={:.4f}".format(H))


def compute_Hc(series, kind="random_walk", min_window=10, max_window=None, simplified=True):
    ndarray_likes = [np.ndarray]
    if "pandas.core.series" in sys.modules.keys():
        ndarray_likes.append(pd.core.series.Series)

    # convert series to numpy array if series is not numpy array or pandas Series
    if type(series) not in ndarray_likes:
        series = np.array(series)

    if "pandas.core.series" in sys.modules.keys() and type(series) == pd.core.series.Series:
        if series.isnull().values.any():
            raise ValueError("Series contains NaNs")
        series = series.values  # convert pandas Series to numpy array
    elif np.isnan(np.min(series)):
        raise ValueError("Series contains NaNs")

    RS_func = __get_simplified_RS

    err = np.geterr()
    np.seterr(all='raise')

    max_window = max_window or len(series) - 1
    window_sizes = list(map(
        lambda x: int(10 ** x),
        np.arange(math.log10(min_window), math.log10(max_window), 0.25)))
    window_sizes.append(len(series))

    RS = []
    # Алгоритм Херста
    for w in window_sizes:
        rs = []
        for start in range(0, len(series), w):
            if (start + w) > len(series):
                break
            _ = RS_func(series[start:start + w], kind)
            if _ != 0:
                rs.append(_)
        RS.append(np.mean(rs))

    A = np.vstack([np.log10(window_sizes), np.ones(len(RS))]).T
    H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0]
    np.seterr(**err)

    c = 10 ** c
    return H, c, [window_sizes, RS]


if __name__ == '__main__':
    # Ефанов Иван Николаевич	2001	Фунт	2012	Франк

    series1 = np.genfromtxt("GBPCB_930101_201231.txt", delimiter=",")
    series2 = np.genfromtxt("CHFCB_930101_201231.txt", delimiter=",")
    ser1 = []
    ser2 = []
    # фильтрация
    for val in series1:
        if val[2] // 10000 == 2020:
            ser1.append(val[7])

    for val in series2:
        if val[2] // 10000 == 2020:
            ser2.append(val[7])

    print(ser1, ser2)

    plt.plot(ser1)
    Hurst(ser1)

    plt.plot(ser2)
    Hurst(ser2)
