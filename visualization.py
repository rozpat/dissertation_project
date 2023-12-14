# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:52:52 2023

@author: Maga Sganga
"""
import matplotlib.pyplot as plt
import pandas as pd


def plot_timeSeries(df, scatflag=True):
    if scatflag:
        # plt.figure()
        plt.scatter(df.timestamp / 10, df.y)
        plt.scatter(df.timestamp / 10, df.x)
        plt.scatter(df.timestamp / 10, df.z)
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        plt.show()

    # plt.figure()
    plt.plot(df.timestamp / 10, df.y, label="y")
    plt.plot(df.timestamp / 10, df.x, label="x")
    plt.plot(df.timestamp / 10, df.z, label="z")
    plt.title("Acceleration")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()


def plot_timeSeriesAll(df, headers, scatflag=True):
    plt.figure()
    for col in df.columns:
        if col in headers:
            if scatflag:
                plt.scatter(df['timestamp'], df[col])
            plt.plot(df['timestamp'], df[col], label=str(col))
    plt.title("CoM location")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.legend()
    plt.show()


def plot_CoMvsCoP(viconDF, treadmillDF, axis):
    # Merge data into a single data frame with only datapoints with the same timestamp
    df = viconDF.merge(treadmillDF, how="inner", on=["timestamp"])

    if axis == "lateral":
        plt.figure(0)
        plt.plot(viconDF['timestamp'], (viconDF['Y.1'] - viconDF['Y.1'].median()), label="CoM")
        plt.plot(treadmillDF['timestamp'], (treadmillDF['CoPx lateral (m)'] - treadmillDF['CoPx lateral (m)'].median()),
                 label="CoP")
        plt.title("Lateral Displacement")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        plt.legend()
        plt.show()

        # Plot the difference between CoP and CoM point to point
        plt.figure(1)
        df['Y.1'] = df['Y.1'] - df['Y.1'].median()
        df['CoPx lateral (m)'] = df['CoPx lateral (m)'] - df['CoPx lateral (m)'].median()
        df['Difference'] = df['Y.1'] - df['CoPx lateral (m)']
        plt.plot(df['timestamp'], df['Difference'])
        plt.title("Difference in Lateral Displacement")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        plt.legend()
        plt.show()

    else:
        plt.figure(2)
        plt.plot(viconDF['timestamp'], (viconDF['X.1'] - viconDF['X.1'].median()), label="CoM")
        plt.plot(treadmillDF['timestamp'],
                 (treadmillDF['CoPy fore-aft (m)'] - treadmillDF['CoPy fore-aft (m)'].median()), label="CoP")
        plt.title("Fore-Back Displacement")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        plt.legend()
        plt.show()

        # Plot the difference between CoP and CoM point to point
        plt.figure(3)
        df['X.1'] = df['X.1'] - df['X.1'].median()
        df['CoPy fore-aft (m)'] = df['CoPy fore-aft (m)'] - df['CoPy fore-aft (m)'].median()
        df['Difference'] = df['X.1'] - df['CoPy fore-aft (m)']
        plt.title("Diference in Fore-Back Displacement")
        plt.plot(df['timestamp'], df['Difference'])
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        plt.show()