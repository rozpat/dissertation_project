# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:08:05 2023

@author: Maga Sganga
"""
import pandas as pd
import numpy as np
import visualization as vis
import glob
import matplotlib.pyplot as plt

vinconHeaders = ['x', 'y', 'z', 'X.1', 'Y.1', 'Z.1']


def load_IMU_AberStrokeApp(path_total, path_subject, path_exercise, v=True):
    """
    Description
    """
    print('Import data from IMU')
    extension = ".csv"
    numberAccFiles = len(glob.glob1(path_total, "ACCELEROMETER*"))
    accDF = pd.DataFrame()
    gyrDF = pd.DataFrame()
    imuData = ["ACCELEROMETER", "GYROSCOPE"]

    for i in range(numberAccFiles):

        for item in imuData:
            auxFileName = path_total + item + '_hour0_minute' + str(i) + extension
            df = pd.read_csv(auxFileName, sep=",", index_col=False)
            df.insert(4, 'minute', i)
            if item == "ACCELEROMETER":
                accDF = pd.concat([accDF, df])
            else:
                gyrDF = pd.concat([gyrDF, df])

    # Convert ms to seconds
    accDF.timestamp = accDF.timestamp / 1000
    gyrDF.timestamp = gyrDF.timestamp / 1000

    # Add subject and excercise name to column in dataframe
    accDF = dataInColumns(accDF, path_subject, path_exercise, "acc")
    gyrDF = dataInColumns(gyrDF, path_subject, path_exercise, "gyr")

    # Save in new excels files
    accDF.to_csv(path_total + "Acc.csv", index=False)
    gyrDF.to_csv(path_total + "Gyr.csv", index=False)

    if v:
        vis.plot_timeSeries(accDF, False)
        # vis.plot_timeSeries(gyrDF,False)

    return (accDF, gyrDF)


def load_IMU_MatlabMobile_CalibrationTesting(path_total, path_subject, path_exercise, v=True):
    """
    This function uploads data from csv files acquired with Matlab Mobile and pre-processes with convertFileToCSV.m created by Maga.
    All data is supposed to be in the same folder and named similarly.
    """
    print('Import data from IMU')
    extension = ".csv"
    accDF = pd.DataFrame()
    gyrDF = pd.DataFrame()

    auxFileName = path_total + 'ACCELEROMETER' + extension
    accDF = pd.read_csv(auxFileName, sep=",", index_col=False)
    auxFileName = path_total + 'GYROSCOPE' + extension
    gyrDF = pd.read_csv(auxFileName, sep=",", index_col=False)

    # Drop the string timestamp column and rename the elapsed time column
    accDF.drop(columns=['Timestamp'], inplace=True)
    gyrDF.drop(columns=['Timestamp'], inplace=True)

    # Rename columns
    accDF.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'}, inplace=True)
    gyrDF.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'}, inplace=True)

    # Add subject and exercise name to column in dataframe
    accDF = dataInColumns(accDF, path_subject, path_exercise, "acc")
    gyrDF = dataInColumns(gyrDF, path_subject, path_exercise, "gyr")

    # Order columns
    columns_order = ['timestamp', 'subject', 'excercise', 'datatype', 'x', 'y', 'z']
    accDF = accDF[columns_order]
    gyrDF = gyrDF[columns_order]

    if v:
        vis.plot_timeSeries(accDF, False)
        # vis.plot_timeSeries(gyrDF,False)

    return (accDF, gyrDF)


def load_Vicon(path_total, path_subject, path_exercise, v=True):
    print('Import Vicon data')
    viconDataFile = path_total + 'vicon_Trajectories_100.csv'
    # import vicon data skkiping first 3 rows and no index
    viconDF = pd.read_csv(viconDataFile, sep=",", index_col=False, skiprows=3)
    # rename columns and drop unwanted columns and rows
    viconDF.rename(columns={'Frame': 'timestamp'}, inplace=True)
    viconDF.drop(viconDF.index[:1], inplace=True)
    viconDF = viconDF.drop(columns=['Sub Frame'])
    viconDF = viconDF.apply(pd.to_numeric)

    # create a list of all columns to plot later
    viconHeaders = viconDF.columns
    viconHeaders = viconHeaders[1:]
    viconHeaders = viconHeaders.values
    viconHeaders = viconHeaders.tolist()

    # Convert frames to s and mm to m
    viconDF.timestamp = viconDF.timestamp / 100
    for col in viconDF.columns:
        if col in viconHeaders:
            viconDF[col] = viconDF[col] / 1000

    # Add subject and excercise name to column in dataframe
    viconDF = dataInColumns(viconDF, path_subject, path_exercise, "vicon")

    if v:
        vis.plot_timeSeriesAll(viconDF, viconHeaders, False)

    return (viconDF)


def load_treadmill(path_total, path_subject, path_exercise, v=True):
    print('Import treadmill data')
    treadmillDataFile = path_total + 'treadmill.txt'
    treadmillDF = pd.DataFrame()
    treadmillDF = pd.read_csv(treadmillDataFile, sep="\t", index_col=False, skiprows=43)

    # Delete columns with string values on
    columnToDelete = ["Gait type", "Contact side", "Foot contact"]
    for col in treadmillDF.columns:
        if col in columnToDelete:
            treadmillDF = treadmillDF.drop(col, axis=1)

    treadmillDF = treadmillDF.apply(pd.to_numeric)
    treadmillDF.rename(columns={'Time (s)': 'timestamp'}, inplace=True)

    # Add subject and excercise name to column in dataframe
    treadmillDF = dataInColumns(treadmillDF, path_subject, path_exercise, "treadmill")

    if v:
        treadmillHeaders = ["CoPx lateral (m)", "CoPy fore-aft (m)"]
        vis.plot_timeSeriesAll(treadmillDF, treadmillHeaders, False)

    return treadmillDF


def dataInColumns(df, s, e, t):
    df.insert(1, 'subject', s.replace("/", ""))
    df.insert(2, 'excercise', e.replace("/", ""))
    df.insert(3, 'datatype', t)
    return df

    # rawTreadmillVariables[np.isnan(rawTreadmillVariables)] = 0
    # markers_track = {}
    # markers_track['Time'] = rawTreadmillVariables[:,0]
    # markers_track['CoPx lat'] = rawTreadmillVariables[:,13]
    # markers_track['CoPy AntPost'] = rawTreadmillVariables[:,14]

    # plt.figure()
    # plt.plot(normalTreadmillVariables[:,13],normalTreadmillVariables[:,14], 'b')
    # plt.title("Normal gait CoP")
    # plt.ylim(0.4,1.5)
    # plt.xlim(0.3,0.7)
    # plt.show()

    # return treadmillDF

