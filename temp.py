import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
# import tensorflow
import visualization as vis
import loadData
import synchronisation as sync
from scipy.signal import find_peaks, argrelmin

# FOLDER------------------------------------------------------------------------
print('Import data from files')
path_file = 'G:/My Drive/DISSERTATION/DATA/'
path_subfolder = 'Dane/'
path_subject = 'test_calibration/'
path_exercise = 'cal1/'

path_total = path_file + path_subfolder + path_subject + path_exercise

# Load data from different systems ----------------------------------------
accDF, gyrDF = loadData.load_IMU_MatlabMobile_CalibrationTesting(path_total, path_subject, path_exercise, False)
viconDF = loadData.load_Vicon(path_total, path_subject, path_exercise, False)
treadmillDF = loadData.load_treadmill(path_total, path_subject, path_exercise, False)



