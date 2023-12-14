import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelmin
import temp
import synchronisation_v2 as sync

accDF = temp.accDF
viconDF = temp.viconDF
treadmillDF = temp.treadmillDF

accDF.to_csv('accDF.csv')
viconDF.to_csv('viconDF.csv')
treadmillDF.to_csv('treadmillDF.csv')

# treadmill peak timestams and values
treadmill_peak_timestamps, treadmill_peak_values = sync.detect_treadmill_peak(treadmillDF)

# vicon peak timestamps and values
lowest_values, timestamps_of_lowest_values = sync.vicon_local_minima(viconDF)

# Calculate the median time difference between the treadmill and vicon peaks timestamps
median_time_difference = sync.compute_median_time_difference(treadmill_peak_timestamps, timestamps_of_lowest_values)

# Adjust the treadmill timestamps so that they are aligned with the vicon timestamps
treadmillDF['timestamp'] -= median_time_difference

# Detect treadmill peaks after the time shift
treadmill_peak_timestamps_corrected, treadmill_peak_values_corrected = sync.detect_treadmill_peak(treadmillDF)

median_Ez3_5s = treadmillDF["Ez3 (bits)"][:5000].median()
signal_treadmill = np.abs(treadmillDF["Ez3 (bits)"][:16000] - median_Ez3_5s)

# Filter the data to show only between 5 and 16 seconds
mask_treadmill = (treadmillDF['timestamp'] >= 5) & (treadmillDF['timestamp'] <= 16)
mask_vicon = (viconDF['timestamp'] >= 5) & (viconDF['timestamp'] <= 16)

# Plotting the data
plt.figure(figsize=(12, 6))
plt.plot(treadmillDF['timestamp'][mask_treadmill], np.abs(treadmillDF["Ez3 (bits)"][mask_treadmill] - treadmillDF["Ez3 (bits)"][:5000].median()),
         label='Adjusted Treadmill Signal', alpha=0.7)
plt.scatter(treadmill_peak_timestamps_corrected, treadmill_peak_values_corrected, color='red', marker='x', label='Detected Treadmill Peaks', s=100)
plt.plot(viconDF['timestamp'][mask_vicon], viconDF['Z'][mask_vicon] * 4000, label='Vicon Data', alpha=0.7)
plt.scatter(timestamps_of_lowest_values, lowest_values * 4000, color='red', marker='x', label='Detected Vicon Minima')
plt.xlabel('Timestamp')
plt.ylabel('Signal Value')
plt.title('Detected Treadmill and Vicon peaks after alignment')
plt.legend()
plt.grid(True)
plt.show()

# Finding peaks for acceleration data
acc_slice, adjusted_signal, acc_peaks_indices, acc_peaks_timestamps = sync.acc_peaks(accDF)

# Plot acceleration signal and peaks
plt.figure(figsize=(12, 6))
plt.plot(acc_slice['timestamp'], adjusted_signal, label='Acceleration', color='green')
plt.scatter(acc_slice['timestamp'].iloc[acc_peaks_indices], adjusted_signal.iloc[acc_peaks_indices], color='red', marker='x',
            label='Acc peaks')
plt.xlabel('Timestamp')
plt.ylabel('Acceleration')
plt.title('Acc between 5 and 16 seconds')
plt.legend()
plt.grid(True)
plt.show()

clean_df_vicon_acc = sync.compute_acceleration_from_location(viconDF)

# Plot acceleration signal from vicon
plt.figure(figsize=(14, 6))
plt.plot(clean_df_vicon_acc['timestamp'], clean_df_vicon_acc['Z'], label='Acceleration', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration')
plt.title('Acceleration vs Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

vic_acc_slice, vic_adjusted_signal, vic_valid_peak_indices, vic_acc_peaks_timestamps = sync.vicon_acc_peaks_with_max_distance(clean_df_vicon_acc)

# Plot VICON and ACC unaligned acceleration signals and peaks
plt.figure(figsize=(12, 6))
plt.plot(vic_acc_slice['timestamp'], vic_adjusted_signal, label='Acceleration', color='green')
plt.scatter(vic_acc_slice['timestamp'].iloc[vic_valid_peak_indices], vic_adjusted_signal.iloc[vic_valid_peak_indices],
            color='red', marker='x', label='Acc peaks')
plt.plot(acc_slice['timestamp'], adjusted_signal, label='Acceleration', color='orange')
plt.scatter(acc_slice['timestamp'].iloc[acc_peaks_indices], adjusted_signal.iloc[acc_peaks_indices], color='blue', marker='x',
            label='Acc peaks')
plt.xlabel('Timestamp')
plt.ylabel('Signal Value')
plt.title('Acc and Vicon signals between 5 and 16 seconds')
plt.legend()
plt.grid(True)
plt.show()

median_time_difference = sync.compute_median_time_difference(vic_acc_peaks_timestamps, acc_peaks_timestamps)

clean_df_vicon_acc['timestamp'] -= median_time_difference
# Detect treadmill peaks using the correct function
acc_sliceV, adjusted_signalV, valid_peak_indicesV, acc_peaks_timestampsV = sync.vicon_acc_peaks_with_max_distance(clean_df_vicon_acc)

# Plot aligned VICON and ACC acceleration signals and peaks
plt.figure(figsize=(12, 6))
plt.plot(acc_sliceV['timestamp'], adjusted_signalV, label='Vicon Acceleration', color='green')
plt.scatter(acc_sliceV['timestamp'].iloc[valid_peak_indicesV], adjusted_signalV.iloc[valid_peak_indicesV], color='red', marker='x',
            label='Vicon acc peaks')
plt.plot(acc_slice['timestamp'], adjusted_signal, label='Acceleration', color='orange')
plt.scatter(acc_slice['timestamp'].iloc[acc_peaks_indices], adjusted_signal.iloc[acc_peaks_indices], color='blue', marker='x',
            label='Acc peaks')
plt.xlabel('Timestamp')
plt.ylabel('Signal Value')
plt.title('Acc and Vicon signals between {} and {} seconds'.format(5, 16))
plt.legend()
plt.grid(True)
plt.show()