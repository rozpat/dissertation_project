import numpy as np
import pandas as pd
from scipy.signal import find_peaks, argrelmin
import matplotlib.pyplot as plt


def sync_vicon_treadmill(df_vicon, df_treadmill, v, name):
    ''' Synchronise vicon and treadmill data.
    The function returns the median time difference between vicon and treadmill peaks occurence and
    plots the results if v = True'''
    try:
        # Find peaks in the treadmill signal
        median_Ez3_5s = df_treadmill["Ez3 (bits)"][:5000].median()
        signal_treadmill = np.abs(df_treadmill["Ez3 (bits)"][:16000] - median_Ez3_5s)

        # Find the peaks for vicon that overcome 500 bits and has a minimun distance between each other of 0,5 sec
        peaks, _ = find_peaks(signal_treadmill, height=500, distance=500)

        # Extract corresponding timestamps for the peaks
        treadmill_peak_timestamps = np.array(df_treadmill['timestamp'].iloc[peaks])  # Array so I'm able to substract the timestamp values
        treadmill_peak_timestamps = np.sort(treadmill_peak_timestamps)

        # Find local minima in the Vicon data
        vicon_local_minima_index = argrelmin(df_vicon[:16000]['Z'].values)[0]
        vicon_minima_timestamps = np.array(df_vicon[:16000]['timestamp'].iloc[vicon_local_minima_index])
        # vicon_minima_values = np.array(viconDF[:16000]['Z'].iloc[vicon_local_minima_index])

        # Calculate time intervals between consecutive timestamps
        time_intervals = np.diff(vicon_minima_timestamps)  # I needed it to see what to do with the data and what is the pattern in taps

        # Define the threshold for time intervals
        threshold = 0.1  # I set it after checking time intervals

        # Initialize lists to store the selected timestamps
        selected_timestamps = []

        # Add the first value from vicon_minima_timestamps
        selected_timestamps.append(vicon_minima_timestamps[0])

        # Iterate through time intervals and select the first timestamp after each interval exceeds the threshold
        prev_timestamp = vicon_minima_timestamps[0]
        for interval, timestamp in zip(time_intervals, vicon_minima_timestamps[1:]):
            if interval > threshold:
                selected_timestamps.append(timestamp)
            prev_timestamp = timestamp

        # Find the indices where the selected_timestamps values would be inserted
        insert_indices = np.searchsorted(vicon_minima_timestamps, selected_timestamps)

        # Use these indices to index the vicon_local_minima_index array
        matching_indices = vicon_local_minima_index[insert_indices]

        vicon_minima_values_updated = np.array(df_vicon[:16000]['Z'].iloc[matching_indices])

        # Filtered values
        filtered_vicon_minima_values = vicon_minima_values_updated[vicon_minima_values_updated <= -0.005]

        # Find the indices where vicon_minima_values_updated is <= 0
        matching_indices_filtered = np.where(vicon_minima_values_updated <= -0.005)[0]

        # Use the indices to filter the selected_timestamps array
        matching_timestamps_filtered = np.array(selected_timestamps)[matching_indices_filtered]

        # ---------- CODE ADDED--------------------
        # Sort indices
        sorted_indices = np.argsort(filtered_vicon_minima_values)

        # Get the indices of the first 5 lowest values
        lowest_indices = sorted_indices[:5]

        # Use the lowest_indices to get the corresponding values
        lowest_values = filtered_vicon_minima_values[lowest_indices]

        # Use the lowest_indices to get the corresponding timestamps
        timestamps_of_lowest_values = np.sort(matching_timestamps_filtered[lowest_indices])
        # ------------- END OF UPDATE---------------

        # Time difference TREADMILL - VICON
        time_diff_treadmill_vicon = (treadmill_peak_timestamps - timestamps_of_lowest_values)  # That might not be useful
        # MEDIAN
        time_diff_treadmill_vicon_median = "{:.3f}".format(
            np.median(time_diff_treadmill_vicon))  # median as float with 3 dec places
        # Change str to float
        time_diff_treadmill_vicon_median = float(time_diff_treadmill_vicon_median)

        if v == True:
            x_min = 5
            x_max = 16
            # y_min = -200
            # y_max = 1700
            plt.xlim(x_min, x_max)
            # plt.ylim(y_min, y_max)

            plt.plot(df_treadmill['timestamp'][:16000], signal_treadmill, label='Treadmill Signal')
            plt.plot(treadmill_peak_timestamps, signal_treadmill[peaks], 'x', color='red', label='Treadmill Peaks')
            plt.plot(df_vicon[:16000]['timestamp'], df_vicon[:16000]['Z'] * 3000, label='Vicon Data')
            plt.plot(timestamps_of_lowest_values, lowest_values * 3000, 'x', color='green',
                     label='Vicon Local Minima')
            plt.xlabel('Timestamp')
            plt.ylabel('Signal Value')
            # plt.title('S2 WALKING')
            plt.legend()
            plt.grid(True)
            plt.savefig(name, dpi=300)
            plt.show()

        return treadmill_peak_timestamps, matching_timestamps_filtered, timestamps_of_lowest_values, time_diff_treadmill_vicon, time_diff_treadmill_vicon_median

    except ValueError:
        print("More than 5 peaks detected!")


# TODO: Think of better approach of how to choose 5 peaks in treadmill when more occur, make it more efficient, have a look at plot and variables used


# # --------------- ACC VICON SYNCHRONISATION -----------------
def sync_vicon_acc(accDF, viconDF, start_time, end_time, plot=True):
    # Filter data based on start and end time
    acc_slice = accDF[(accDF['timestamp'] >= start_time) & (accDF['timestamp'] <= end_time)]

    # Adjust accelerometer 'z' signal
    median_acc_z_5s = acc_slice["z"][:500].median()
    adjusted_signal = np.abs(acc_slice["z"] - median_acc_z_5s)

    # Calculate time difference for finding peaks in accelerometer data
    time_difference = acc_slice['timestamp'].iloc[1] - acc_slice['timestamp'].iloc[0]
    distance = max(1, int(0.5 / time_difference))
    acc_peaks_indices, _ = find_peaks(adjusted_signal, height=40, distance=distance)
    acc_peaks_timestamps = np.round(acc_slice['timestamp'].iloc[acc_peaks_indices].values, 2)

    # Sort the acc timestamps in ascending order
    acc_peaks_timestamps = np.sort(acc_peaks_timestamps)

    # Find local minima in the Vicon data
    vicon_local_minima_index = argrelmin(viconDF[:16000]['Z'].values)[0]
    vicon_minima_timestamps = np.array(viconDF[:16000]['timestamp'].iloc[vicon_local_minima_index])

    # Calculate time intervals between consecutive timestamps
    time_intervals = np.diff(vicon_minima_timestamps)  # I needed it to see what to do with the data and what is the pattern in taps

    # Define the threshold for time intervals
    threshold = 0.1  # I set it after checking time intervals

    # Initialize lists to store the selected timestamps
    selected_timestamps = []

    # Add the first value from vicon_minima_timestamps
    selected_timestamps.append(vicon_minima_timestamps[0])

    # Iterate through time intervals and select the first timestamp after each interval exceeds the threshold
    prev_timestamp = vicon_minima_timestamps[0]
    for interval, timestamp in zip(time_intervals, vicon_minima_timestamps[1:]):
        if interval > threshold:
            selected_timestamps.append(timestamp)
        prev_timestamp = timestamp

    # Find the indices where the selected_timestamps values would be inserted
    insert_indices = np.searchsorted(vicon_minima_timestamps, selected_timestamps)

    # Use these indices to index the vicon_local_minima_index array
    matching_indices = vicon_local_minima_index[insert_indices]

    vicon_minima_values_updated = np.array(viconDF[:16000]['Z'].iloc[matching_indices])

    # Filtered values
    filtered_vicon_minima_values = vicon_minima_values_updated[vicon_minima_values_updated <= -0.005]

    # Find the indices where vicon_minima_values_updated is <= 0
    matching_indices_filtered = np.where(vicon_minima_values_updated <= -0.005)[0]

    # Use the indices to filter the selected_timestamps array
    matching_timestamps_filtered = np.array(selected_timestamps)[matching_indices_filtered]

    # Sort indices
    sorted_indices = np.argsort(filtered_vicon_minima_values)

    # Get the indices of the first 5 lowest values
    lowest_indices = sorted_indices[:5]

    # Use the lowest_indices to get the corresponding values
    lowest_values = filtered_vicon_minima_values[lowest_indices]

    # Time difference TREADMILL - VICON
    time_diff_acc_vicon = (acc_peaks_timestamps - matching_timestamps_filtered)  # That might not be useful
    # MEDIAN
    time_diff_acc_vicon_median = "{:.2f}".format(np.median(time_diff_acc_vicon))  # median as float with 3 dec places
    # Change str to float
    time_diff_acc_vicon_median = float(time_diff_acc_vicon_median)


    if plot:
        # Plot acceleration and vicon signals
        plt.figure(figsize=(12, 6))
        plt.plot(acc_slice['timestamp'], adjusted_signal, label='Acceleration', color='green')
        plt.scatter(acc_slice['timestamp'].iloc[acc_peaks_indices], adjusted_signal.iloc[acc_peaks_indices], color='red', marker='x',
                    label='Acc peaks')
        plt.plot(viconDF[:16000]['timestamp'], viconDF[:16000]['Z'] * 300, label='Vicon Data')
        plt.plot(matching_timestamps_filtered, lowest_values * 300, 'x', color='orange',
                     label='Vicon Local Minima')
        plt.xlabel('Timestamp')
        plt.ylabel('Signal Value')
        plt.title('Acc and Vicon signals between {} and {} seconds'.format(start_time, end_time))
        plt.legend()
        plt.grid(True)
        plt.show()

    return time_diff_acc_vicon, time_diff_acc_vicon_median

# ---- CUT DATA AND ALIGN ------------------
def cut_and_move_data(df, time_difference):
    ''' This function synchronises datasets and deletes first 30s worth of data'''
    try:
        datatype_list = ['treadmill', 'acc', 'gyr']
        for datatype in datatype_list:
            if datatype in df['datatype'].values:
                df['timestamp'] -= time_difference
        # Cut the first 30s
        df = df[df['timestamp'] >= 30.000]

        return df

    except KeyError:
        print('Columns are missing.')
        return None
