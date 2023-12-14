import numpy as np
from scipy.signal import find_peaks, argrelmin


def detect_treadmill_peak(df_treadmill):
    """
    Function detects 5 peaks in the treadmill signal.
    The peaks are the local minima of the signal.

    Parameters
    ----------
    df_treadmill : Dataframe with the treadmill signal.

    Returns
    -------
    treadmill_peak_timestamps : Array with the timestamps of the peaks
    treadmill_peak_values : Array with the values of the peaks
    """
    # Find peaks in the treadmill signal
    median_Ez3_5s = df_treadmill["Ez3 (bits)"][:5000].median()
    signal_treadmill = np.abs(df_treadmill["Ez3 (bits)"][:16000] - median_Ez3_5s)

    # Find the peaks for vicon that overcome 500 bits and has a minimun distance between each other of 0,5 sec
    peaks, _ = find_peaks(signal_treadmill, height=500, distance=500)

    # Extract corresponding timestamps and values for the peaks
    treadmill_peak_timestamps = np.array(df_treadmill['timestamp'].iloc[peaks])  # Array so I'm able to subtract the timestamp values
    treadmill_peak_timestamps = np.sort(treadmill_peak_timestamps)
    treadmill_peak_values = signal_treadmill.iloc[peaks].values

    return treadmill_peak_timestamps, treadmill_peak_values

def vicon_local_minima(df_vicon):
    """

    Function finds local minima in the Vicon data.

    Parameters
    ----------
    df_vicon : Dataframe with the Vicon data.

    Returns
    -------
    vicon_local_minima_index : Array with the indices of the local minima
    vicon_minima_timestamps : Array with the timestamps of the local minima
    vicon_minima_values : Array with the values of the local minima
    """
    # Find local minima in the Vicon data
    vicon_local_minima_index = argrelmin(df_vicon[:16000]['Z'].values)[0]
    vicon_minima_timestamps = np.array(df_vicon[:16000]['timestamp'].iloc[vicon_local_minima_index])

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

    #  Use the lowest_indices to get the corresponding values
    lowest_values = filtered_vicon_minima_values[lowest_indices]

    # Use the lowest_indices to get the corresponding timestamps
    timestamps_of_lowest_values = np.sort(matching_timestamps_filtered[lowest_indices])

    return lowest_values, timestamps_of_lowest_values


def compute_median_time_difference(treadmill_or_acc_timestamps, vicon_timestamps):
    """
    Calculate the median time difference between the provided treadmill and vicon timestamps.

    Parameters
    ----------
    treadmill_or_acc_timestamps: Timestamps of detected peaks in treadmill data
    vicon_timestamps: Timestamps of detected peaks in Vicon data

    Returns
    ----------
    median: Median time difference between the two sets of timestamps
    """

    # Calculate the time differences and median of them
    time_diffs = treadmill_or_acc_timestamps - vicon_timestamps
    median = np.median(time_diffs)

    return median

def acc_peaks(accDF):
    """
    Function finds the peaks in the accelerometer data.

    Parameters
    ----------
    accDF: Dataframe with the accelerometer data.
    start_time: Start time of the data.
    end_time: End time of the data.

    Returns
    ----------
    acc_slice: Dataframe with the accelerometer data.
    adjusted_signal: Adjusted signal of the accelerometer data.
    acc_peaks_indices: Indices of the peaks.
    acc_peaks_timestamps: Timestamps of the peaks.
    """

    # Filter data based on start and end time
    acc_slice = accDF[(accDF['timestamp'] >= 5) & (accDF['timestamp'] <= 16)]

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

    return acc_slice, adjusted_signal, acc_peaks_indices, acc_peaks_timestamps


def compute_acceleration_from_location(vicon_df):
    """
    Function computes the acceleration from the VICON dataset.

    Parameters
    ----------
    df: Dataframe with the location data.

    Returns
    ----------
    cleaned_df: Dataframe with the acceleration data.

    """
    # Drop rows with NaN values in the 'Z' column
    cleaned_df = vicon_df.dropna(subset=['Z'])

    # Calculate velocity from location data using finite differences
    velocity = cleaned_df['Z'].diff() / cleaned_df['timestamp'].diff()

    # Calculate acceleration from velocity
    acceleration = velocity.diff() / cleaned_df['timestamp'].diff()

    # Replace 'Z' column values with acceleration values
    cleaned_df['Z'] = acceleration

    # Drop the first two rows since their acceleration values will be NaN due to differentiation
    cleaned_df = cleaned_df.iloc[2:]

    return cleaned_df


def vicon_acc_peaks_with_max_distance(cleaned_df):
    """
    Function finds the peaks in the accelerometer data.

    Parameters
    ----------
    cleaned_df: Dataframe with the accelerometer data.

    Returns
    ----------
    acc_slice: Dataframe with the accelerometer data.
    adjusted_signal: Adjusted signal of the accelerometer data.
    valid_peak_indices: Indices of the peaks.
    acc_peaks_timestamps: Timestamps of the peaks.

    """

    # Filter data based on start and end time
    acc_slice = cleaned_df[(cleaned_df['timestamp'] >= 5) & (cleaned_df['timestamp'] <= 16)]

    # Adjust accelerometer 'z' signal
    median_acc_Z_5s = acc_slice["Z"][:500].median()
    adjusted_signal = np.abs(acc_slice["Z"] - median_acc_Z_5s)

    # Calculate time difference for finding peaks in accelerometer data
    time_difference = acc_slice['timestamp'].iloc[1] - acc_slice['timestamp'].iloc[0]
    min_distance = max(1, int(0.5 / time_difference))
    max_distance = int(1.5 / time_difference)

    acc_peaks_indices, _ = find_peaks(adjusted_signal, height=40, distance=min_distance)

    # Filter peaks based on max distance constraint
    valid_peak_indices = [acc_peaks_indices[0]]
    for i in range(1, len(acc_peaks_indices)):
        if (acc_slice['timestamp'].iloc[acc_peaks_indices[i]] - acc_slice['timestamp'].iloc[
            valid_peak_indices[-1]]) <= max_distance:
            valid_peak_indices.append(acc_peaks_indices[i])

    # Only keep the first 5 peaks
    vic_valid_peak_indices = valid_peak_indices[:5]
    vic_acc_peaks_timestamps = np.round(acc_slice['timestamp'].iloc[vic_valid_peak_indices].values, 2)

    # Sort the acc timestamps in ascending order
    vic_acc_peaks_timestamps = np.sort(vic_acc_peaks_timestamps)

    return acc_slice, adjusted_signal, vic_valid_peak_indices, vic_acc_peaks_timestamps
