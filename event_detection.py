import scipy
import numpy as np
import scipy.signal
from scipy.signal import butter, filtfilt, detrend, find_peaks
import plots


def load_and_preprocess_cop_data(cut_treadmill_df, cutoff_time=120, v=False):
    """
    Load and preprocess data from a treadmill dataframe.

    Parameters:
    - cut_treadmill_df: treadmill dataframe after synchronisation.
    - cutoff_time: The cutoff time for the data.
    - v: Whether to plot filtered and detrended data

    Returns:
    - time: Timestamps.
    - filtered_detrended_copx: Filtered and detrended CoPx data (leaning to the sides).
    - filtered_detrended_copy: Filtered and detrended CoPy data (leaning forward and backwards).
    """
    df_treadmill = cut_treadmill_df[cut_treadmill_df['timestamp'] <= cutoff_time]
    df_treadmill = df_treadmill.dropna(subset=['CoPx lateral (m)', 'CoPy fore-aft (m)'])

    copx = np.asarray(df_treadmill['CoPx lateral (m)'], float)
    copy = np.asarray(df_treadmill['CoPy fore-aft (m)'], float)
    time = np.asarray(df_treadmill['timestamp'], float)

    # Low-pass filter design with the selected cutoff frequency of 0.5 Hz
    sampling_rate = 1 / np.mean(np.diff(time))
    nyquist_rate = 0.3 * sampling_rate
    b_selected, a_selected = scipy.signal.butter(N=4, Wn=0.3 / nyquist_rate, btype='low')

    # Apply the filter to CoP data and detrend
    filtered_cop_x = scipy.signal.filtfilt(b_selected, a_selected, copx)
    filtered_detrended_cop_x = scipy.signal.detrend(filtered_cop_x)

    filtered_cop_y = scipy.signal.filtfilt(b_selected, a_selected, copy)
    filtered_detrended_cop_y = scipy.signal.detrend(filtered_cop_y)

    if v:
        plots.plot_filtered_detrended_cop(filtered_detrended_cop_x, filtered_detrended_cop_y, time)

    return time, filtered_detrended_cop_x, filtered_detrended_cop_y


def calculate_median_cop(filtered_signal_cop, time, start_time=65, end_time=85):
    """
    Calculate the median of the CoP signal within a specified time range.

    Parameters:
    - signal_cop: The input cop signal.
    - time: Timestamps corresponding to the data points.
    - start_time: Start time of the range.
    - end_time: End time of the range.

    Returns:
    - cop_median_value: The median value of data within the specified time range.
    """
    mask = (time >= start_time) & (time <= end_time)
    cop_median_value = np.median(filtered_signal_cop[mask])

    return cop_median_value


def detect_movements(time, filtered_data, median_value, name, p=2, seconds=20, prominence_threshold=0.02, v=False):
    """
    Detect forward, backward and side to side movements based on filtered data.

    Parameters:
    - p: Number of peaks to be detected
    - time: Timestamps corresponding to the data points
    - filtered_data: Filtered and detrended data
    - median_value: Median value for movement detection
    - seconds: Minimum horizontal distance (in seconds) between peaks
    - prominence_threshold: Minimum prominence of peaks for detection
    - v: Boolean indicating if the data should be plotted
    - name: CoPx or CoPy

    Returns:
    - forward_starts, forward_stops: Lists of start and stop indices for forward movements
    - backward_starts, backward_stops: Lists of start and stop indices for backward movements
    """
    # Find peaks with prominence
    distance_threshold = int(seconds / np.mean(np.diff(time)))  # Convert x seconds to indices

    # Detect two highest peaks for leaning FORWARD and to the RIGHT
    forward_peaks, forward_properties = find_peaks(filtered_data, distance=distance_threshold,
                                                   prominence=prominence_threshold)
    highest_peaks = forward_peaks[np.argsort(forward_properties["prominences"])[-p:]]

    forward_starts = []
    forward_stops = []

    for peak in highest_peaks:
        start_index = peak
        while filtered_data[start_index] > median_value and start_index > 0:
            start_index -= 1
        forward_starts.append(start_index)

        end_index = peak
        while filtered_data[end_index] > median_value and end_index < len(filtered_data) - 1:
            end_index += 1
        forward_stops.append(end_index)

    # Detect two lowest peaks for leaning BACKWARD and to the LEFT
    backward_peaks, backward_properties = find_peaks(-filtered_data, distance=distance_threshold,
                                                     prominence=prominence_threshold)
    lowest_peaks = backward_peaks[np.argsort(backward_properties["prominences"])[-p:]]

    backward_starts = []
    backward_stops = []

    for peak in lowest_peaks:
        start_index = peak
        while filtered_data[start_index] < median_value and start_index > 0:
            start_index -= 1
        backward_starts.append(start_index)

        end_index = peak
        while filtered_data[end_index] < median_value and end_index < len(filtered_data) - 1:
            end_index += 1
        backward_stops.append(end_index)

    highest_peaks_values = np.sort(filtered_data[highest_peaks])
    lowest_peaks_values = np.sort(filtered_data[lowest_peaks])

    # Sort the indices
    forward_starts.sort()
    forward_stops.sort()
    backward_starts.sort()
    backward_stops.sort()

    if v:
        plots.plot_detected_events(filtered_data, time, name, median_value, highest_peaks, forward_starts,
                                   forward_stops, lowest_peaks, backward_starts, backward_stops)

    return forward_starts, forward_stops, backward_starts, backward_stops, highest_peaks_values, lowest_peaks_values


def extract_movement_timestamps(time, movement_starts, movement_stops):
    """
    Extract data within the periods when the movement starts and ends.

    Parameters:
    - time
    - movement_starts: List of start indices for forward movements.
    - movement_stops: List of stop indices for forward movements.
    - filtered_data: The original filtered data array.

    Returns:
    - movement_start_timestamps: Timestamps when the movement starts.
    - movement_end_timestamps: Timestamps when the movement ends.
    """
    movement_start_timestamps = []
    movement_end_timestamps = []

    for start, stop in zip(movement_starts, movement_stops):
        movement_start_timestamps.append(time[start])
        movement_end_timestamps.append(time[stop])

    return movement_start_timestamps, movement_end_timestamps

