import numpy as np
import loadData
import synchronisation_v2 as sync


def find_median_time_difference(path_file, path_subfolder, subject, chosen_exercise):
    """
    This function finds the median time difference between:
     - accelerometer signal peaks and vicon acceleration signal peaks, AND
     - treadmill signal peaks and vicon signal peaks
    for any exercises chosen.

    Parameters:
        - path_file: path to the file
        - path_subfolder: path to the subfolder
        - subject: subject number
        - chosen_exercise: exercise number
    Returns:
        - acc_df: accelerometer dataframe
        - gyr_df: gyroscope dataframe
        - vicon_df: vicon dataframe
        - treadmill_df: treadmill dataframe
        - clean_df_vicon_acc: vicon acceleration dataframe after cleaning
        - time_diffs_acc: time difference between accelerometer peaks and vicon acceleration peaks
        - time_diffs_treadmill: time difference between treadmill peaks and vicon peaks
        - median_time_difference_acc: median time difference between accelerometer peaks and vicon acceleration peaks
        - median_time_difference_treadmill: median time difference between treadmill peaks and vicon peaks

    """
    try:
        # Load files
        acc_df, gyr_df, vicon_df, treadmill_df = loadData.load_data_for_subject_and_exercise(path_file, path_subfolder,
                                                                                             subject, chosen_exercise)
        # Vicon acceleration
        clean_df_vicon_acc = sync.compute_acceleration_from_location(vicon_df)

        # Find peaks in acc, treadmill and vicon datasets
        _, _, _, acc_peaks_timestamps = sync.acc_peaks(acc_df)
        _, vicon_local_minima_timestamps = sync.vicon_local_minima(vicon_df)
        _, _, _, vic_acc_peaks_timestamp = sync.vicon_acc_peaks_with_max_distance(clean_df_vicon_acc)
        treadmill_peaks_timestamps, _ = sync.detect_treadmill_peak(treadmill_df)

        # Median time difference between acc peaks and vicon peaks AND treadmill peaks and vicon peaks
        time_diffs_acc, median_time_difference_acc = sync.compute_median_time_difference(acc_peaks_timestamps,
                                                                                         vic_acc_peaks_timestamp)
        time_diffs_treadmill, median_time_difference_treadmill = sync.compute_median_time_difference(
            treadmill_peaks_timestamps,
            vicon_local_minima_timestamps)

        return acc_df, gyr_df, vicon_df, treadmill_df, clean_df_vicon_acc, time_diffs_acc, time_diffs_treadmill, \
            median_time_difference_acc, median_time_difference_treadmill

    except FileNotFoundError:
        print(f"File not found: {path_file}{path_subfolder}{subject}{chosen_exercise}")



def median_time_difference_for_aligned_data(acc_df, vicon_df, treadmill_df, clean_df_vicon_acc,
                                            median_time_difference_acc, median_time_difference_treadmill):
    """
    This function finds the median time difference between:
     - accelerometer signal peaks and vicon acceleration signal peaks, AND
     - treadmill signal peaks and vicon signal peaks
    for any exercises chosen (after the alignment).

    Parameters:
        - acc_df: accelerometer dataframe
        - vicon_df: vicon dataframe
        - treadmill_df: treadmill dataframe
        - clean_df_vicon_acc: vicon acceleration dataframe after cleaning
        - median_time_difference_acc: median time difference between accelerometer peaks and vicon acceleration peaks
        - median_time_difference_treadmill: median time difference between treadmill peaks and vicon peaks
    Returns:
        - time_diffs_acc_aligned: time difference between accelerometer peaks and vicon acceleration peaks
        - time_diffs_treadmill_aligned: time difference between treadmill peaks and vicon peaks
        - median_time_difference_acc_aligned: median time difference between accelerometer peaks and vicon acceleration peaks
        - median_time_difference_treadmill_aligned: median time difference between treadmill peaks and vicon peaks
    """
    # Align data
    acc_df['timestamp'] -= median_time_difference_acc
    treadmill_df['timestamp'] -= median_time_difference_treadmill

    # Find peaks in acc, treadmill, and vicon datasets after the alignment
    _, _, _, acc_peaks_timestamps = sync.acc_peaks(acc_df)
    _, vicon_local_minima_timestamps = sync.vicon_local_minima(vicon_df)
    _, _, _, vic_acc_peaks_timestamp = sync.vicon_acc_peaks_with_max_distance(clean_df_vicon_acc)
    treadmill_peaks_timestamps, _ = sync.detect_treadmill_peak(treadmill_df)

    # Compute median time difference for aligned data
    time_diffs_acc_aligned, median_time_difference_acc_aligned = sync.compute_median_time_difference(
        acc_peaks_timestamps,
        vic_acc_peaks_timestamp)
    time_diffs_treadmill_aligned, median_time_difference_treadmill_aligned = sync.compute_median_time_difference(
        treadmill_peaks_timestamps,
        vicon_local_minima_timestamps)

    return time_diffs_acc_aligned, time_diffs_treadmill_aligned, median_time_difference_acc_aligned, \
        median_time_difference_treadmill_aligned


def process_data_for_subject_exercise(subject, chosen_exercise, path_file, path_subfolder):
    acc_df, gyr_df, vicon_df, treadmill_df, clean_df_vicon_acc, time_diffs_acc, time_diffs_treadmill, \
        median_time_difference_acc, median_time_difference_treadmill = find_median_time_difference(
        path_file,
        path_subfolder,
        subject, chosen_exercise)

    time_diffs_acc = time_diffs_acc.tolist()
    time_diffs_treadmill = time_diffs_treadmill.tolist()

    time_diffs_acc_aligned, time_diffs_treadmill_aligned, median_time_difference_acc_aligned, \
        median_time_difference_treadmill_aligned = \
        median_time_difference_for_aligned_data(acc_df, vicon_df, treadmill_df, clean_df_vicon_acc,
                                               median_time_difference_acc,
                                               median_time_difference_treadmill)

    time_diffs_acc_aligned = time_diffs_acc_aligned.tolist()
    time_diffs_treadmill_aligned = time_diffs_treadmill_aligned.tolist()

    return time_diffs_acc, time_diffs_treadmill, time_diffs_acc_aligned, time_diffs_treadmill_aligned


def calculate_statistics(data_list):
    mean_value = round(np.mean(data_list), 2)
    std_value = round(np.std(data_list), 2)
    return mean_value, std_value


if __name__ == "__main__":
    path_file = 'D:/My Drive/DISSERTATION/DATA/'
    path_subfolder = 'Dane/real_data/'
    subject_list = [f"S{i}" for i in range(1, 21)]
    exercises = ['Walk1', 'Walk2', 'Bend', 'Static']

    # Initialize lists to store median values
    acc_vicon_time_diff_list = []
    treadmill_vicon_time_diff_list = []
    acc_vicon_aligned_list = []
    treadmill_vicon_aligned_list = []

    for chosen_exercise in exercises:
        for subject in subject_list:
            if subject == 'S5' and (chosen_exercise == 'Walk1' or chosen_exercise == 'Static'):
                continue
            if subject == 'S12' and chosen_exercise == 'Walk1':
                continue

            time_diffs_acc, time_diffs_treadmill, time_diffs_acc_aligned, time_diffs_treadmill_aligned = \
                process_data_for_subject_exercise(subject, chosen_exercise, path_file, path_subfolder)

            acc_vicon_time_diff_list += time_diffs_acc
            treadmill_vicon_time_diff_list += time_diffs_treadmill
            acc_vicon_aligned_list += time_diffs_acc_aligned
            treadmill_vicon_aligned_list += time_diffs_treadmill_aligned

    # Calculate the mean and standard deviation of median_acc_vicon and median_treadmill_vicon before alignment
    mean_median_acc_vicon, std_median_acc_vicon = calculate_statistics(acc_vicon_time_diff_list)
    mean_median_treadmill_vicon, std_median_treadmill_vicon = calculate_statistics(treadmill_vicon_time_diff_list)
    # Calculate the mean and standard deviation of median_acc_vicon_aligned and median_treadmill_vicon_aligned
    mean_median_acc_vicon_aligned, std_median_acc_vicon_aligned = calculate_statistics(acc_vicon_aligned_list)
    mean_median_treadmill_vicon_aligned, std_median_treadmill_vicon_aligned = calculate_statistics(treadmill_vicon_aligned_list)

    # Print the rounded results
    print(f"Mean median_acc_vicon: {mean_median_acc_vicon}")
    print(f"Standard deviation median_acc_vicon: {std_median_acc_vicon}")
    print(f"Mean median_treadmill_vicon: {mean_median_treadmill_vicon}")
    print(f"Standard deviation median_treadmill_vicon: {std_median_treadmill_vicon}")

    # Print the rounded results for aligned data
    print(f"Mean median_acc_vicon_aligned: {mean_median_acc_vicon_aligned}")
    print(f"Standard deviation median_acc_vicon_aligned: {std_median_acc_vicon_aligned}")
    print(f"Mean median_treadmill_vicon_aligned: {mean_median_treadmill_vicon_aligned}")
    print(f"Standard deviation median_treadmill_vicon_aligned: {std_median_treadmill_vicon_aligned}")