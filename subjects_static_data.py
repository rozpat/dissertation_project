import numpy as np
import pandas as pd
from numpy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
import synchronisation_v2 as sync
import loadData as temp


def load_subject_datasets(subject, path_file, path_subfolder, chosen_exercise='Static'):
    """
    This function loads the data from treadmill, accelerometer, gyroscope and vicon for a specific subject and exercise.
    The data is cut and moved to the same starting point.
    The data is saved as csv files.

    Parameters:
        - subject: chosen subject
        - path_file: path to the file
        - path_subfolder: path to the subfolder
        - chosen_exercise: chosen exercise

    Returns:
        - acc_df, gyr_df, vicon_df, treadmill_df: dataframes containing the data for the subject and exercise

    Raises:
        - FileNotFoundError: if the file is not found for the subject
    """
    try:
        acc_df, gyr_df, vicon_df, treadmill_df = temp.load_data_for_subject_and_exercise(path_file, path_subfolder,
                                                                                         subject, chosen_exercise)
        clean_df_vicon_acc = sync.compute_acceleration_from_location(vicon_df)

        _, _, _, acc_peaks_timestamps = sync.acc_peaks(acc_df)
        _, vicon_local_minima_timestamps = sync.vicon_local_minima(vicon_df)
        _, _, _, vic_acc_peaks_timestamp = sync.vicon_acc_peaks_with_max_distance(clean_df_vicon_acc)
        treadmill_peak_timestamps, _ = sync.detect_treadmill_peak(treadmill_df)

        _, median_time_difference_acc = sync.compute_median_time_difference(acc_peaks_timestamps,
                                                                            vic_acc_peaks_timestamp)
        _, median_time_difference_treadmill = sync.compute_median_time_difference(treadmill_peak_timestamps,
                                                                                  vicon_local_minima_timestamps)
        acc_df = sync.cut_and_move_data(acc_df, median_time_difference_acc)
        gyr_df = sync.cut_and_move_data(gyr_df, median_time_difference_acc)
        treadmill_df = sync.cut_and_move_data(treadmill_df, median_time_difference_treadmill)
        #Save the dataframes
        acc_df.to_csv(f'{subject}_acc.csv')
        gyr_df.to_csv(f'{subject}_gyr.csv')
        treadmill_df.to_csv(f'{subject}_treadmill.csv')

        return acc_df, gyr_df, treadmill_df

    except FileNotFoundError:
        print(f'File not found for subject: {subject}')
        return None


def split_data_into_tasks(acc_gyr_df, task_duration=30, tasks=4):
    """
    This function splits the data into tasks.
    The data is cut and moved to the same starting point.

    Parameters:
        - acc_gyr_df: dataframe containing the data for the subject and exercise
        - task_duration: duration of each task
        - tasks: number of tasks

    Returns:
        - tasks_data: list of dataframes containing the data for each task

    Raises:
        - TypeError: if the data is not a dataframe
    """
    try:
        tasks_data = []
        for task in range(1, tasks + 1):
            tasks_data.append(acc_gyr_df[(acc_gyr_df['timestamp'] > (task - 1) * task_duration + 30) & (
                    acc_gyr_df['timestamp'] <= task * task_duration + 30)])

        return tasks_data

    except TypeError:
        return None

def features_for_tasks_with_overlap(tasks_data, window_size=2, overlap=0.5):
    """
    This function computes the features for each task with overlap.

    Parameters:
    - tasks_data: list of dataframes containing the data for each task
    - window_size: size of the window
    - overlap: overlap between windows

    Returns:
    - features_dict: dictionary containing the features for each task

    Raises:
    - TypeError: if the data is not a dataframe
    """
    try:
        features_dict = {}

        for task_index, task in enumerate(tasks_data):
            task_dict = {}  # Create a dictionary for the current task
            window_id = 1  # Initialise window_id for the current task

            for axis in ['x', 'y', 'z']:
                series = task[axis].values
                sampling_rate = 400

                # Calculate the number of data points in each window
                window_length = int(window_size * sampling_rate)

                # Calculate the step size for the given overlap
                step_size = int(window_length * (1 - overlap))

                # Split the series into overlapping windows
                start = 0
                while start + window_length <= len(series):
                    window = series[start:start + window_length]

                    # Calculate statistics for the current window
                    minimum = window.min()
                    maximum = window.max()
                    mean = window.mean()
                    std_dev = window.std()
                    skewness = skew(window)
                    kurt = kurtosis(window)

                    # Compute the FFT for the window
                    yf = fft(window)
                    xf = fftfreq(len(window), 1 / sampling_rate)
                    yf = yf[xf > 0]
                    xf = xf[xf > 0]
                    dominant_freq = xf[np.argmax(np.abs(yf))]
                    dominant_amplitude = np.abs(yf).max()

                    # Create the feature list for the current window
                    feature_list = [
                        minimum, maximum, mean, std_dev, skewness, kurt,
                        dominant_freq, dominant_amplitude
                    ]

                    # Add the 'eyes' feature for the current window
                    eyes_feature = [0 if task_index == 0 or task_index == 2 else 1]
                    feature_list.extend(eyes_feature)

                    # Add the 'window_id' for the current window
                    window_id_feature = [window_id]
                    feature_list.extend(window_id_feature)

                    # Add the feature list to the task dictionary
                    if axis not in task_dict:
                        task_dict[axis] = []
                    task_dict[axis].append(feature_list)

                    # Move the window by the step size
                    start += step_size
                    window_id += 1  # Increment the window_id for the next window

            # Add the task dictionary to the features_dict
            features_dict[f'task_{task_index + 1}'] = task_dict

        return features_dict

    except TypeError:
        return None




def create_csv_with_headers(column_names, csv_filename):
    """
    Create a new CSV file with the provided column names and no data.

    Parameters:
    - column_names: List of the column names.
    - csv_filename: Name of the csv file to be created.

    """
    df = pd.DataFrame(columns=column_names)
    df.to_csv(csv_filename, index=False)


def append_subject_to_csv(subject, features_dict_acc, features_dict_gyr, column_names, csv_filename):
    """
    Append a new subject to the existing CSV file.

    Parameters:
    - subject: Subject ID.
    - features_dict_acc: Dictionary of features for the accelerometer.
    - features_dict_gyr: Dictionary of features for the gyroscope.
    - column_names: List of column names.
    - csv_filename: Name of the CSV file.

    """
    try:
        # Read the existing CSV file
        existing_df = pd.read_csv(csv_filename)

        # Check if the new subject is already present
        if subject not in existing_df['subject'].unique():
            # Convert the features_dict to a list of lists
            subject_data_list = []
            for task_id, task_data in features_dict_acc.items():
                for window_id, window_data in enumerate(task_data['x']):
                    # Extract 'class' and 'window_id'
                    class_value = window_data[-2]
                    window_id_value = window_data[-1]

                    # Extract features from the acc data
                    acc_x_features = window_data[:-2]
                    acc_y_features = features_dict_acc[task_id]['y'][window_id][:-2]
                    acc_z_features = features_dict_acc[task_id]['z'][window_id][:-2]

                    # Extract features from the gyroscope data
                    gyr_x_features = features_dict_gyr[task_id]['x'][window_id][:-2]
                    gyr_y_features = features_dict_gyr[task_id]['y'][window_id][:-2]
                    gyr_z_features = features_dict_gyr[task_id]['z'][window_id][:-2]

                    # Append all the features to a row
                    row = [subject, task_id, window_id_value]  # Using window_id_value
                    row.extend(acc_x_features)
                    row.extend(acc_y_features)
                    row.extend(acc_z_features)
                    row.extend(gyr_x_features)
                    row.extend(gyr_y_features)
                    row.extend(gyr_z_features)
                    row.append(class_value)

                    subject_data_list.append(row)

            # Convert the list of lists to a DataFrame
            new_subject_df = pd.DataFrame(subject_data_list, columns=column_names)

            # Append this DataFrame to the existing CSV file
            new_subject_df.to_csv(csv_filename, mode='a', header=False, index=False)

        else:
            print(f"Data for subject {subject} already exists in the CSV file. No data was appended.")
            return None

    except AttributeError:
        return None


if __name__ == "__main__":

    # Define the column names as specified
    column_names = [
        'subject', 'task', 'window_id',
        'acc_x_minimum', 'acc_x_maximum', 'acc_x_mean', 'acc_x_std_dev', 'acc_x_skewness', 'acc_x_kurtosis', 'acc_x_dominant_freq', 'acc_x_dominant_amplitude',
        'acc_y_minimum', 'acc_y_maximum', 'acc_y_mean', 'acc_y_std_dev', 'acc_y_skewness', 'acc_y_kurtosis', 'acc_y_dominant_freq', 'acc_y_dominant_amplitude',
        'acc_z_minimum', 'acc_z_maximum', 'acc_z_mean', 'acc_z_std_dev', 'acc_z_skewness', 'acc_z_kurtosis', 'acc_z_dominant_freq', 'acc_z_dominant_amplitude',
        'gyr_x_minimum', 'gyr_x_maximum', 'gyr_x_mean', 'gyr_x_std_dev', 'gyr_x_skewness', 'gyr_x_kurtosis', 'gyr_x_dominant_freq', 'gyr_x_dominant_amplitude',
        'gyr_y_minimum', 'gyr_y_maximum', 'gyr_y_mean', 'gyr_y_std_dev', 'gyr_y_skewness', 'gyr_y_kurtosis', 'gyr_y_dominant_freq', 'gyr_y_dominant_amplitude',
        'gyr_z_minimum', 'gyr_z_maximum', 'gyr_z_mean', 'gyr_z_std_dev', 'gyr_z_skewness', 'gyr_z_kurtosis', 'gyr_z_dominant_freq', 'gyr_z_dominant_amplitude',
        'class'
    ]


    # Load data for chosen subject and exercise
    path_file = 'D:/My Drive/DISSERTATION/DATA/'
    path_subfolder = 'Dane/real_data/'
    subject_list = [f"S{i}" for i in range(1, 21)]

    create_csv_with_headers(column_names, csv_filename="classifier_training_data.csv")

    # Append the new subjects to the CSV file
    for subject in subject_list:
        if subject == 'S5':
            continue
        accDF, gyrDF, _ = load_subject_datasets(subject, path_file, path_subfolder)
        tasks_data_acc = split_data_into_tasks(accDF)
        tasks_data_gyr = split_data_into_tasks(gyrDF)
        features_dict_acc = features_for_tasks_with_overlap(tasks_data_acc)
        features_dict_gyr = features_for_tasks_with_overlap(tasks_data_gyr)
        # print(features_dict_acc)
        append_subject_to_csv(subject, features_dict_acc, features_dict_gyr, column_names,  csv_filename="classifier_training_data.csv")


