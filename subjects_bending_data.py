from numpy.fft import fft
import event_detection as ed
import subjects_static_data as ssd
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd
import os

def indices_to_timestamps(start_indices, stop_indices, dataframe):
    """
    Convert start and stop indices to corresponding timestamps from the provided dataframe.
    """
    start_timestamps = dataframe.iloc[start_indices]['timestamp'].tolist()
    stop_timestamps = dataframe.iloc[stop_indices]['timestamp'].tolist()

    return start_timestamps, stop_timestamps


def extract_data_between_timestamps(data_df, start_timestamps_list, stop_timestamps_list):
    """
    Extracts segments of data between pairs of start and stop timestamps.

    Parameters:
    - data_df: DataFrame containing the data (e.g., accDF or gyrDF)
    - start_timestamps: List of start timestamps
    - stop_timestamps: List of stop timestamps

    Returns:
    - A list of DataFrames, each containing a segment of data between a pair of start and stop timestamps.
    """
    # Check if the length of start and stop timestamp lists are equal
    if len(start_timestamps_list) != len(stop_timestamps_list):
        print("Error: Mismatched length of start and stop timestamp lists.")
        return None

    extracted_segments = []
    for start, stop in zip(start_timestamps_list, stop_timestamps_list):
        # Extract the segment of data between start and stop timestamps
        segment = data_df[(data_df['timestamp'] >= start) & (data_df['timestamp'] <= stop)]
        extracted_segments.append(segment)

    return extracted_segments

def calculate_features(data_segments, cop_values_list):
    """
    Calculates specified features for each segment of data and assigns a corresponding max CoP value.

    Parameters:
    - data_segments: List of DataFrames, each containing a segment of data.
    - cop_values_list: List of CoP values corresponding to each data segment.

    Returns:
    - A list of dictionaries, each containing the calculated features and a CoP value for one data segment.
    """
    if len(data_segments) != len(cop_values_list):
        print("Error: Mismatched length of data segments and CoP values list.")
        return None

    features_list = []

    for segment, cop_value in zip(data_segments, cop_values_list):
        # Check if the segment is not empty
        if segment.empty:
            print("Warning: Empty segment encountered. Skipping.")
            continue

        features = {}
        for axis in ['x', 'y', 'z']:
            # Basic statistical features
            features[f'{axis}_min'] = segment[axis].min()
            features[f'{axis}_max'] = segment[axis].max()
            features[f'{axis}_mean'] = segment[axis].mean()
            features[f'{axis}_std_dev'] = segment[axis].std()
            features[f'{axis}_skewness'] = skew(segment[axis])
            features[f'{axis}_kurtosis'] = kurtosis(segment[axis])

            # Frequency domain features
            fft_vals = fft(segment[axis].to_numpy())
            fft_freq = np.fft.fftfreq(len(fft_vals))
            fft_mag = np.abs(fft_vals)

            dominant_freq_idx = np.argmax(fft_mag)
            features[f'{axis}_dominant_freq'] = np.abs(fft_freq[dominant_freq_idx])
            features[f'{axis}_dominant_amplitude'] = fft_mag[dominant_freq_idx]

        # Add CoP value to the features
        features['CoP'] = cop_value

        features_list.append(features)

    return features_list

def append_features_to_csv(subject, acc_features_list, gyr_features_list, filename):
    """
    Appends the features from acc and gyr to a CSV file.

    Parameters:
    - subject: String, identifier of the subject.
    - acc_features_list: List of dictionaries, each containing acc features for a data segment.
    - gyr_features_list: List of dictionaries, each containing gyr features for a data segment.
    - filename: String, name of the file (or path) to save the data.

    Returns:
    - None
    """
    # Ensure the filename ends with '.csv'
    if not filename.endswith('.csv'):
        filename += '.csv'
    # Check if the length of acc and gyr feature lists are equal
    if len(acc_features_list) != len(gyr_features_list):
        print("Error: Mismatched length of acc and gyr feature lists.")
        return None

    # Convert the features lists to DataFrames
    acc_df = pd.DataFrame(acc_features_list)
    gyr_df = pd.DataFrame(gyr_features_list)

    # Remove the 'CoP' column from the gyr_df
    acc_df = acc_df.drop(columns=['CoP'])

    # Prefix column names with 'acc_' and 'gyr_'
    gyr_df.columns = [f'gyr_{col}' if col != 'CoP' else col for col in gyr_df.columns]
    acc_df.columns = [f'acc_{col}' for col in acc_df.columns]

    # Concatenate the df horizontally
    features_df = pd.concat([acc_df, gyr_df], axis=1)

    # Add a column for the subject
    features_df.insert(0, 'subject', subject)

    # Append the df to the CSV file
    with open(filename, 'a', newline='') as f:
        features_df.to_csv(f, header=False, index=False)

    print(f"Features appended to {filename}")



def create_csv_with_headers(column_names, csv_filename):
    """
    Create a new CSV file with the provided column names and no data.

    Parameters:
    - column_names: List of the column names.
    - csv_filename: Name of the csv file to be created.

    """
    try:
        # Ensure the filename ends with '.csv'
        if not csv_filename.endswith('.csv'):
            csv_filename += '.csv'

        # Create DataFrame and save as CSV
        df = pd.DataFrame(columns=column_names)
        df.to_csv(csv_filename, index=False)
        print(f"CSV file {csv_filename} created successfully in {os.getcwd()}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":

    # Load data for chosen subject and exercise
    path_file = 'D:/My Drive/DISSERTATION/DATA/'
    path_subfolder = 'Dane/real_data/'
    subject_list = [f"S{i}" for i in range(1, 21)]

    all_subject_data_rows = []
    major_dict_features = {}

    columns_names = [
        'subject',
        'acc_x_minimum', 'acc_x_maximum', 'acc_x_mean', 'acc_x_std_dev', 'acc_x_skewness', 'acc_x_kurtosis', 'acc_x_dominant_freq', 'acc_x_dominant_amplitude',
        'acc_y_minimum', 'acc_y_maximum', 'acc_y_mean', 'acc_y_std_dev', 'acc_y_skewness', 'acc_y_kurtosis', 'acc_y_dominant_freq', 'acc_y_dominant_amplitude',
        'acc_z_minimum', 'acc_z_maximum', 'acc_z_mean', 'acc_z_std_dev', 'acc_z_skewness', 'acc_z_kurtosis', 'acc_z_dominant_freq', 'acc_z_dominant_amplitude',
        'gyr_x_minimum', 'gyr_x_maximum', 'gyr_x_mean', 'gyr_x_std_dev', 'gyr_x_skewness', 'gyr_x_kurtosis', 'gyr_x_dominant_freq', 'gyr_x_dominant_amplitude',
        'gyr_y_minimum', 'gyr_y_maximum', 'gyr_y_mean', 'gyr_y_std_dev', 'gyr_y_skewness', 'gyr_y_kurtosis', 'gyr_y_dominant_freq', 'gyr_y_dominant_amplitude',
        'gyr_z_minimum', 'gyr_z_maximum', 'gyr_z_mean', 'gyr_z_std_dev', 'gyr_z_skewness', 'gyr_z_kurtosis', 'gyr_z_dominant_freq', 'gyr_z_dominant_amplitude',
        'COP'
    ]

    csv_filename = 'cop_training_dataset.csv'

    create_csv_with_headers(columns_names, csv_filename=csv_filename)

    for subject in subject_list:

        # Load data for the current subject
        accDF, gyrDF, treadmillDF = ssd.load_subject_datasets(subject, path_file, path_subfolder, chosen_exercise='Bend')

        # Special processing for S1
        if subject == 'S1':
            mask1 = (treadmillDF['timestamp'] >= 70) & (treadmillDF['timestamp'] <= 85)
            mask2 = treadmillDF['timestamp'] <= 32
            cop_median_s1_1 = np.median(treadmillDF['CoPy fore-aft (m)'][mask1])
            cop_median_s1_2 = np.median(treadmillDF['CoPy fore-aft (m)'][mask2])
            substraction = cop_median_s1_2 - cop_median_s1_1

            # Subtract the median only from rows where timestamp <= 43
            mask1 = treadmillDF['timestamp'] <= 65.5
            treadmillDF.loc[mask1, 'CoPy fore-aft (m)'] -= substraction

        elif subject == 'S2':
            # df_treadmil that starts at second 35
            treadmillDF = treadmillDF[treadmillDF['timestamp'] >= 33]

        # Load and preprocess data from a treadmill dataframe
        time, filtered_detrended_cop_x, filtered_detrended_cop_y = ed.load_and_preprocess_cop_data(treadmillDF)
        # Calculate the median of the CoP signal between 65 and 85 s
        cop_x_median = ed.calculate_median_cop(filtered_detrended_cop_x, time)
        cop_y_median = ed.calculate_median_cop(filtered_detrended_cop_y, time)

        # Detect forward, backward and side to side movements' indices
        left_starts, left_stops, right_starts, right_stops, side_highest_peaks, side_lowest_peaks = ed.detect_movements(time,
                                                                                 filtered_detrended_cop_x,
                                                                                 cop_x_median, "CoP_x", p=2,
                                                                                 seconds=20,
                                                                                 prominence_threshold=0.02,
                                                                                 v=False)
        forward_starts, forward_stops, backward_starts, backward_stops, forward_peaks, backward_peaks = ed.detect_movements(
                                                                                            time, filtered_detrended_cop_y,
                                                                                             cop_y_median, "CoP_y", p=2,
                                                                                             seconds=20,
                                                                                             prominence_threshold=0.02,
                                                                                             v=False)

        forward_start_timestamps, forward_stop_timestamps = indices_to_timestamps(forward_starts, forward_stops, treadmillDF)

        acc_extracted_segments = extract_data_between_timestamps(accDF, forward_start_timestamps, forward_stop_timestamps)
        gyr_extracted_segments = extract_data_between_timestamps(gyrDF, forward_start_timestamps, forward_stop_timestamps)

        acc_features_list = calculate_features(acc_extracted_segments, forward_peaks)
        gyr_features_list = calculate_features(gyr_extracted_segments, forward_peaks)

        # Print lists of features to ensure the values were appended to the right columns in the csv file
        print('Accelerometer features:')
        print(acc_features_list)
        print('Gyroscope features:')
        print(gyr_features_list)

        # Append features to CSV file
        append_features_to_csv(subject, acc_features_list, gyr_features_list, csv_filename)





