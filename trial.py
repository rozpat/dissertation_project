from pprint import pprint
import numpy as np
import pandas as pd
import loadData
import synchronisation_v2 as sync
import plots
import event_detection as event
import participants_details as part_data

# -------------------------------------------------------------------------------------------------------------------
# Summary statistics for all participants
participants_file_name = 'patient_data.xlsx'
data = pd.read_excel(participants_file_name)
# Exclude subject S21
data_filtered = data[data['Subject'] != 'S21']

statistics = part_data.calculate_statistics(data_filtered)
print("Statistics for participants that took place in data collection:")
pprint(statistics)
# -------------------------------------------------------------------------------------------------------------------

subjects = [f"S{i}" for i in range(1, 21)]  # Subjects S1-S20
exercises = ['Walk1', 'Walk2', 'Bend', 'Static']

# Get user's choice for subject and exercise
chosen_subject = loadData.get_user_choice(subjects, "Select a subject:")
chosen_exercise = loadData.get_user_choice(exercises, "Select an exercise:")

print(f"Processing data for {chosen_subject} and {chosen_exercise}...")

# Load data for chosen subject and exercise
path_file = 'D:/My Drive/DISSERTATION/DATA/'
path_subfolder = 'Dane/real_data/'
accDF, gyrDF, viconDF, treadmillDF = loadData.load_data_for_subject_and_exercise(path_file, path_subfolder,
                                                                                 chosen_subject, chosen_exercise)
# Create csv files for each dataframe
treadmillDF.to_csv("treadmillDF.csv")
accDF.to_csv("accDF.csv")
viconDF.to_csv("viconDF.csv")

# -------------------------------------------------------------------------------------------------------------------
# treadmill peak timestamps and values
treadmill_peak_timestamps, treadmill_peak_values = sync.detect_treadmill_peak(treadmillDF)
# vicon peak timestamps and values
lowest_values, timestamps_of_lowest_values = sync.vicon_local_minima(viconDF)

# Plotting before alignment:
plots.plot_T_V_signals(treadmillDF, viconDF, 'Detected Treadmill and Vicon peaks before alignment')

# Plotting after alignment:
_, median_time_difference_treadmill = sync.compute_median_time_difference(treadmill_peak_timestamps,
                                                                          timestamps_of_lowest_values)
plots.plot_T_V_signals(treadmillDF, viconDF, 'Detected Treadmill and Vicon peaks after alignment',
                       is_aligned=True,
                       median_time_difference=median_time_difference_treadmill)
# -------------------------------------------------------------------------------------------------------------------
# Time difference before and after alignment
treadmill_peak_timestamps_aligned, _ = sync.detect_treadmill_peak(treadmillDF)

time_diffs = treadmill_peak_timestamps - timestamps_of_lowest_values
time_diffs_aligned = treadmill_peak_timestamps_aligned - timestamps_of_lowest_values
print(f"Time difference between VICON and treadmill peaks before alignment: {time_diffs}")
print(f"Time difference between VICON and treadmill peaks after alignment: {time_diffs_aligned}")
# -------------------------------------------------------------------------------------------------------------------
# Vicon location to acceleration
clean_df_vicon_acc = sync.compute_acceleration_from_location(viconDF)

plots.plot_treadmill_signal(treadmillDF)
plots.plot_acc_signal(accDF)
plots.plot_vicon_signals(viconDF, False)
plots.plot_vicon_signals(viconDF, True, clean_df_vicon_acc)

# --------------------------------------------------------------------------------------------------------------------
# Plotting unaligned AccDF and VICON signals:
plots.plot_acc_A_V_signals(accDF, viconDF, 'Acc and Vicon signals between 4 and 12 seconds (Unaligned)')

# --------------------------------------------------------------------------------------------------------------------
# Time difference before and after the alignment
_, _, _, acc_peaks_timestamps = sync.acc_peaks(accDF)
print(acc_peaks_timestamps)
print(timestamps_of_lowest_values)
_, median_time_difference_acc = sync.compute_median_time_difference(acc_peaks_timestamps,
                                                                    timestamps_of_lowest_values)
print(f"Median time difference: {median_time_difference_acc}")
# --------------------------------------------------------------------------------------------------------------------
# Finding peaks for VICON acceleration data
vic_acc_slice, vic_adjusted_signal, vic_valid_peak_indices, vic_acc_peaks_timestamps = sync.vicon_acc_peaks_with_max_distance(
        clean_df_vicon_acc)
print(f"Vicon acceleration signal peaks timestamps: {vic_acc_peaks_timestamps}")

# For plotting aligned AccDF and VICON signals:
plots.plot_acc_A_V_signals(accDF, viconDF, 'Acc and Vicon signals between 4 and 12 seconds (Aligned)',
                           is_aligned=True)

df_treadmill = sync.cut_and_move_data(treadmillDF, median_time_difference_treadmill)
print(df_treadmill.head(3))
df_treadmill.to_csv('df_treadmill.csv')
df_acc = sync.cut_and_move_data(accDF, median_time_difference_acc)
print(df_acc.head(3))
df_acc.to_csv('df_acc.csv')
df_vicon = sync.cut_and_move_data(viconDF, median_time_difference_treadmill)
print(df_vicon.head(3))

# --------------------------------------------------------------------------------------------------------------------
# Movement detection only for Bend movement

if df_treadmill['excercise'].values.any() == 'Bend':

    if chosen_subject == 'S1':
        mask1 = (df_treadmill['timestamp'] >= 70) & (df_treadmill['timestamp'] <= 85)
        mask2 = df_treadmill['timestamp'] <= 32
        cop_median_s1_1 = np.median(df_treadmill['CoPy fore-aft (m)'][mask1])
        cop_median_s1_2 = np.median(df_treadmill['CoPy fore-aft (m)'][mask2])
        substraction = cop_median_s1_2 - cop_median_s1_1

        # Subtract the median only from rows where timestamp <= 43
        mask1 = df_treadmill['timestamp'] <= 65.5
        df_treadmill.loc[mask1, 'CoPy fore-aft (m)'] -= substraction

    if chosen_subject == 'S2':
        # df_treadmil that starts at second 33
        df_treadmill = df_treadmill[df_treadmill['timestamp'] >= 33]

    plots.cop_raw_data(df_treadmill)

    time, filtered_detrended_cop_x, filtered_detrended_cop_y = event.load_and_preprocess_cop_data(df_treadmill,
                                                                                                  cutoff_time=120,
                                                                                                  v=True)

    median_copx = event.calculate_median_cop(filtered_detrended_cop_x, time)
    median_copy = event.calculate_median_cop(filtered_detrended_cop_y, time)

    forward_starts_x, forward_stops_x, backward_starts_x, backward_stops_x, _, _ = \
        event.detect_movements(time, filtered_detrended_cop_x, median_copx, name='CoPx', v=True)
    forward_starts_y, forward_stops_y, backward_starts_y, backward_stops_y, _, _ = \
        event.detect_movements(time, filtered_detrended_cop_y, median_copy, name='CoPy', v=True)

# ----------------------------------------------------------------------------------------------------
