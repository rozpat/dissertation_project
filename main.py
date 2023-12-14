#--------------ACC VICON SYNCHRONISATION-------
import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import temp
import synchronisation as sync
import plots
# # DataFrames with timestamp <= 16
# viconDF_filtered = temp.viconDF[temp.viconDF['timestamp'] <= 16.00]
# accDF_filtered = temp.accDF[temp.accDF['timestamp'] <= 16.00]
#
# # vicon Z values
# vicon_timestamp = np.array(viconDF_filtered['timestamp'])
# signal_vicon_Z = np.array(viconDF_filtered['Z'])
#
# # Acc timestamp and z values
# acc_timestamp = np.array(accDF_filtered['timestamp'])
# signal_acc_z = np.array(accDF_filtered['z'])
#
# # Velocity
# velocity = cumtrapz(signal_acc_z, acc_timestamp, initial = 0)
#
# # displacement
# displacement = cumtrapz(velocity, acc_timestamp, initial = 0)
#
#
# # print(acc_filtered)
# # # # Compute cross-correlation
# # # cross_correlation = np.correlate(signal_acc_z, signal_vicon_Z, mode='full')
# #
# # Plotting acceleration, velocity and displacement
# plt.figure(figsize=(10, 6))
# plt.plot(vicon_timestamp, signal_vicon_Z*3000, label='Vicon')
# plt.plot(acc_timestamp, velocity, label='Velocity')
# plt.plot(acc_timestamp, displacement, label='Displacement')
# plt.plot(acc_timestamp, signal_acc_z, label='Acceleration')
# plt.xlabel('Time (s)')
# plt.ylabel('Value')
# plt.title('Acceleration and Velocity')
# plt.legend()
# plt.grid(True)
# plt.show()

# ----------------------------------------
accDF = temp.accDF
viconDF = temp.viconDF

viconDF.to_csv('viconDF.csv')

y_axis_name = 'Acceleration [°/s]'
name = 'acceleration-cal1'

# Plot values starting from time equal to 30
start_time = 0
plots.plot_acc_or_gyr(accDF, name, y_axis_name, start_time)

# Set the timestamp as the index for both dataframes
accDF.set_index('timestamp', inplace=True)
viconDF.set_index('timestamp', inplace=True)

# Create a new index with the frequency 0.001
new_index = pd.Index(np.arange(accDF.index.min(), accDF.index.max() + 0.001, 0.001), name='timestamp')

# Reindex the dataframe and interpolate the missing values
interpolated_accDF = accDF.reindex(new_index).interpolate(method='linear')
# ------------ PLOT ------------------
time_sec = interpolated_accDF.index.to_series() - interpolated_accDF.index.to_series().iloc[0]  # Calculate time in seconds

# Filter data based on the start_time
valid_indices = time_sec >= start_time
time_sec = time_sec[valid_indices]

# Plotting acceleration or gyroscope plot
plt.plot(time_sec, interpolated_accDF['x'][valid_indices], label='x')
plt.plot(time_sec, interpolated_accDF['y'][valid_indices], label='y')
plt.plot(time_sec, interpolated_accDF['z'][valid_indices], label='z')

# Customize the plot
plt.xlabel('Time [s]', fontsize=13)
plt.ylabel(y_axis_name, fontsize=13)
plt.title(f'3 axis {name}', fontsize=18)
plt.legend()

# Show the plot
plt.show()
# -----------------------------
viconDF_interpolated = viconDF.reindex(accDF.index).interpolate(method='linear')

# Plot vicon data 'Z' against accDF.index
plt.figure(figsize=(10, 6))
plt.plot(viconDF.index, viconDF['Z'], label='viconDF')
plt.show()

print(viconDF_interpolated.head())

#
# # Reindex the interpolated_accDF dataframe using the viconDF timestamps
# matched_accDF = interpolated_accDF.reindex(viconDF.index).interpolate(method='linear')

# # Define the list of categorical columns
# categorical_columns = ['subject', 'excercise', 'datatype']
#
# # Use forward fill (ffill) followed by backward fill (bfill) to fill NaN values in categorical columns
# matched_accDF[categorical_columns] = matched_accDF[categorical_columns].ffill().bfill()
#
# # Calculate velocity
# acc_velocity = cumtrapz(matched_accDF['z'], matched_accDF.index, initial=0)
#
# # Calculate displacement (location) using cumulative trapezoidal integration on the velocity data
# displacement = cumtrapz(acc_velocity, matched_accDF.index, initial=0)
#
# # Filter the dataframes based on the timestamp <= 16
# viconDF_filtered = viconDF[viconDF.index <= 16.00]
# accDF_filtered = matched_accDF[matched_accDF.index <= 16.00]
#
# # Vicon Z values
# vicon_timestamp = np.array(viconDF_filtered.index)
# signal_vicon_Z = np.array(viconDF_filtered['Z'])
#
# # Acc timestamp and z values
# acc_timestamp = np.array(accDF_filtered.index)
# signal_acc_z = np.array(accDF_filtered['z'])
# signal_acc_x = np.array(accDF_filtered['x'])
# signal_acc_y = np.array(accDF_filtered['y'])
#
# # Filter the velocity and displacement arrays to match the dimensions of accDF_filtered
# acc_velocity_filtered = acc_velocity[:len(accDF_filtered)]
# displacement_filtered = displacement[:len(accDF_filtered)]
#
# # Plot signal_acc_z, signal_vicon_Z, acc_velocity_filtered, and displacement_filtered
# plt.figure(figsize=(10, 6))
# plt.plot(acc_timestamp, signal_acc_z, label='Acceleration z')
# plt.plot(acc_timestamp, signal_acc_x, label='Acceleration x')
# plt.plot(acc_timestamp, signal_acc_y, label='Acceleration y')
# plt.plot(acc_timestamp, acc_velocity_filtered, label='Velocity z')
# plt.plot(acc_timestamp, displacement_filtered, label='Displacement z')
# plt.plot(vicon_timestamp, signal_vicon_Z*3000, label='Vicon')
# plt.xlabel('Time (s)')
# plt.ylabel('Value')
# plt.title('Acceleration and Vicon')
# plt.legend()
# plt.grid(True)
# plt.show()


# # save interpolated accDF to csv
# matched_accDF.to_csv('acc_interploated_cal3.csv')
# viconDF.to_csv('vicon_cal3.csv')
