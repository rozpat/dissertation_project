import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import synchronisation as syn
import temp

# # Plot unsynchronised vicon and treadmill data
# time_difference = syn.sync_vicon_treadmill(temp.viconDF, temp.treadmillDF, True, 'unsynchronised_vicon_treadmill.png')
#
# # Plot synchronised vicon and treadmill data
# temp.treadmillDF['timestamp'] -= time_difference
# synchronised_data_plot = syn.sync_vicon_treadmill(temp.viconDF, temp.treadmillDF, True, 'synchronised_vicon_treadmill.png')

# Function to plot bending accelerometer
def plot_acc_or_gyr(dataframe, name, y_axis_name, start_time=0):
  time_sec = dataframe['timestamp'] - dataframe['timestamp'].iloc[0]  # Calculate time in seconds

  # Filter data based on the start_time
  valid_indices = time_sec >= start_time
  time_sec = time_sec[valid_indices]

  # Plotting acceleration or gyroscope plot
  plt.plot(time_sec, dataframe['x'][valid_indices], label='x')
  plt.plot(time_sec, dataframe['y'][valid_indices], label='y')
  plt.plot(time_sec, dataframe['z'][valid_indices], label='z')

  # Customize the plot
  plt.xlabel('Time [s]', fontsize=13)
  plt.ylabel(y_axis_name, fontsize=13)
  plt.title(f'3 axis {name}', fontsize=18)
  plt.legend()

  # Show the plot
  plt.show()

# # ------- Plot acceleration -------
# y_axis_name = 'Acceleration [°/s]'
# name = 'acceleration-bending'
#
# # Plot values starting from time equal to 30
# start_time = 0
# plot_acc_or_gyr(temp.accDF, name, y_axis_name, start_time)

# ---------------------------------

# ------- Plot treadmill 'CoPx and CoPy' against time -------
copx = np.asarray(temp.treadmillDF['CoPx lateral (m)'], float)
copy = np.asarray(temp.treadmillDF['CoPy fore-aft (m)'], float)
time = np.asarray(temp.treadmillDF['timestamp'], float)
plt.plot(time, copx, label='CoPx')
plt.plot(time, copy,  label='CoPy')
plt.xlabel('Time [s]', fontsize=13)
plt.ylabel('CoPx and CoPy [m]', fontsize=13)
plt.title('CoPx and CoPy against time', fontsize=18)
plt.legend()
plt.show()

# ------- Plot CoPx against CoPy, but only for first try of leaning forward, backwards and to the sides -------

copx = np.asarray(temp.treadmillDF['CoPx lateral (m)'], dtype=float)
copy = np.asarray(temp.treadmillDF['CoPy fore-aft (m)'], dtype=float)
time = np.asarray(temp.treadmillDF['timestamp'], dtype=float)

# Calculate the median of CoPx and CoPy
median_copx = np.nanmedian(copx)
median_copy = np.nanmedian(copy)

# Subtract the median from the data
copx -= median_copx
copy -= median_copy

# Filter data based on timestamp range (40 to 65)
start_idx = np.where(time >= 30)[0][0]
end_idx = np.where(time <= 55)[0][-1]

filtered_copx = copx[start_idx:end_idx+1]
filtered_copy = copy[start_idx:end_idx+1]

# Plot the filtered data
plt.plot(filtered_copx, filtered_copy)
plt.plot([0, 0], [0.16, -0.16], 'r')
plt.plot([-0.16, 0.16], [0, 0], 'r')
plt.axis([-0.16, 0.16, -0.16, 0.16])
plt.xlabel('CoPx [m]', fontsize=13)
plt.ylabel('CoPy [m]', fontsize=13)
plt.title('Leaning forward, backwards and to the sides', fontsize=18)

# Show the plot
plt.show()

# -------------------------------------------------------------