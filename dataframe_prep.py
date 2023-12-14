import temp
import synchronisation as syn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import temp
from scipy.signal import find_peaks

# Synchronise vicon and treadmill data and cut the first 30s
time_difference = syn.sync_vicon_treadmill(temp.viconDF, temp.treadmillDF, False, 'unsynchronised_vicon_treadmill.png')
viconDF = syn.cut_and_move_data(temp.viconDF, time_difference)
treadmillDF = syn.cut_and_move_data(temp.treadmillDF, time_difference)

# display all columns
pd.set_option('display.max_columns', None)

def select_columns(df):
    """
    This function selects the columns of the dataframe that are needed for the analysis and renames them.
    Returns a new DataFrame with selected and renamed columns.
    """
    if 'CoPx lateral (m)' in df.columns:
        new_df = df.rename(columns={'CoPx lateral (m)': 'CoPx',
                                    'CoPy fore-aft (m)': 'CoPy'})
        return new_df[['timestamp', 'CoPx', 'CoPy']]

    elif 'X.1' in df.columns:
        new_df = df.rename(columns={'X.1': 'CoMx',
                                    'Y.1': 'CoMy'})
        return new_df[['timestamp', 'CoMx', 'CoMy']]

    elif 'datatype' in df.columns and (df['datatype'] == 'acc').all():
        new_df = df.rename(columns={'x': 'acc_x',
                                    'y': 'acc_y',
                                    'z': 'acc_z'})
        return new_df[['timestamp', 'acc_x', 'acc_y', 'acc_z']]
    elif 'datatype' in df.columns and (df['datatype'] == 'gyr').all():
        new_df = df.rename(columns={'x': 'gyr_x',
                                    'y': 'gyr_y',
                                    'z': 'gyr_z'})
        return new_df[['timestamp', 'gyr_x', 'gyr_y', 'gyr_z']]
    else:
        return None


def merge_data(df1, df2):
    """
    Merge dataframes and adds columns
    """
    # merge the two dataframes
    vicon_treadmill_df = pd.merge(df1, df2, on='timestamp')

    # add columns for the difference between cop and com
    vicon_treadmill_df['CoPx-CoMx'] = vicon_treadmill_df['CoPx'] - vicon_treadmill_df['CoMx']
    vicon_treadmill_df['CoPy-CoMy'] = vicon_treadmill_df['CoPy'] - vicon_treadmill_df['CoMy']

    return vicon_treadmill_df

# Select the columns
treadmill_df_selected = select_columns(treadmillDF)
vicon_df_selected = select_columns(viconDF)

# Merged the two dataframes
merged_df = merge_data(treadmill_df_selected, vicon_df_selected)
# print(merged_df.head())  #Shows timestamp, CoMx, CoMy, CoPx, CoPy, CoPx-CoMx, CoPy-CoMy

df_acc = temp.accDF
df_gyr = temp.gyrDF

# print(df_acc.info(), df_gyr.info())
# print(df_acc.shape, df_gyr.shape)
# print(df_gyr.head())
# print(df_acc.head())

# Merge all rows of the acc and gyr dataframes on timestamp
# df_acc_gyr = pd.merge(df_acc, df_gyr, on='timestamp')
# print(df_acc_gyr.head())

# # interpolate the missing values
# df_acc = df_acc.interpolate()
# df_gyr = df_gyr.interpolate()
gyr_df_selected = select_columns(df_gyr)
acc_df_selected = select_columns(df_acc)

# print(acc_df_selected.head())
# print(gyr_df_selected.head())

# Outer join the dataframes on timestamp
df_acc_gyr = pd.merge(acc_df_selected, gyr_df_selected, on='timestamp', how='outer')
print(df_acc_gyr.head())

# combined_df = pd.concat([merged_df, df_acc_gyr], axis=1, join = 'inner')
print(merged_df.head())

# Plot aligned signals
# print(viconDF)

# plt.plot(treadmillDF['timestamp'], np.abs(treadmillDF["Ez3 (bits)"]), label='Treadmill Signal')
# plt.plot(viconDF['timestamp'], viconDF['Z.1'] * 3000, label='Vicon Data')
# plt.xlabel('Timestamp')
# plt.ylabel('Signal Value')
# # plt.title('S2 WALKING')
# plt.legend()
# plt.grid(True)
# plt.show()