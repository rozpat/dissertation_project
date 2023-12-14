import temp


def timestamp_index(df):
    '''
    This function takes in a dataframe and returns a dataframe with the index
    as the timestamp.
    '''
    df.set_index('timestamp', inplace=True)
    return df


def sequence_prep(df):
    '''
    This function takes in a dataframe and returns a dataframe with the index
    as the timestamp and the columns as the inputs and outputs to create sequences for LSTM model.
    '''
    accDF = timestamp_index(temp.accDF)
    gyrDF = timestamp_index(temp.gyrDF)
    # print(gyrDF.shape, accDF.shape)

    # Check columns and their data types
    # df.info()

    # Defining the inputs and outputs of the LSTM model to create the sequences
    input_1 = accDF['x'].values
    input_2 = accDF['y'].values
    input_3 = accDF['z'].values
    input_4 = gyrDF['x'].values
    input_5 = gyrDF['y'].values
    input_6 = gyrDF['z'].values

    # output_1 = df['CoM_x-CoP_x'].values
    # output_2 = df['CoM_y-CoP_y'].values

    return input_1, input_2, input_3, input_4, input_5, input_6, output_1, output_2



# ----------------------------------------------------- im not sure about the rest of the code

# Reshaping for converting the inputs/output to 2d shape
input_1 = input_1.reshape((len(input_1), 1))
input_2 = input_2.reshape((len(input_2), 1))
input_3 = input_3.reshape((len(input_3), 1))
input_4 = input_4.reshape((len(input_4), 1))
input_5 = input_5.reshape((len(input_5), 1))
input_6 = input_6.reshape((len(input_6), 1))

# output_feat_1 = output_feat_1.reshape((len(output_feat_1), 1))
# output_feat_2 = output_feat_2.reshape((len(output_feat_2), 1))

# QUESTION: CAN I DO IT BELOW??? IT HAS DIFFERENT TIMESTAMPS
# Use of hstack to put together the input sequence arrays horizontally (column wise)
from numpy import hstack

df = hstack((input_1, input_2, input_3, input_4, input_5, input_6))
df[:7]

print(df[:7])

# OR
df_acc = hstack((input_1, input_2, input_3))
df_gyr = hstack((input_4, input_5, input_6))


