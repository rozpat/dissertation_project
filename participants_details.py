def calculate_statistics(participants_df):
    """
    This function calculates statistics for the participants data.

    Parameters:
        - participants_df: dataframe containing participants data

    Returns:
        - participants_data: dictionary containing the statistics
    """

    # Initialise a dictionary to store results
    participants_data = {}

    # Calculate number of males and females
    participants_data['num_males'] = participants_df['Sex'].value_counts().get('M', 0)
    participants_data['num_females'] = participants_df['Sex'].value_counts().get('F', 0)

    # Calculate mean and SD for specified columns
    columns_to_analyse = {
        'Age': 'Age',
        'Height': 'Height [cm]',
        'Weight': 'Weight [kg]',
        'Asis distance': 'Asis distance [cm]',
        'Leg length': 'Leg length [cm]',
        'Shoe size': 'Shoe size [UK]'
    }

    for label, col in columns_to_analyse.items():
        if label == 'Age' or label == 'Shoe size':
            participants_data[f'{label} mean'] = round(participants_df[col].mean(), 1)
            participants_data[f'{label} std'] = round(participants_df[col].std(), 1)
        else:
            participants_data[f'{label} mean'] = round(participants_df[col].mean(), 2)
            participants_data[f'{label} std'] = round(participants_df[col].std(), 2)

    return participants_data
