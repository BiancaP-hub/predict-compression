import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import extract_patient_id
from constants import MSCC_LABELS_FILE, METRICS_DIR


def split_patient_data_vert_level(patient_id, ap_ratio_df, metrics_df):
    """
    Splits the metrics data of a given patient based on the vert level midpoints between pairs 
    of vert levels associated with their labels in the ap_ratio dataset.

    Parameters:
    patient_id (str): The ID of the patient whose data needs to be split.
    ap_ratio_df (pd.DataFrame): DataFrame containing the ap_ratio data with vert level information.
    metrics_df (pd.DataFrame): DataFrame containing the metrics data for the patient.

    Returns:
    list of pd.DataFrame: A list of DataFrames, each representing a split portion of the patient's metrics data.
    """

    # Filter the ap_ratio data for the given patient and extract vert levels
    patient_ap_ratio_df = ap_ratio_df[ap_ratio_df['filename'].str.contains(patient_id)]
    vert_levels = patient_ap_ratio_df['compression_level'].sort_values().tolist()

    # Prepare for splitting the metrics data
    split_dataframes = []
    start_vert = metrics_df['VertLevel'].min()

    for i in range(len(vert_levels) - 1):
        # Calculate the splitting point based on the vert levels
        if vert_levels[i + 1] - vert_levels[i] == 1:
            # Consecutive vert levels, split at the junction
            split_point = vert_levels[i + 1]
        else:
            # Gap between vert levels, split at the midpoint
            split_point = (vert_levels[i] + vert_levels[i + 1]) / 2

        # Split the metrics data up to the split point
        split_df = metrics_df[(metrics_df['VertLevel'] >= start_vert) & (metrics_df['VertLevel'] < split_point)]
        split_dataframes.append(split_df)

        # Update the start vert for the next iteration
        start_vert = split_point

    # Add the remaining data after the last split point
    split_dataframes.append(metrics_df[metrics_df['VertLevel'] >= start_vert])

    return split_dataframes


def split_patient_samples():
    """
    Process data and split it based on patient IDs and vertebrae levels.

    Parameters:
    mscc_labels_file (str): Path to the MSCC labels file.
    metrics_dir (str): Directory containing metrics files.

    Returns:
    List of tuples containing split data and corresponding labels.
    """
    # Load mscc values as pandas dataframe
    ap_ratio_df = pd.read_csv(MSCC_LABELS_FILE, delimiter=',')

    # Apply the function to each filename in the column to extract patient IDs
    patients = ap_ratio_df['filename'].apply(extract_patient_id).unique()

    data_splits = []  # List to store the split data and corresponding labels

    for patient in patients:
        # Load metrics file
        metrics_file = f"{METRICS_DIR}/{patient}.csv"
        metrics_df = pd.read_csv(metrics_file, delimiter=',')

        # Split the data
        split_data_vert_level = split_patient_data_vert_level(patient, ap_ratio_df, metrics_df)

        # For each split, find the corresponding label
        for split_df in split_data_vert_level:
            # Get the VertLevel range of the split
            min_vert_level = split_df['VertLevel'].min()
            max_vert_level = split_df['VertLevel'].max()

            # Find the corresponding label in ap_ratio_df
            label = ap_ratio_df[(ap_ratio_df['compression_level'] >= min_vert_level) &
                                (ap_ratio_df['compression_level'] <= max_vert_level) &
                                (ap_ratio_df['filename'].str.contains(patient))]['diameter_AP_ratio_PAM50_normalized']

            # Store the split data and label
            data_splits.append((split_df, label.tolist()))

    return data_splits

# Example usage
# data_splits = process_data_and_split(MSCC_LABELS_FILE, METRICS_DIR)


def preprocess_data(data_splits, model_type='cnn'):
    all_data = []
    y = []

    # Process each split in data_splits
    for split_df, labels in data_splits:
        # Assuming each split has a single corresponding label
        label = labels[0] if labels else None

        if label is not None:
            # Extract feature data
            patient_data = split_df.iloc[:, 1:].values
            all_data.append(patient_data)
            y.append(label)

    # Pad each patient's data to have the same number of rows
    max_length = max([data.shape[0] for data in all_data])
    padded_data = []
    for data in all_data:
        padded = np.pad(data, ((0, max_length - data.shape[0]), (0, 0)), 'constant', constant_values=0)
        padded_data.append(padded)

    # Flatten the data for certain model types
    if model_type in ['random_forest', 'gradient_boosting']:
        X = np.array([data.flatten() for data in padded_data])
    else:
        X = np.stack(padded_data, axis=0)

    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # Reshape the data for 1D-CNN or LSTM
    if model_type in ['cnn', 'lstm', 'cnn_lstm']:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # Print the shapes of the data
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', len(y_train))
    print('y_test shape:', len(y_test))
    
    return X_train, X_test, y_train, y_test
