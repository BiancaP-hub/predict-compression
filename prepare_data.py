import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import extract_patient_id
from constants import MSCC_LABELS_FILE, METRICS_DIR, FEATURE_NAMES

def split_patient_data_slice_number(patient, ap_ratio_df, metrics_df):
    # Filter the ap_ratio_df for the current patient
    patient_ap_ratio = ap_ratio_df[ap_ratio_df['filename'].str.contains(patient)]

    # Sort the patient data by slice number in ascending order (to correct the reversed order)
    patient_ap_ratio = patient_ap_ratio.sort_values(by='slice(I->S)', ascending=True)

    # Get the list of slice numbers and compression values
    slice_numbers = patient_ap_ratio['slice(I->S)'].tolist()
    compression_values = patient_ap_ratio['diameter_AP_ratio_PAM50_normalized'].tolist()

    # Initialize a list to store the splits
    splits = []

    if len(slice_numbers) > 1:
        # Calculate the mean of the slice numbers for splitting
        split_slices = [(slice_numbers[i] + slice_numbers[i+1]) // 2 for i in range(len(slice_numbers) - 1)]

        prev_slice = 0
        for index, split_slice in enumerate(split_slices):
            # Select data from the previous slice up to the current split slice
            split_df = metrics_df[(metrics_df['slice'] >= prev_slice) & (metrics_df['slice'] < split_slice)]

            # Update the previous slice for the next iteration
            prev_slice = split_slice

            # Add the split data and corresponding compression value to the list
            # The compression values are associated in reverse order
            splits.append((split_df, compression_values[-(index + 1)]))

        # Handle the remaining data after the last split
        remaining_df = metrics_df[metrics_df['slice'] >= prev_slice]
        if not remaining_df.empty:
            # Use the first compression value for the remaining data
            splits.append((remaining_df, compression_values[0]))
    else:
        # If there is only one compression value, return the entire dataset with that value
        splits.append((metrics_df, compression_values[-1]))

    return splits

def split_patient_samples(mscc_labels_file=MSCC_LABELS_FILE, metrics_dir=METRICS_DIR):
    # Load mscc values as pandas dataframe
    ap_ratio_df = pd.read_csv(mscc_labels_file, delimiter=',')

    # Apply the function to each filename in the column to extract patient IDs
    # Assuming extract_patient_id is a defined function
    patients = ap_ratio_df['filename'].apply(extract_patient_id).unique()

    data_splits = []  # List to store the split data and corresponding labels

    for patient in patients:
        # Load metrics file
        metrics_file = f"{metrics_dir}/{patient}.csv"
        metrics_df = pd.read_csv(metrics_file, delimiter=',')

        # Split the data
        split_data_slice_number = split_patient_data_slice_number(patient, ap_ratio_df, metrics_df)

        # For each split, find the corresponding label and append to data_splits
        for split_df, compression_value in split_data_slice_number:
            data_splits.append((split_df, compression_value))

    return data_splits

def extract_data_and_labels(data_splits, features_to_include=FEATURE_NAMES):
    all_data = []
    y = []

    # Process each split in data_splits
    for split_df, label in data_splits:
        if label is not None:
            patient_data = split_df[features_to_include]
            all_data.append(patient_data.values)
            y.append(label)
    
    return all_data, y

def preprocess_data(all_data, y, model_type):
    # Pad each patient's data to have the same number of rows
    max_length = max([data.shape[0] for data in all_data])
    padded_data = [np.pad(data, ((0, max_length - data.shape[0]), (0, 0)), 'constant', constant_values=0) for data in all_data]

    # Flatten the data for ensemble models
    if model_type in ['random_forest', 'gradient_boosting']:
        X = np.array([data.flatten() for data in padded_data])
    else:
        X = np.stack(padded_data, axis=0)

    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the labels to numpy arrays
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # Print the shapes of the data
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', len(y_train))
    print('y_test shape:', len(y_test))

    return X_train, X_test, y_train, y_test

