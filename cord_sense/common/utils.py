import csv
import pandas as pd
import numpy as np
import os

def extract_patient_id(filename):
    # Split by '/' and then by '_', and take the second and first elements respectively
    return filename.split('/')[1].split('_')[0]

# Make sure results folder exists and create it if not
def ensure_results_folder():
    if not os.path.exists('results'):
        os.makedirs('results')


def update_best_configuration(model_type, feature_subset, best_config, best_mse, file_path='results/best_configurations.csv'):
    """
    Updates or appends the best configuration for a given model type and feature subset in a CSV file,
    only if the new MSE is better than the existing one.

    :param model_type: The type of the model to update.
    :param feature_subset: The subset of features used for training.
    :param best_config: The best configuration of the model.
    :param best_mse: The mean squared error of the model.
    :param file_path: The path to the CSV file.
    """
    # Make sure results folder exists
    ensure_results_folder()

    # Define the header
    header = ['Model Type', 'Feature Subset', 'Best Configuration', 'MSE']
    
    # Read existing data and initialize a flag for updating
    should_update = True
    existing_data = {}
    
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            # Check if the file has a header
            if reader.fieldnames != header:
                # If not, treat it as a regular row and reset the flag
                should_update = False
                existing_data = {(row['Model Type'], row['Feature Subset']): (row['Best Configuration'], float(row['MSE'])) for row in reader}
            else:
                # Process the file with the header
                for row in reader:
                    existing_model = row['Model Type']
                    existing_features = row['Feature Subset']
                    existing_config = row['Best Configuration']
                    existing_mse = float(row['MSE'])
                    existing_data[(existing_model, existing_features)] = (existing_config, existing_mse)
                    if existing_model == model_type and existing_features == str(feature_subset) and best_mse >= existing_mse:
                        should_update = False
    except FileNotFoundError:
        # File not found, create new
        pass

    # Update the data if the current score is better
    if should_update:
        existing_data[(model_type, str(feature_subset))] = (str(best_config), best_mse)

    # Write the updated data back to the file
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for (model, features), (config, mse) in existing_data.items():
            writer.writerow({'Model Type': model, 'Feature Subset': features, 'Best Configuration': config, 'MSE': mse})

# Example usage:
# update_best_configuration('cnn_lstm', ['feature1', 'feature2'], best_config, best_mse)


def load_best_configuration(model_type, file_path='results/best_configurations.csv'):
    """
    Loads the best configuration for a given model type from a CSV file.

    :param model_type: The type of the model to load.
    :param file_path: The path to the CSV file.
    :return: The best configuration of the model.
    """
    # Load best configuration of first column = model_type, second column is config dict
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == model_type:
                return eval(row[2])
    raise ValueError(f"Best configuration for {model_type} not found in {file_path}")


# Update the function to check and update MSE only if it's better
def update_mse_with_confidence_interval(model_type, mse, mse_ci, mse_std_dev, baseline_mse, file_path='results/mse_with_confidence_interval.csv'):
    """
    Updates or appends the mean squared error (MSE), its confidence interval (CI), and the baseline MSE
    for a given model type in a CSV file, only if the new MSE is better than the existing one.

    :param model_type: The type of the model to update.
    :param mse: The mean squared error of the model.
    :param mse_ci: The confidence interval of the mean squared error.
    :param baseline_mse: The baseline mean squared error.
    :param file_path: The path to the CSV file.
    """
    # Make sure results folder exists
    ensure_results_folder()

    updated_lines = []
    found = False
    header = 'Model Type,MSE,MSE Confidence Interval,MSE Standard Deviation,Baseline MSE\n'

    # Read existing data
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Check if the first line is the header
            if lines[0].strip() != header.strip():
                updated_lines.append(header)  # Add header if not present
            else:
                updated_lines.append(lines[0])  # Keep existing header

            for line in lines[1:]:  # Start from the second line to skip existing header
                line_parts = line.strip().split(',')
                existing_model_type = line_parts[0]

                if existing_model_type == model_type:
                    found = True
                    existing_mse = float(line_parts[1])
                    if mse < existing_mse:
                        updated_lines.append(f'{model_type},{mse},{mse_ci},{mse_std_dev},{baseline_mse}\n')
                    else:
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)
    except FileNotFoundError:
        updated_lines.append(header)  # Add header for a new file

    # Append new model type if not found
    if not found:
        updated_lines.append(f'{model_type},{mse},{mse_ci},{mse_std_dev},{baseline_mse}\n')

    # Write updated data back to the file
    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

# Example usage:
# update_mse_with_confidence_interval('cnn_lstm', mse, mse_ci, baseline_mse)


def save_predictions(patient_ids, actual_values, predicted_values, model_type, file_path='results/predictions.csv'):
    """
    Updates or appends the actual and predicted compression values for each patient and model type to a CSV file.

    :param patient_ids: A list of patient IDs.
    :param actual_values: The actual compression values.
    :param predicted_values: The predicted compression values.
    :param model_type: The type of the model used for predictions.
    :param file_path: The path to the CSV file where results will be saved.
    """

    # Make sure results folder exists
    ensure_results_folder()

    predicted_values = ensure_1d(predicted_values)

    # Create a DataFrame for the new results
    new_results_df = pd.DataFrame({
        'Patient_ID': patient_ids,
        'Actual_Value': actual_values,
        'Predicted_Value': predicted_values,
        'Model_Type': model_type
    })

    try:
        # Read existing data from the file
        existing_results_df = pd.read_csv(file_path)

        # Filter out the rows for the current model type
        filtered_df = existing_results_df[existing_results_df['Model_Type'] != model_type]

        # Concatenate the filtered data with the new results
        updated_results_df = pd.concat([filtered_df, new_results_df], ignore_index=True)
    except FileNotFoundError:
        # If the file doesn't exist, use the new results as the updated data
        updated_results_df = new_results_df

    # Save updated results to CSV file
    updated_results_df.to_csv(file_path, index=False)


def ensure_1d(predictions):
    """
    Ensure that the predictions are in a 1-dimensional array format.

    :param predictions: The predicted values, which can be either 1D or 2D numpy arrays.
    :return: A 1D numpy array of predictions.
    """
    # Convert to numpy array if not already
    predictions = np.array(predictions)

    # Check if predictions are already 1D
    if predictions.ndim == 1:
        return predictions
    # Flatten 2D predictions to 1D
    elif predictions.ndim == 2 and predictions.shape[1] == 1:
        return predictions.flatten()
    else:
        raise ValueError("Predictions are not in a valid format")


def get_statistics_on_patient_data(data_splits, ap_ratio_df):
    unique_patients = set(patient for _, _, patient in data_splits)
    num_unique_patients = len(unique_patients)

    total_splits = len(data_splits)
    splits_with_labels = sum(1 for _, label, _ in data_splits if label != 0)
    splits_without_labels = total_splits - splits_with_labels

    patients_with_compression = set(patient for _, label, patient in data_splits if label != 0)
    patients_without_compression = unique_patients - patients_with_compression

    num_patients_with_compression = len(patients_with_compression)
    num_patients_without_compression = len(patients_without_compression)

    # Aggregating MSCC values across all patients
    mscc_values = ap_ratio_df['diameter_AP_ratio_PAM50_normalized'].dropna()
    min_mscc = mscc_values.min() if not mscc_values.empty else None
    max_mscc = mscc_values.max() if not mscc_values.empty else None
    mean_mscc = mscc_values.mean() if not mscc_values.empty else None

    # Max number of MSCC values for a single patient
    patient_mscc_counts = ap_ratio_df.groupby('filename')['diameter_AP_ratio_PAM50_normalized'].count()
    max_mscc_per_patient = patient_mscc_counts.max()

    return {
        "num_unique_patients": num_unique_patients,
        "total_splits": total_splits,
        "splits_with_labels": splits_with_labels,
        "splits_without_labels": splits_without_labels,
        "max_mscc_per_patient": max_mscc_per_patient,
        "min_mscc_across_patients": min_mscc,
        "max_mscc_across_patients": max_mscc,
        "mean_mscc_across_patients": mean_mscc,
        "num_patients_with_compression": num_patients_with_compression,
        "num_patients_without_compression": num_patients_without_compression
    }






