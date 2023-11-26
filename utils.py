import csv

def extract_patient_id(filename):
    # Split by '/' and then by '_', and take the second and first elements respectively
    return filename.split('/')[1].split('_')[0]


def update_best_configuration(model_type, feature_subset, best_config, best_mse, file_path='results/best_configurations.csv'):
    # Read existing data and initialize a flag for updating
    should_update = True
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            existing_data = {}
            for row in reader:
                existing_model, existing_features, existing_config, existing_mse = row
                existing_mse = float(existing_mse)
                existing_data[(existing_model, existing_features)] = (existing_config, existing_mse)
                # Check if we have a matching model type and feature subset
                if existing_model == model_type and existing_features == str(feature_subset) and best_mse >= existing_mse:
                    should_update = False
    except FileNotFoundError:
        existing_data = {}

    # Update the data if the current score is better
    if should_update:
        existing_data[(model_type, str(feature_subset))] = (str(best_config), best_mse)

    # Write the updated data back to the file
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for (model, features), (config, mse) in existing_data.items():
            writer.writerow([model, features, config, mse])

# Example usage
# update_best_configurations('cnn', best_config, best_scores)


def load_best_configuration(model_type, file_path='results/best_configurations.csv'):
    # Load best configuration of first column = model_type, second column is config dict
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == model_type:
                return eval(row[2])
    raise ValueError(f"Best configuration for {model_type} not found in {file_path}")


# Update the function to check and update MSE only if it's better
def update_mse_with_confidence_interval(model_type, mse, mse_ci, file_path='results/mse_with_confidence_interval.csv'):
    """
    Updates or appends the mean squared error (MSE) and its confidence interval (CI)
    for a given model type in a CSV file, only if the new MSE is better than the existing one.

    :param model_type: The type of the model to update.
    :param mse: The mean squared error of the model.
    :param mse_ci: The confidence interval of the mean squared error.
    :param file_path: The path to the CSV file.
    """
    updated_lines = []
    found = False

    # Read existing data
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line_parts = line.strip().split(',')
                existing_model_type = line_parts[0]  # Extract the model type

                if existing_model_type == model_type:
                    found = True
                    existing_mse = float(line_parts[1])  # Extract existing MSE
                    print(f'Found existing MSE for {model_type}: {existing_mse}')
                    if mse < existing_mse:  # Update only if new MSE is better
                        print(f'Updating MSE for {model_type} from {existing_mse} to {mse}')
                        updated_lines.append(f'{model_type},{mse},{mse_ci}\n')
                    else:
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)
    except FileNotFoundError:
        # File not found, create new
        pass

    # Append new model type if not found
    if not found:
        updated_lines.append(f'{model_type},{mse},{mse_ci}\n')

    # Write updated data back to the file
    with open(file_path, 'w') as file:
        file.writelines(updated_lines)
