import csv

def extract_patient_id(filename):
    # Split by '/' and then by '_', and take the second and first elements respectively
    return filename.split('/')[1].split('_')[0]

def update_best_configuration(model_type, best_config, best_scores, file_path='results/best_configurations.csv'):
    # Read existing data and initialize a flag for updating
    should_update = True
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            existing_data = {}
            for row in reader:
                existing_model, existing_config, existing_mse = row
                existing_data[existing_model] = (existing_config, float(existing_mse))
                if existing_model == model_type and best_scores[model_type] >= existing_mse:
                    should_update = False
    except FileNotFoundError:
        existing_data = {}

    # Update the data if the current score is better
    if should_update:
        existing_data[model_type] = (str(best_config), best_scores[model_type])

    # Write the updated data back to the file
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for model, (config, mse) in existing_data.items():
            writer.writerow([model, config, mse])

# Example usage
# update_best_configurations('cnn', best_config, best_scores)

def load_best_configuration(model_type, file_path='results/best_configurations.csv'):
    # Load best configuration of first column = model_type, second column is config dict
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == model_type:
                return eval(row[1])
    raise ValueError(f"Best configuration for {model_type} not found in {file_path}")