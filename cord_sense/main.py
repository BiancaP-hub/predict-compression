# Main execution file for the project
import os
from cord_sense.common.request_to_download import download_file_from_github
from cord_sense.common.constants import MODEL_TYPES
import cord_sense.image_processing.process_segmentation # leave
from cord_sense.data_preparation.prepare_data import split_patient_samples, extract_data_and_labels, preprocess_data
from cord_sense.data_preparation.select_features import forward_feature_selection
from cord_sense.modeling_pipeline.train_model import train_model
from cord_sense.metrics.confidence_interval import bootstrap_confidence_interval
from cord_sense.modeling_pipeline.save_model import save_model
from cord_sense.metrics.mse_with_constant_prediction import calculate_baseline_mse
from cord_sense.common.utils import update_mse_with_confidence_interval, save_predictions
from sklearn.metrics import mean_squared_error
from cord_sense.metrics.visualize_results import residuals_plot, actual_vs_predicted_plot, loss_plot

# Add argument to script to pass the path of the local data-multi-subject repository
import argparse
parser = argparse.ArgumentParser()
# Add dataset_dir argument with explicit warning if not provided
parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the local data-multi-subject repository')
parser.add_argument('--model_type', type=str, choices=MODEL_TYPES, default='random_forest')
args = parser.parse_args()


def main(dataset_dir, model_type):
    # Set dataset directory as global variable for other scripts in this project to use
    os.environ['DATASET_DIR'] = dataset_dir

    github_file_url = "https://raw.githubusercontent.com/spine-generic/data-multi-subject/master"
    participants_file_path = "participants.tsv"
    local_data_dir = "data"
    local_participants_path = os.path.join(local_data_dir, participants_file_path)

    os.makedirs(local_data_dir, exist_ok=True)
    download_file_from_github(os.path.join(github_file_url, participants_file_path), local_participants_path)

    # Not working yet
    # participants = pd.read_csv(local_participants_path, sep='\t')
    # local_t2w_dir = os.path.join(local_data_dir, "t2w")

    # for participant_id in participants['participant_id']:
    #     remote_file_path = f"{participant_id}/anat/{participant_id}_T2w.nii.gz"
    #     os.makedirs(local_t2w_dir, exist_ok=True)
    #     download_file_from_github(os.path.join(github_file_url, remote_file_path), 
    #                               os.path.join(local_t2w_dir, f"{participant_id}_T2w.nii.gz"))

    # # Initialize and retrieve files with git-annex
    # install_git_annex()
    # initialize_git_annex(local_t2w_dir)
    # retrieve_files_with_git_annex()

    # Load the dataset
    data_splits = split_patient_samples()

    # Forward feature selection
    selected_features, best_overall_config = forward_feature_selection(data_splits, model_type, stop_on_no_improvement=True)

    # Final model training with selected features and best configuration
    all_data, y, patient_ids = extract_data_and_labels(data_splits, features_to_include=selected_features)
    X_train, X_test, y_train, y_test, train_ids, test_ids = preprocess_data(all_data, y, patient_ids, model_type)
    final_model, history = train_model(best_overall_config, X_train, y_train)

    # Plot the loss
    if model_type in ['cnn', 'lstm', 'cnn_lstm']:
        loss_plot(history, model_type)

    # Save the final model
    save_model(final_model, model_type)

    # Generate predictions
    y_pred = final_model.predict(X_test)

    # Save the actual values and predictions to a CSV file
    save_predictions(test_ids, y_test, y_pred, model_type)

    # Calculate MSE and baseline MSE for comparison
    mse = mean_squared_error(y_test, y_pred)
    baseline_mse = calculate_baseline_mse(y_train, y_test)

    # Bootstrap confidence interval for the MSE
    mse_ci, mse_std_dev = bootstrap_confidence_interval(final_model, X_test, y_test)

    # Print the baseline MSE, MSE, and MSE confidence interval
    print(f"Baseline MSE (predicting mean): {baseline_mse}")
    print(f'MSE: {mse}')
    print(f'MSE confidence interval: {mse_ci}')
    print(f'MSE Standard Deviation: {mse_std_dev}')

    # Save the results to a CSV file
    update_mse_with_confidence_interval(model_type, mse, mse_ci, mse_std_dev, baseline_mse)

    # Visualization of the results
    residuals_plot(y_test, y_pred, model_type)
    actual_vs_predicted_plot(y_test, y_pred, model_type)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.dataset_dir, args.model_type)