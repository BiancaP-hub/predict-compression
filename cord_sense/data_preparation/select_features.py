from sklearn.metrics import mean_squared_error
from cord_sense.common.constants import FEATURE_NAMES
from cord_sense.common.utils import update_best_configuration
from prepare_data import preprocess_data, extract_data_and_labels
from cord_sense.modeling_pipeline.train_model import train_model
from cord_sense.modeling_pipeline.tune_hyperparameters import optimize_hyperparameters
import tensorflow as tf

def forward_feature_selection(data_splits, model_type, feature_names=FEATURE_NAMES, stop_on_no_improvement=True):
    selected_features = []
    best_mse = float('inf')
    best_overall_config = None
    remaining_features = feature_names.copy()

    while remaining_features:
        iteration_improvement = False
        iteration_best_mse = float('inf')
        iteration_best_config = None
        best_feature_this_iteration = None

        for feature in remaining_features:
            temp_selected_features = selected_features + [feature]
            all_data, y, patient_ids = extract_data_and_labels(data_splits, features_to_include=temp_selected_features)
            X_train, X_test, y_train, y_test, _, _ = preprocess_data(all_data, y, patient_ids, model_type)

            _, best_models = optimize_hyperparameters(X_train, y_train, model_type)
            temp_config = best_models[model_type]
            temp_model, _ = train_model(temp_config, X_train, y_train)
            X_test = tf.convert_to_tensor(X_test)
            y_pred = temp_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            if mse < iteration_best_mse:
                iteration_best_mse = mse
                iteration_best_config = temp_config
                best_feature_this_iteration = feature

        if best_feature_this_iteration and iteration_best_mse < best_mse:
            selected_features.append(best_feature_this_iteration)
            remaining_features.remove(best_feature_this_iteration)
            best_mse = iteration_best_mse
            best_overall_config = iteration_best_config
            update_best_configuration(model_type, selected_features, iteration_best_config, iteration_best_mse)
            print(f"Selected feature: {best_feature_this_iteration}, Iteration best MSE: {iteration_best_mse}, Iteration Best Config: {iteration_best_config}")
            iteration_improvement = True
        else:
            remaining_features.remove(best_feature_this_iteration)

        if not iteration_improvement and stop_on_no_improvement:
            break  # No improvement in this iteration and flag is set

    return selected_features, best_overall_config








