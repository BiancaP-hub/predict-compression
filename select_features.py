from sklearn.metrics import mean_squared_error
from constants import FEATURE_NAMES
from utils import update_best_configuration
from prepare_data import preprocess_data, extract_data_and_labels
from train_model import train_model
from tune_hyperparameters import optimize_hyperparameters

def forward_feature_selection(data_splits, model_type, feature_names=FEATURE_NAMES):
    selected_features = []
    best_mse = float('inf')
    best_overall_config = None
    remaining_features = feature_names.copy()

    while remaining_features:
        best_feature = None
        best_config_for_feature = None

        for feature in remaining_features:
            temp_selected_features = selected_features + [feature]
            all_data, y = extract_data_and_labels(data_splits, features_to_include=temp_selected_features)
            X_train, X_test, y_train, y_test = preprocess_data(all_data, y, model_type)

            _, best_models = optimize_hyperparameters(X_train, y_train, model_type)
            temp_config = best_models[model_type]
            temp_model = train_model(temp_config, X_train, y_train)
            y_pred = temp_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            if mse < best_mse:
                best_mse = mse
                best_feature = feature
                best_config_for_feature = temp_config

        if best_feature:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_overall_config = best_config_for_feature
            update_best_configuration(model_type, selected_features, best_config_for_feature, best_mse)
            print(f"Selected feature: {best_feature}, Current best MSE: {best_mse}, Best Config: {best_config_for_feature}")
        else:
            break  # No improvement

    return selected_features, best_overall_config




