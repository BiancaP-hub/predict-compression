import joblib
import os
from define_model import create_model_wrapper

best_model_paths = {
    'random_forest': 'models/best_random_forest.pkl',
    'gradient_boosting': 'models/best_gradient_boosting.pkl',
    'cnn': 'models/best_cnn.h5',
    'lstm': 'models/best_lstm.h5',
    'cnn_lstm': 'models/best_cnn_lstm.h5',
    'xgb_regressor': 'models/best_xgb_regressor.pkl',
    'stacked_rf_gb': 'models/best_stacked_rf_gb.pkl'
}

def save_model(model, model_type):
    """
    Save the trained model to the specified path.

    Parameters:
    model: Trained model to be saved.
    model_path (str): Path where the model should be saved.
    model_type (str): Type of the model ('cnn', 'lstm', 'cnn_lstm', 'random_forest', 'gradient_boosting').
    """
    if not os.path.exists(os.path.dirname(best_model_paths[model_type])):
        os.makedirs(os.path.dirname(best_model_paths[model_type]))
    
    if model_type in ['cnn', 'lstm', 'cnn_lstm']:
        model.save(best_model_paths[model_type])
    else:
        joblib.dump(model, best_model_paths[model_type])
    print(f"Model saved to {best_model_paths[model_type]}")


def load_model(model_type):
    """
    Load the trained model from the specified path.

    Parameters:
    model_path (str): Path where the model should be loaded from.
    model_type (str): Type of the model ('cnn', 'lstm', 'cnn_lstm', 'random_forest', 'gradient_boosting').

    Returns:
    Trained model.
    """
    # Add a check for the path existence
    if not os.path.exists(best_model_paths[model_type]):
        raise FileNotFoundError(f"Model file not found at {best_model_paths[model_type]}")
    
    if model_type in ['cnn', 'lstm', 'cnn_lstm']:
        model = create_model_wrapper({'model_type': model_type}, None)
        model.load_weights(best_model_paths[model_type])
    else:
        model = joblib.load(best_model_paths[model_type])
    return model