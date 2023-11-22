from ray import tune, train
from ray.tune.schedulers import HyperBandScheduler
from define_model import define_model, create_model_wrapper
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_squared_error
from visualize_results import loss_plot
import tensorflow as tf


def optimize_hyperparameters(X_train, y_train, model_type, num_samples=10):
    """
    Optimize hyperparameters for a specified model type using Hyperband.

    Parameters:
    X_train (ndarray): Training data features.
    y_train (ndarray): Training data labels.
    model_type (str): Model type to optimize.
    num_samples (int): Number of samples to explore in the hyperparameter space.

    Returns:
    Tuple containing the best configuration, best scores, and best models.
    """
    # Initialize best score and model for each model type
    best_scores = {
        'random_forest': float('inf'),
        'gradient_boosting': float('inf'),
        'cnn': float('inf'),
        'lstm': float('inf'),
        'cnn_lstm': float('inf')
    }
    best_models = {
        'random_forest': None,
        'gradient_boosting': None,
        'cnn': None,
        'lstm': None,
        'cnn_lstm': None
    }
    
    search_space = define_search_space(model_type)

    # Configure Hyperband scheduler
    hyperband = HyperBandScheduler(time_attr="training_iteration", metric="mean_squared_error", mode="min")

    # Execute the hyperparameter search
    analysis = tune.run(
        lambda config: evaluate_model(config, X_train, y_train),
        config=search_space,
        num_samples=num_samples,
        scheduler=hyperband
    )

    # Update best_scores and best_models based on the trials
    for trial in analysis.trials:
        trial_config = trial.config
        trial_score = trial.last_result["mean_squared_error"]
        model_type = trial_config['model_type']

        if trial_score < best_scores.get(model_type, float('inf')):
            best_scores[model_type] = trial_score
            best_models[model_type] = trial_config

    return best_scores, best_models


def evaluate_model(config, X_train, y_train):
    """
    Evaluate a model configuration using custom cross-validation, including loss plotting and MSE standard deviation calculation.

    Parameters:
    config (dict): Configuration dictionary containing hyperparameters and model type.
    X_train (ndarray): Training data features used for model training.
    y_train (ndarray): Training data labels used for model training.
    """
    model_type = config['model_type']
    input_shape = X_train.shape[1:] if model_type in ['cnn', 'lstm', 'cnn_lstm'] else None
    mse_scores = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        if model_type in ['cnn', 'lstm', 'cnn_lstm']:
            model = create_model_wrapper(config, input_shape)
            assert model is not None, "Failed to define the model. Check the configuration."

            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            history = model.fit(X_train_fold, y_train_fold, epochs=1000, batch_size=32, validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping], verbose=1)

            # Plot training and validation loss for each fold
            loss_plot(history, model_type)

            # Convert to tensor
            X_val_fold = tf.convert_to_tensor(X_val_fold)

            y_pred = model.predict(X_val_fold)
        else:
            model = define_model(config, input_shape)
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)

        mse = mean_squared_error(y_val_fold, y_pred)
        mse_scores.append(mse)

    # Calculate and return the average and standard deviation of MSE
    average_mse = np.mean(mse_scores)
    std_dev_mse = np.std(mse_scores)
    print(f'Average MSE: {average_mse} +/- {std_dev_mse}')

    train.report({'mean_squared_error': average_mse})


def define_search_space(model_type):
    # Base search_space including 'model_type'
    search_space = {'model_type': tune.choice([model_type])}

    if model_type == 'cnn':
        search_space.update({
            'learning_rate': tune.choice([0.0001, 0.001, 0.01, 0.1]),
            'conv_units': tune.choice([32, 64, 128]),
            'dense_units': tune.choice([[50], [100, 50]])
        })

    elif model_type == 'lstm':
        search_space.update({
            'learning_rate': tune.choice([0.0001, 0.001, 0.01, 0.1]),
            'lstm_units': tune.choice([[64], [128, 64]]),
            'dense_units': tune.choice([[50], [100, 50]])
        })

    elif model_type == 'cnn_lstm':
        search_space.update({
            'learning_rate': tune.choice([0.0001, 0.001, 0.01, 0.1]),
            'conv_units': tune.choice([32, 64]),
            'lstm_units': tune.choice([[64], [128, 64]]),
            'dense_units': tune.choice([[50], [100, 50]])
        })

    elif model_type == 'random_forest':
        search_space.update({
            'n_estimators': tune.choice([100, 200, 300]),
            'max_depth': tune.choice([10, 20, 30]),
            'min_samples_split': tune.choice([2, 5, 10]),
            'min_samples_leaf': tune.choice([1, 2, 4])
        })

    elif model_type == 'gradient_boosting':
        search_space.update({
            'n_estimators': tune.choice([100, 200, 300]),
            'learning_rate': tune.choice([0.01, 0.1, 0.2]),
            'max_depth': tune.choice([3, 5, 10])
        })

    return search_space
