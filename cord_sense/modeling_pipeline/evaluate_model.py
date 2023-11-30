import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from ray import train
from cord_sense.modeling_pipeline.define_model import define_model, create_model_wrapper
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from cord_sense.config.config import suppress_stdout

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

            if np.isnan(X_train_fold).any() or np.isnan(X_val_fold).any() or np.isnan(y_train_fold).any() or np.isnan(y_val_fold).any():
                print("NaN values found in the dataset")
                continue

            history = model.fit(X_train_fold, y_train_fold, epochs=1000, batch_size=32, validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping], verbose=1)

            # Convert to tensor
            X_val_fold = tf.convert_to_tensor(X_val_fold)

            y_pred = model.predict(X_val_fold)

            if np.isnan(y_pred).any():
                print("NaN values found in predictions")
                continue
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

    with suppress_stdout():
        train.report({'mean_squared_error': average_mse})