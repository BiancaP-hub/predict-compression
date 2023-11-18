from define_model import define_model
from define_model import create_model_wrapper
from tensorflow.keras.callbacks import EarlyStopping


def train_model(config, X_train, y_train):
    """
    Train a model with the provided configuration on the given dataset.

    Parameters:
    config (dict): Configuration dictionary containing hyperparameters and model type.
    X_train (ndarray): Training data features.
    y_train (ndarray): Training data labels.

    Returns:
    Trained model.
    """
    model_type = config['model_type']
    input_shape = X_train.shape[1:] if model_type in ['cnn', 'lstm', 'cnn_lstm'] else None

    if model_type in ['cnn', 'lstm', 'cnn_lstm']:
        model = create_model_wrapper(config, input_shape)

        # Add EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train the model with early stopping
        model.fit(X_train, y_train, epochs=1000, batch_size=32, callbacks=[early_stopping], validation_split=0.2)
    else:
        model = define_model(config, input_shape)
        model.fit(X_train, y_train)

    return model
