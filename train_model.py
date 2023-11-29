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
    nb_epochs = 1000
    if model_type == 'cnn':
        nb_epochs = 2000

    if model_type in ['cnn', 'lstm', 'cnn_lstm']:
        model = create_model_wrapper(config, input_shape)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Save the history of training
        history = model.fit(X_train, y_train, epochs=nb_epochs, batch_size=32, callbacks=[early_stopping], validation_split=0.2)

        # Return both model and history
        return model, history
    else:
        model = define_model(config, input_shape)
        model.fit(X_train, y_train)
        return model, None
