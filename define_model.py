from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dense, LSTM
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import tensorflow as tf

def define_model(config, input_shape=None):
    """
    Define a model (neural network or ensemble) based on the specified type and parameters.

    Parameters:
    config (dict): Configuration parameters for the model.
    input_shape (tuple, optional): Shape of the input data for neural networks.

    Returns:
    Model instance (Keras or Scikit-learn model).
    """
    model_type = config['model_type']

    if model_type in ['cnn', 'lstm', 'cnn_lstm']:
        model = Sequential()

        if model_type == 'cnn':
            model.add(Conv1D(config['conv_units'], kernel_size=3, activation='relu', input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())

        elif model_type in ['lstm', 'cnn_lstm']:
            if model_type == 'cnn_lstm':
                model.add(Conv1D(config['conv_units'], kernel_size=3, activation='relu', input_shape=input_shape))
                model.add(BatchNormalization())
                model.add(MaxPooling1D(pool_size=2))

            lstm_units = config['lstm_units']
            for i, units in enumerate(lstm_units):
                return_sequences = i < len(lstm_units) - 1  # True if not the last LSTM layer
                model.add(LSTM(units, activation='tanh', return_sequences=return_sequences))
                model.add(BatchNormalization())

        for units in config['dense_units']:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
        model.add(Dense(1))

    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"],
            random_state=42
        )

    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            random_state=42
        )

    else:
        raise ValueError("Invalid model type. Options: 'cnn', 'lstm', 'cnn_lstm', 'random_forest', 'gradient_boosting'.")

    return model

def create_model_wrapper(config, input_shape):
    """
    Wrapper function that creates and compiles a Keras model.

    Parameters:
    config (dict): Configuration parameters for the model.
    input_shape (tuple): Shape of the input data for neural networks.

    Returns:
    Compiled Keras model.
    """
    model = define_model(config, input_shape)
    
    # Ensure the model is defined
    if model is None:
        raise ValueError("Failed to define the model. Check the configuration.")

    # Compile the model if it's a neural network model
    if config['model_type'] in ['cnn', 'lstm', 'cnn_lstm']:
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Ensure the model is compiled
    if not hasattr(model, 'compile') or model.optimizer is None:
        raise RuntimeError("The model is not compiled properly. Check the model definition and compilation.")

    return model
