from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, LSTM
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from keras.layers import BatchNormalization

def define_model(model_type, input_shape=None):
    if model_type == 'cnn':
        model = Sequential()
        model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))
        return model

    elif model_type == 'lstm':
        model = Sequential()
        model.add(LSTM(64, activation='tanh', input_shape=input_shape, return_sequences=True))
        model.add(BatchNormalization())
        model.add(LSTM(32, activation='tanh'))
        model.add(BatchNormalization())
        model.add(Dense(50, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))
        return model

    elif model_type == 'cnn_lstm':
        model = Sequential()
        model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(32, activation='tanh', return_sequences=True))
        model.add(BatchNormalization())
        model.add(LSTM(16, activation='tanh'))
        model.add(BatchNormalization())
        model.add(Dense(50, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))
        return model

    elif model_type == 'random_forest':
        return RandomForestRegressor(n_estimators=100)

    elif model_type == 'gradient_boosting':
        return GradientBoostingRegressor(n_estimators=100)

    else:
        raise ValueError("Invalid model type. Options: 'cnn', 'lstm', 'cnn_lstm', 'random_forest', 'gradient_boosting'.")

