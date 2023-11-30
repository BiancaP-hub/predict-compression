from keras.models import Model
from keras.layers import Input, LSTM
from keras.optimizers import Adam

def create_autoencoder(time_steps, input_dim, latent_dim):
    inputs = Input(shape=(time_steps, input_dim))
    
    # Encoder
    encoded = LSTM(latent_dim, return_sequences=True)(inputs)

    # Decoder
    decoded = LSTM(input_dim, return_sequences=True)(encoded)

    # Autoencoder
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder, encoder

def extract_features_with_autoencoder(X_train, X_test, latent_dim=8):
    autoencoder, encoder = create_autoencoder(X_train.shape[1], X_train.shape[2], latent_dim)
    autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, validation_split=0.2)
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)
    return X_train_encoded, X_test_encoded