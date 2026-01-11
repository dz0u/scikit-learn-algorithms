import tensorflow as tf
from tensorflow.keras import layers, models

# Defining the encoder
def build_encoder():
    input_img = layers.Input(shape=(784,)),
    encoded = layers.Dense(128, activation='relu')(input_img),
    encoded = layers.Dense(64, activation='relu')(encoded)
    return models.Model(input_img, encoded)

# Define the decoder
def build_decoder():
    encoded_input = layers.Input(shape=(64,))
    decoded = layers.Dense(128, activation='relu')(encoded_input)
    decoded = layers.Dense(784, activation='sigmoid')(decoded)
    return models.Model(encoded_input, decoded)

# Build full Autoencoder
encoder = build_encoder()
decoder = build_decoder()

input_img = layers.Input(shape=(784,))
encoded_img = encoder(input_img)
decoded_img = decoder(encoded_img)

autoencoder = models.Model(input_img, decoded_img)
