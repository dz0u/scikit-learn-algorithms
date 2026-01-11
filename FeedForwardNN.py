import tensorflow as tf
from tensorflow.keras import layers, models, Input

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Loading Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode labels
encoder = OneHotEncoder()
y = encoder.fit_transform(y)

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Forward-Neural-Network Architecture
model_fnn = models.Sequential([
    Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax'),
    ])

# Compile the model
model_fnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model_fnn.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

