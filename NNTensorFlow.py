import tensorflow as tf
from tensorflow.keras import layers, models

# Loading dataset (Fashion MNIST)
(train_images, train_activityels), (test_images, test_activityels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 225.0

# Define the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)), # every image is 28x28
    layers.Dense(128, activation='relu'), # RELU activation algorithm (hidden layer)
    layers.Dense(10, activation='softmax') #output layer, softmax algorithm
    ])

# Compile the model, optimizer is Adam
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # classification
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_activityels, epochs=1, batch_size=16) # commonly about 10, 32

# Evaluate model on test set
test_loss, test_acc = model.evaluate(test_images, test_activityels)

print(f'Test accuracy: {test_acc}')



