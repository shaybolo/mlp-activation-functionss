import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize the data
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define function to build the MLP model
def build_mlp_model(activation_func):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(32, 32, 3)))  # Flatten the input
    model.add(layers.Dense(512, activation=activation_func))  # First hidden layer
    model.add(layers.Dense(256, activation=activation_func))  # Second hidden layer
    model.add(layers.Dense(10, activation='softmax'))  # Output layer for classification
    return model

# Build three different models with ReLU, LeakyReLU, and Sigmoid
model_relu = build_mlp_model('relu')
model_leaky_relu = build_mlp_model(LeakyReLU(alpha=0.01))  # LeakyReLU needs an instance
model_sigmoid = build_mlp_model('sigmoid')

# Compile the models
model_relu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_leaky_relu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_sigmoid.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the models
history_relu = model_relu.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=64)
history_leaky_relu = model_leaky_relu.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=64)
history_sigmoid = model_sigmoid.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=64)

# Evaluate the models
acc_relu = model_relu.evaluate(test_images, test_labels, verbose=0)[1]
acc_leaky_relu = model_leaky_relu.evaluate(test_images, test_labels, verbose=0)[1]
acc_sigmoid = model_sigmoid.evaluate(test_images, test_labels, verbose=0)[1]

# Print the accuracy of each model
print(f"ReLU Model Accuracy: {acc_relu*100:.2f}%")
print(f"LeakyReLU Model Accuracy: {acc_leaky_relu*100:.2f}%")
print(f"Sigmoid Model Accuracy: {acc_sigmoid*100:.2f}%")
