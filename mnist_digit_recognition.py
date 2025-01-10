"""
MNIST Digit Recognition using Keras
Description:
    This script demonstrates the implementation of a simple neural network using Keras
    to classify handwritten digits from the MNIST dataset. It includes model training,
    evaluation, and visualization of predictions.
Dependencies: TensorFlow, Matplotlib, NumPy
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

def main():
    # Load the MNIST dataset
    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    except Exception as e:
        print(f"Error loading MNIST dataset: {e}")
        exit()

    # Normalize input features to 0-1 range
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Visualize the first 10 instances
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train[i], cmap="gray")
        plt.title(f"Label: {y_train[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Build the neural network model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    # Display the model's architecture
    model.summary()

    # Compile the model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="SGD",
        metrics=["accuracy"]
    )

    # Train the model with a validation split
    model.fit(x_train, y_train, epochs=20, validation_split=0.1)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Save the trained model
    model.save("mnist_digit_recognition_model.keras")

    # Test the model: Predict labels for the first 10 instances in the test set
    y_pred = model.predict(x_test[:10])

    # Visualize predictions alongside true labels
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i], cmap="gray")
        plt.title(f"True: {y_test[i]}\nPred: {np.argmax(y_pred[i])}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()