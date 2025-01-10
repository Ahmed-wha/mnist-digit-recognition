# MNIST Digit Recognition

This project implements a simple neural network using Keras to classify handwritten digits from the MNIST dataset. The model is built, trained, evaluated, and tested on the dataset, demonstrating the foundations of neural network development and image classification.

## Features
- **Data Preprocessing**: Normalized pixel values and visualized the dataset.
- **Model Building**: Created a neural network using Keras with two hidden layers.
- **Training and Evaluation**: Trained the model with a validation split and evaluated its performance.
- **Visualization**: Displayed model predictions alongside true labels for test data.
- **Model Saving**: Saved the trained model for future use.

## Technologies
- **Python**: Programming language used for implementation.
- **Keras**: High-level deep learning API for building and training the model.
- **TensorFlow**: Backend for Keras.
- **NumPy**: Numerical computations for data manipulation.
- **Matplotlib**: Visualization library for displaying images and predictions.

## Dataset
The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is a benchmark dataset for handwritten digit classification. It consists of:
- **Training Set**: 60,000 grayscale images of digits (28x28 pixels) with labels (0-9).
- **Test Set**: 10,000 grayscale images.

## How to Run

### Prerequisites
Make sure you have Python installed and the following libraries available:
- TensorFlow
- NumPy
- Matplotlib

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ahmed-wha/mnist-digit-recognition.git
2. **Navigate to project directory**
3. **Install required libraries**:
   ```bash
   pip install -r requirements.txt
4.**Run the script**


## Results
The trained neural network achieves approximately 97.55% accuracy on the test dataset.
