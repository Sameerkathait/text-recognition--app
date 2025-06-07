# Handwritten Digit Recognition

This project implements a real-time handwritten digit recognition system using the MNIST dataset. The application allows you to draw digits on a canvas and get instant predictions using a trained Convolutional Neural Network (CNN).

## Features

- Real-time digit drawing interface
- CNN model trained on MNIST dataset
- Instant predictions with confidence scores
- Simple and intuitive user interface

## Requirements

- Python 3.7 or higher
- TensorFlow 2.x
- NumPy
- Pillow
- tkinter (usually comes with Python)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. First, train the model by running:
```bash
python train_model.py
```
This will download the MNIST dataset, train the model, and save it as 'mnist_model.h5'.

2. Launch the application:
```bash
python app.py
```

3. Using the application:
   - Draw a digit (0-9) on the white canvas using your mouse
   - Click the "Predict" button to get the prediction
   - Click "Clear" to erase the canvas and draw a new digit

## How it Works

1. The model is a CNN trained on the MNIST dataset
2. When you draw a digit, it's captured from the canvas
3. The image is preprocessed (resized to 28x28 pixels and normalized)
4. The model makes a prediction and displays the result with confidence score

## Model Architecture

The model uses a CNN architecture with:
- 3 Convolutional layers
- MaxPooling layers
- Dense layers
- Softmax output layer for 10 digit classes (0-9) 