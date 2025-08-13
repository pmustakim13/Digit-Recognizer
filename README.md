# Digit-Recognizer
An interactive MNIST digit recognizer built with TensorFlow and Gradio.  Train a CNN on the MNIST dataset, then draw digits in your browser to get real-time predictions with top confidence scores.

# MNIST Digit Recognizer with Gradio

This project demonstrates a convolutional neural network (CNN) trained on the MNIST dataset to classify handwritten digits (0–9). Using [Gradio](https://gradio.app/), you can interactively draw digits in your browser and get instant predictions with confidence scores.

## Features
- **Model**: TensorFlow/Keras CNN trained on MNIST.
- **Interface**: Gradio sketchpad for drawing digits.
- **Predictions**: Displays top 3 predictions with confidence.
- **Real-time processing**: Preprocessing and prediction happen instantly in the browser.

## How It Works
1. Train the CNN using `train.py` (or your training script).
2. Save the trained model as `my_mnist_model.keras`.
3. Run `final.py` to launch the Gradio interface.
4. Draw a digit and view the predictions in real time.

## Installation
```bash
pip install tensorflow gradio numpy
