# import gradio as gr
# import tensorflow as tf
# import numpy as np

# model = tf.keras.models.load_model('my_mnist_model.keras')

# def predict_digit(drawing_dict):
#     try:
#         # Check if the input is None
#         if drawing_dict is None:
#             return "Please draw a digit first."

#         image = drawing_dict['composite']
#         processed_image = tf.image.rgb_to_grayscale(image)
#         processed_image = 255 - processed_image
#         processed_image = tf.image.resize(processed_image, (28, 28))
#         processed_image = processed_image / 255.0
#         processed_image = tf.reshape(processed_image, (1, 28, 28, 1))

#         # Make prediction
#         prediction = model.predict(processed_image)
#         confidences = {str(i): float(prediction[0][i]) for i in range(10)}
#         return confidences

#     except Exception as e:
#         return str(e)
    
# interface = gr.Interface(
#     fn=predict_digit,
#     inputs=gr.Sketchpad(type="numpy"),
#     outputs=gr.Label(num_top_classes=3),
#     title="MNIST Digit Recognizer - Final Version",
#     description="Draw a digit (0-9) to get a prediction. This version correctly handles the sketchpad input."
# )



# print("Launching the final Gradio interface...")
# interface.launch()


# import gradio as gr
# import tensorflow as tf
# import numpy as np

# # Load trained model
# model = tf.keras.models.load_model('my_mnist_model.keras')

# def predict_digit(drawing_dict):
#     try:
#         if drawing_dict is None:
#             return "Please draw a digit first."

#         # Extract image from Sketchpad
#         image = drawing_dict['composite']

#         # Remove alpha channel if present (RGBA → RGB)
#         if image.shape[-1] == 4:
#             image = image[..., :3]

#         # Convert to grayscale
#         processed_image = tf.image.rgb_to_grayscale(image)

#         # Invert colors (Sketchpad draws black on transparent/white background)
#         processed_image = 255 - processed_image

#         # Resize to 28x28
#         processed_image = tf.image.resize(processed_image, (28, 28))

#         # Normalize to 0-1
#         processed_image = processed_image / 255.0

#         # Reshape for model input
#         processed_image = tf.reshape(processed_image, (1, 28, 28, 1))

#         # Debug shape
#         print("Processed image shape:", processed_image.shape)

#         # Predict
#         prediction = model.predict(processed_image)
#         confidences = {str(i): float(prediction[0][i]) for i in range(10)}
#         return confidences

#     except Exception as e:
#         return str(e)

# interface = gr.Interface(
#     fn=predict_digit,
#     inputs=gr.Sketchpad(type="numpy"),
#     outputs=gr.Label(num_top_classes=3),
#     title="MNIST Digit Recognizer - Final Version",
#     description="Draw a digit (0-9) to get a prediction."
# )

# print("Launching the final Gradio interface...")
# interface.launch()

import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('my_mnist_model.keras')

def preprocess_image(image):
    # Remove alpha channel if present
    if image.shape[-1] == 4:
        image = image[..., :3]

    # Convert to grayscale
    gray = tf.image.rgb_to_grayscale(image)

    # Invert colors to match MNIST's white-on-black style
    gray = 255 - gray

    # Remove last channel so shape is (H, W)
    gray_np = tf.squeeze(gray).numpy().astype(np.uint8)

    # Find bounding box of the digit
    coords = np.column_stack(np.where(gray_np > 0))
    if coords.size == 0:
        return np.zeros((28, 28, 1), dtype=np.float32)  # blank if nothing drawn

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop to bounding box
    cropped = gray_np[y_min:y_max+1, x_min:x_max+1]

    # Resize cropped digit to 20×20
    resized = tf.image.resize(cropped[..., np.newaxis], (20, 20)).numpy()

    # Pad to 28×28
    padded = np.pad(resized, ((4, 4), (4, 4), (0, 0)), mode='constant', constant_values=0)

    # Normalize 0–1
    padded = padded / 255.0

    return padded


def predict_digit(drawing_dict):
    if drawing_dict is None:
        return "Please draw a digit first."

    processed_image = preprocess_image(drawing_dict['composite'])
    processed_image = processed_image.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(processed_image)
    confidences = {str(i): float(prediction[0][i]) for i in range(10)}
    return confidences

interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="MNIST Digit Recognizer - Centered & Scaled",
    description="Draw a digit (0-9). This version preprocesses to match MNIST format."
)

print("Launching the final Gradio interface...")
interface.launch()
