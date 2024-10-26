import tensorflow as tf
import numpy as np


# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of the model
    # image = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    image = image.resize((224, 224))

    # Convert to numpy array
    image = tf.keras.preprocessing.image.img_to_array(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Normalize the image
    # image = image / 255.0
    image = tf.keras.applications.mobilenet_v3.preprocess_input(image)

    return image


def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    return interpreter


# Function to make predictions
def predict(image, class_names, interpreter):
    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Set the input tensor
    interpreter.set_tensor(input_tensor_index, preprocessed_image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    predicted_class = interpreter.get_tensor(output_tensor_index)

    # Get the predicted class
    predicted_class_name = class_names[int(predicted_class.argmax())]

    probability = np.max(predicted_class) * 100

    return predicted_class_name, probability
