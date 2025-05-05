import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Paths to dataset
data_dir = "Dataset"
train_dir = os.path.join(data_dir, 'train')

# Load the pre-trained model
@st.cache_resource
def load_trained_model():
    model = load_model('my_model.h5')
    return model

model = load_trained_model()

# Load and preprocess data
@st.cache_resource
def load_data(data_dir):
    X, Y = [], []
    labels = {'0': 0, '1': 1}  # Folder names 0 and 1 correspond to labels
    for label, idx in labels.items():
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            if file.endswith('.jpg'):
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                X.append(img_resized)
                Y.append(idx)
    X = np.array(X) / 255.0  # Normalize data
    Y = to_categorical(np.array(Y), 2)  # Convert labels to categorical
    return X, Y

# Load the training data
X_train, Y_train = load_data(train_dir)

# Evaluate model accuracy
@st.cache_resource
def evaluate_model():
    loss, accuracy = model.evaluate(X_train, Y_train, verbose=0)
    return accuracy * 100  # Convert to percentage

model_accuracy = evaluate_model()

# Prediction function
def predict_image(image_path):
    image = cv2.imread(image_path)
    img_resized = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    return predicted_class, confidence, image

# Streamlit app
st.title("CANCER DETECTION")
st.write("This app classifies medical images as Normal or Infected.")

# Display model accuracy
st.subheader("Model Accuracy")
st.write(f"**The model's accuracy is {model_accuracy:.2f}%**")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = "temp_image.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict the class of the image
    predicted_class, confidence, original_image = predict_image(temp_file_path)

    # Display the uploaded image
    st.image(original_image, channels="BGR", caption="Uploaded Image", use_container_width=True)

    # Show the prediction results
    st.subheader("Prediction Results")
    labels = ['Normal', 'Infected']
    st.write(f"**Prediction:** {labels[predicted_class]}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
