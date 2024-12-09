import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from PIL import Image
import os


# Helper Functions
def load_and_preprocess_data(image_dir, csv_file):
    """
    Preprocess the dataset based on images in a directory and their labels in the CSV.
    Assumes CSV contains paths and labels.
    """
    df = pd.read_csv(csv_file)
    
    # Map images and labels
    images = []
    labels = []
    
    # Map disease categories
    DISEASE_MAPPING = {
        "Melanoma": 0,
        "Basal Cell Carcinoma": 1,
        "Squamous Cell Carcinoma": 2,
        "Benign Lesion": 3,
    }

    for index, row in df.iterrows():
        try:
            img_path = os.path.join(image_dir, row["image_id"])
            # Open the image if the path exists
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB').resize((128, 128))
                img = np.array(img) / 255.0  # Normalize
                images.append(img)
                # Map disease type from the CSV
                labels.append(DISEASE_MAPPING[row["dx"]])
        except Exception as e:
            print(f"Skipping file: {row['image_id']}, Error: {e}")
    
    X = np.array(images)  # Input features
    y = np.array(labels)  # Target classes
    y = tf.keras.utils.to_categorical(y, num_classes=len(DISEASE_MAPPING))  # One-hot encoding
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def create_and_train_cnn_model(X_train, y_train):
    """
    Create and train a CNN model for classification.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(y_train.shape[1], activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train model
    model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=32, verbose=2)

    # Save the model after training
    model.save('trained_skin_cancer_cnn_model.h5')
    st.success("✅ Model trained and saved successfully!")

    return model


def preprocess_uploaded_image(image_file):
    """
    Preprocess the uploaded image into input expected by CNN model.
    """
    try:
        # Open and preprocess the image
        image = Image.open(image_file).convert('RGB').resize((128, 128))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)  # Make it batch-ready
        return image
    except Exception as e:
        st.error(f"Error preprocessing the image: {e}")
        return None


def run_prediction(image_file):
    """
    Perform a prediction on the uploaded image using the trained CNN model.
    """
    model = tf.keras.models.load_model('trained_skin_cancer_cnn_model.h5')
    img = preprocess_uploaded_image(image_file)

    if img is None:
        return None, None

    try:
        # Perform prediction
        predictions = model.predict(img)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]

        # Map index back to disease name
        DISEASE_MAPPING = {
            0: "Melanoma",
            1: "Basal Cell Carcinoma",
            2: "Squamous Cell Carcinoma",
            3: "Benign Lesion",
        }

        predicted_disease = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")

        return predicted_disease, confidence
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        print(e)
        return None, None


# Sidebar Menu
st.sidebar.title("🩺 Skin Cancer Detection Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])

# Main Pages
if app_mode == "Home":
    st.title("🌿 Skin Cancer Detection App")
    st.markdown("""
    This app allows:
    - Model training with your own dataset.
    - Testing uploaded images for disease classification using a trained CNN model.
    """)

elif app_mode == "Train & Test Model":
    st.header("🛠 Train & Test Model")
    uploaded_csv = st.file_uploader("Upload the CSV file with `image_id` and `dx` (labels)", type=["csv"])
    uploaded_image_dir = st.text_input("Enter the path to the image directory", "")

    if uploaded_csv and uploaded_image_dir:
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                X_train, X_test, y_train, y_test = load_and_preprocess_data(uploaded_image_dir, uploaded_csv)
                create_and_train_cnn_model(X_train, y_train)

elif app_mode == "Prediction":
    st.header("🔮 Make Predictions")
    uploaded_image = st.file_uploader("Upload a skin image for prediction", type=["jpg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image")
        if st.button("Run Prediction"):
            with st.spinner("Running prediction..."):
                predicted_disease, confidence = run_prediction(uploaded_image)
                if predicted_disease:
                    st.success(f"✅ Disease Predicted: {predicted_disease}")
                    st.success(f"Prediction Confidence: {confidence:.2f}")


elif app_mode == "About":
    st.header("📖 About")
    st.markdown("""
    This web application uses **CNNs** to predict skin cancer risk from dermoscopic image data.
    It allows:
    - Training models with CSV data and associated image directories.
    - Testing custom images uploaded for real-time skin cancer risk prediction.
    Built with **TensorFlow, Streamlit, and Python**.
    """)






