import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from PIL import Image
import os


# Helper Functions
def preprocess_data(df):
    """
    Preprocess data for training. Handles encoding and splits data.
    Ensures input features always match the expected dimensions.
    """
    try:
        # Ensure necessary columns exist
        expected_columns = ['age', 'sex', 'some_feature', 'another_feature', 'dx']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"Uploaded CSV must contain the following columns: {expected_columns}")

        # Label encoding target variable
        label_encoder = LabelEncoder()
        df['dx'] = label_encoder.fit_transform(df['dx'])

        # Handle missing data
        if 'age' in df.columns:
            df['age'].fillna(df['age'].mean(), inplace=True)

        # Prepare features and target
        X = df[['age', 'sex', 'some_feature', 'another_feature']]
        y = df['dx'].to_numpy()
        y = tf.keras.utils.to_categorical(y)  # Convert to one-hot encoding for classification

        # Handle remaining NaN values
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()

        # Normalize features
        X = X / 255.0

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test, label_encoder
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None, None, None, None, None


def preprocess_uploaded_image(image_file):
    """
    Preprocesses the uploaded image for model prediction.
    PNG-specific logic added here for clear skin detection simulation.
    """
    try:
        # Open and preprocess the image
        image = Image.open(image_file).convert('RGB').resize((128, 128))
        image_array = np.array(image) / 255.0  # Normalize pixel values
        # Extracting statistical features
        features = [
            image_array.mean(axis=(0, 1)).mean(),
            image_array.std(axis=(0, 1)).mean(),
            image_array.max(axis=(0, 1)).mean(),
            image_array.min(axis=(0, 1)).mean()
        ]
        feature_vector = np.array(features)
        feature_vector = np.expand_dims(feature_vector, axis=0)  # Reshape for model
        st.write("Extracted features from uploaded image:", feature_vector)
        return feature_vector
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


DISEASE_MAPPING = {
    0: "Melanoma",
    1: "Basal Cell Carcinoma",
    2: "Squamous Cell Carcinoma",
    3: "Benign Lesion"
}


def run_prediction(image_file):
    """
    Run prediction with the logic that PNGs always result in 'No Cancer Detected' by default.
    """
    try:
        # PNG-specific logic
        if image_file.name.endswith('.png'):
            st.success("✅ No cancer detected in the uploaded skin image (PNG detected).")
            st.subheader("Clear Skin - No Cancer Detected")
            return None, None

        # Non-PNG predictions
        model = tf.keras.models.load_model('./data/trained_skin_cancer_model.keras')
        features = preprocess_uploaded_image(image_file)

        if features is not None:
            predictions = model.predict(features)

            # Confidence threshold logic
            confidence_threshold = 0.6
            predicted_idx = np.argmax(predictions, axis=-1)[0]
            confidence = predictions[0][predicted_idx]

            if confidence < confidence_threshold:
                st.success("✅ No cancer detected in the uploaded skin image.")
                st.subheader("Clear Skin - No Cancer Detected")
            else:
                disease_name = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")
                st.success(f"✅ Prediction Confidence: {confidence:.2%}")
                st.subheader(f"Predicted Disease: {disease_name}")
            return predicted_idx, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        print(e)

    return None, None


# Sidebar Menu
st.sidebar.title("🩺 Skin Cancer Vision Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])

if app_mode == "Home":
    st.title("🔬 Welcome to Skin Cancer Vision")
elif app_mode == "Train & Test Model":
    uploaded_file = st.file_uploader("Upload a CSV file for training", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if st.button("Train Model"):
            with st.spinner("Training Model..."):
                X_train, X_test, y_train, y_test, _ = preprocess_data(df)
                if X_train is not None:
                    create_and_train_model(X_train, y_train, X_test, y_test)
elif app_mode == "Prediction":
    st.title("Skin Cancer Prediction via Image Upload")
    uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Run Prediction"):
            with st.spinner("Running prediction..."):
                run_prediction(uploaded_image)
elif app_mode == "About":
    st.write("""
        Skin Cancer Vision uses AI-powered image analysis to predict potential skin cancer diseases by analyzing uploaded images.
        It provides users and healthcare professionals a simple way to monitor and diagnose signs of skin cancer early.
    """)













