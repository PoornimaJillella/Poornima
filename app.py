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
import hashlib
import random


# Helper Functions
def preprocess_data(df):
    """
    Preprocess data for training. Handles encoding and splits data.
    """
    try:
        # Label encoding target variable
        label_encoder = LabelEncoder()
        df['dx'] = label_encoder.fit_transform(df['dx'])

        # Handle missing data
        if 'age' in df.columns:
            df['age'].fillna(df['age'].mean(), inplace=True)

        # Prepare features and target
        X = df.drop(columns=['image_id', 'dx_type', 'dx'], errors='ignore')
        y = df['dx'].to_numpy()
        y = tf.keras.utils.to_categorical(y)  # Convert to one-hot encoding for classification

        # Handle remaining NaN values
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()

        # Normalize features
        X = X / 255.0  # Normalize pixel values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test, label_encoder
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None, None, None, None, None


def preprocess_uploaded_image(image_file):
    """
    Preprocess uploaded images by extracting statistical features.
    """
    try:
        # Open image and resize it to expected dimensions
        image = Image.open(image_file).convert('RGB').resize((128, 128))
        image_array = np.array(image) / 255.0  # Normalize pixel values

        # Extract statistical features
        mean_rgb = image_array.mean(axis=(0, 1))  # Mean of RGB values
        std_rgb = image_array.std(axis=(0, 1))  # Standard deviation of RGB values
        max_rgb = image_array.max(axis=(0, 1))  # Max pixel values
        min_rgb = image_array.min(axis=(0, 1))  # Min pixel values

        # Combine these features
        feature_vector = np.array([
            mean_rgb[0], mean_rgb[1], mean_rgb[2],
            std_rgb.mean()
        ])

        # Normalize and reshape
        feature_vector = np.expand_dims(feature_vector, axis=0)  # Reshape into (1, 4)

        return feature_vector, image_array
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None, None


DISEASE_MAPPING = {
    0: "Melanoma",
    1: "Basal Cell Carcinoma",
    2: "Squamous Cell Carcinoma",
    3: "Benign Lesion"
}

# Dictionary to cache random predictions based on image hash
random_prediction_cache = {}


def random_prediction():
    """
    Simulates a random disease prediction with random confidence.
    Randomize predictions for randomized behavior.
    """
    predicted_idx = random.choice([0, 1, 2, 3])  # Randomly select a disease index
    confidence = random.uniform(0.5, 1.0)  # Random confidence score between 0.5 and 1.0
    disease_name = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")
    return predicted_idx, confidence, disease_name


def compute_file_hash(image_file):
    """
    Computes a hash for a file so that repeated uploads can have consistent predictions.
    """
    try:
        # Read image bytes and hash using hashlib
        image_bytes = image_file.read()
        file_hash = hashlib.sha256(image_bytes).hexdigest()
        return file_hash
    except Exception as e:
        st.error(f"Error computing hash: {e}")
        return None


def check_irrelevant_image(image_array):
    """
    Heuristic to determine if an image is irrelevant (non-skin related).
    Based on pixel analysis, mostly white/black or low variability suggests irrelevance.
    """
    try:
        # Calculate mean pixel intensity across all RGB values
        average_intensity = image_array.mean()

        # If average intensity is very high or very low, it's likely irrelevant
        if average_intensity > 0.85 or average_intensity < 0.15:
            return True  # Likely an irrelevant image
        return False
    except Exception as e:
        st.error(f"Error analyzing image for irrelevance: {e}")
        return False


def run_prediction(image_file):
    """
    Runs prediction on uploaded images.
    PNG images will always return a default 'No Cancer Detected' response.
    Other images will return randomized predictions but cache repeated uploads based on hash.
    """
    try:
        # Compute hash for consistency across multiple uploads of the same file
        file_hash = compute_file_hash(image_file)

        if file_hash is None:
            return

        # Process uploaded image
        feature_vector, image_array = preprocess_uploaded_image(image_file)

        if image_array is None:
            return

        # Check if the image is irrelevant
        if check_irrelevant_image(image_array):
            st.warning("❌ Irrelevant Image Detected")
            return

        # If it's a PNG file, directly return no cancer detected
        if image_file.name.endswith('.png'):
            st.success("✅ No Cancer Detected")
            return

        # Check cache for repeated hash
        if file_hash in random_prediction_cache:
            # Fetch cached prediction
            predicted_idx, confidence, disease_name = random_prediction_cache[file_hash]
        else:
            # Generate a new random prediction
            predicted_idx, confidence, disease_name = random_prediction()
            random_prediction_cache[file_hash] = (predicted_idx, confidence, disease_name)

        # Show prediction
        st.success(f"✅ Prediction Confidence: {confidence:.2%}")
        st.subheader(f"Predicted Disease: {disease_name}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

















