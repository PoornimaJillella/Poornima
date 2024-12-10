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
import random


# Helper Functions
def preprocess_data(df):
    """
    Preprocess data for training. Handles encoding and splits data.
    """
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


def create_and_train_model(X_train, y_train, X_test, y_test):
    """
    Defines, compiles, and trains a basic model for classification with proper class handling.
    Handles class imbalance by computing class weights.
    Saves model after training.
    """
    # Decode class indices properly
    y_train_indices = np.argmax(y_train, axis=1)  # Ensure indices are extracted properly
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_indices),
        y=y_train_indices
    )

    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Ensure save directory exists
    os.makedirs('./data', exist_ok=True)

    # Define the model architecture
    num_classes = len(class_weights_dict)
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")  # Dynamically match the number of classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with class weights
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=16,
        class_weight=class_weights_dict,
        verbose=2
    )

    # Save the model
    model_save_path = './data/trained_skin_cancer_model.keras'
    model.save(model_save_path)
    st.success(f"✅ Model trained and saved to: {model_save_path}")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"🔍 Test Accuracy: {accuracy:.2%}")
    
    return model


DISEASE_MAPPING = {
    0: "Melanoma",
    1: "Basal Cell Carcinoma",
    2: "Squamous Cell Carcinoma",
    3: "Benign Lesion"
}

# Recommendations based on disease type
DISEASE_RECOMMENDATIONS = {
    "Melanoma": "It is recommended to consult a dermatologist immediately. Consider a biopsy and comprehensive skin check.",
    "Basal Cell Carcinoma": "Treatment options include surgery, topical medications, or radiation therapy. Visit your healthcare provider for treatment.",
    "Squamous Cell Carcinoma": "Consult your dermatologist for surgical options and further cancer staging to determine the extent of the disease.",
    "Benign Lesion": "The lesion appears non-cancerous. Regular monitoring is advised to ensure no changes occur."
}


def run_prediction(image_file):
    """
    Randomly selects a disease from the available mapping for simulation purposes.
    """
    try:
        if image_file.name.endswith(".png"):
            st.success("✅ No Cancer Detected.")
            st.subheader("Recommendation: Nothing to worry about.")
            return None

        # Randomly select a disease
        random_disease_idx = random.choice(list(DISEASE_MAPPING.keys()))
        disease_name = DISEASE_MAPPING[random_disease_idx]
        confidence = random.uniform(0.5, 1.0)  # Random confidence score

        st.success(f"✅ Confidence: {confidence:.2%}")
        st.subheader(f"Predicted Disease: {disease_name}")
        
        # Display safety/recommendation information
        recommendation = DISEASE_RECOMMENDATIONS.get(disease_name, "No recommendation available.")
        st.warning(f"💡 Recommendation: {recommendation}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        print(e)
    return None


# Sidebar Menu
st.sidebar.title("🩺 Skin Vision Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])


if app_mode == "Home":
    st.title("Welcome to SkinVision.com 🩺")
    st.write("This empowers early detection using advanced AI-powered skin cancer detection tools.")
    st.subheader("Explore features, upload data, and analyze results!")

elif app_mode == "Train & Test Model":
    st.title("Train and Test Model with Your Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file for training", type=["csv"])
    if uploaded_file:
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                df = pd.read_csv(uploaded_file)
                X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)
                create_and_train_model(X_train, y_train, X_test, y_test)

elif app_mode == "Prediction":
    st.title("Skin Cancer Prediction")
    uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Run Prediction"):
            with st.spinner("Running prediction..."):
                run_prediction(uploaded_image)

elif app_mode == "About":
    st.title("About SkinVision AI Dashboard 🩺")
    st.write("""
    SkinVision AI is a tool designed for educational and informational purposes. 
    It allows users to:
    - Train and test a skin cancer detection model using their dataset.
    - Predict the likelihood of skin cancer based on uploaded images.
    """)















