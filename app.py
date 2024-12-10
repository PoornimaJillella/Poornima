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


def create_and_train_model(X_train, y_train, X_test, y_test):
    """
    Handles training and saves the trained model.
    """
    try:
        # Handle class imbalance
        y_train_indices = np.argmax(y_train, axis=1)
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_indices),
            y=y_train_indices
        )
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

        st.write("Class weights computed:", class_weights_dict)

        # Create the model dynamically
        num_classes = len(class_weights_dict)
        model = Sequential([
            Dense(64, activation="relu", input_shape=(X_train.shape[1],)),  # Dynamically adjust input shape
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax")
        ])

        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
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

        # Evaluate model performance
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        st.success(f"🔍 Test Accuracy: {accuracy:.2%}")

        return model
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        return None


# Main App UI
st.sidebar.title("🩺 Skin Cancer Vision Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])


if app_mode == "Home":
    # Home Page with Structured Headings
    st.title("Welcome to **Skin Cancer Vision** 🩺")
    st.subheader("Empowering Skin Health with Artificial Intelligence")
    st.markdown("""
    Skin Cancer Vision uses machine learning algorithms to detect potential signs of skin cancer
    by analyzing skin images or uploaded CSV data. This innovative web application provides insights
    into early detection and risk factors associated with various types of skin conditions.
    """)
    st.subheader("🔬 Key Features")
    st.markdown("""
    - **Train & Test Model**: Train AI models with uploaded datasets.
    - **Prediction Dashboard**: Upload and predict diseases based on your image.
    - **Insightful Predictions**: Detect early signs of skin cancer.
    """)

elif app_mode == "Train & Test Model":
    uploaded_file = st.file_uploader("Upload your CSV for training", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if st.button("Train & Test Model"):
            with st.spinner("Training & Testing model..."):
                X_train, X_test, y_train, y_test, _ = preprocess_data(df)
                if X_train is not None:
                    create_and_train_model(X_train, y_train, X_test, y_test)

elif app_mode == "Prediction":
    uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Run Prediction"):
            with st.spinner("Running prediction..."):
                run_prediction(uploaded_image)

elif app_mode == "About":
    # About Page with Structured Content
    st.title("About Skin Cancer Vision 🩺")
    st.subheader("Our Mission")
    st.markdown("""
    Our goal is to provide an AI-powered tool for early detection of skin cancer. Using advanced
    machine learning techniques, **Skin Cancer Vision** provides insights by analyzing images
    and datasets to predict the likelihood of skin conditions.
    """)
    st.subheader("How It Works")
    st.markdown("""
    - **Step 1**: Upload a dataset (CSV) or an image.
    - **Step 2**: Train a machine learning model with real-world data or make predictions.
    - **Step 3**: View the analysis, confidence levels, and recommended actions.
    """)
    st.subheader("Why Early Detection Matters")
    st.markdown("""
    Early detection of skin cancer can save lives. Regular monitoring and analysis of skin conditions
    lead to proactive measures and early treatment.
    """)




