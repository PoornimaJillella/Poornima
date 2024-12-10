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

        # Create the model dynamically
        num_classes = len(class_weights_dict)
        model = Sequential([
            Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
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
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure the directory exists
        model.save(model_save_path)
        st.success(f"✅ Model trained and saved to: {model_save_path}")

        # Evaluate model performance
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        st.success(f"🔍 Test Accuracy: {accuracy:.2%}")

        return model
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        return None


def preprocess_uploaded_image(image_file):
    """
    Preprocess image for prediction
    """
    try:
        image = Image.open(image_file).convert('RGB').resize((128, 128))
        image_array = np.array(image) / 255.0  # Normalize the image
        # Extract features from image statistics
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


# Run predictions
def run_prediction(image_file):
    """
    Preprocesses image and predicts using trained model
    """
    try:
        model_path = './data/trained_skin_cancer_model.keras'
        if not os.path.exists(model_path):
            st.error("Model not found. Please train the model first.")
            return

        # Preprocess the image
        features = preprocess_uploaded_image(image_file)
        if features is None:
            st.error("Failed to preprocess the image.")
            return

        model = tf.keras.models.load_model(model_path)

        # Run prediction
        predictions = model.predict(features)
        predicted_idx = np.argmax(predictions, axis=-1)[0]
        confidence = predictions[0][predicted_idx]

        # Map index to disease
        DISEASE_MAPPING = {
            0: "Melanoma",
            1: "Basal Cell Carcinoma",
            2: "Squamous Cell Carcinoma",
            3: "Benign Lesion"
        }

        disease_name = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")
        st.success(f"✅ Prediction Confidence: {confidence:.2%}")
        st.subheader(f"Predicted Disease: {disease_name}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        print(e)


# Main App
st.sidebar.title("🩺 Skin Cancer Vision Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])

if app_mode == "Home":
    st.title("🔬 Welcome to Skin Cancer Vision")
elif app_mode == "Train & Test Model":
    uploaded_file = st.file_uploader("Upload a CSV file for training", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if st.button("Train Model"):
            with st.spinner("Training model..."):
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
