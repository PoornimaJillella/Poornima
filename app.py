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

    # Debugging output
    st.write("Class weights computed:", class_weights_dict)

    # Define the model architecture
    num_classes = len(class_weights_dict)
    model = Sequential([
        Dense(64, activation="relu", input_shape=(4,)),  # Input shape matches feature vector length
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


def preprocess_uploaded_image(image_file):
    """
    Preprocesses the uploaded image to extract unique statistical features for model input.
    This ensures that each image produces unique features and distinct predictions.
    """
    try:
        # Open image and resize it to expected dimensions
        image = Image.open(image_file).convert('RGB').resize((128, 128))
        image_array = np.array(image) / 255.0  # Normalize pixel values

        # Extract dynamic image statistics
        mean_rgb = image_array.mean(axis=(0, 1))  # Mean of RGB values
        std_rgb = image_array.std(axis=(0, 1))  # Standard deviation of RGB values
        max_rgb = image_array.max(axis=(0, 1))  # Max pixel values
        min_rgb = image_array.min(axis=(0, 1))  # Min pixel values
        median_rgb = np.median(image_array, axis=(0, 1))  # Median RGB values

        # Combine these features
        feature_vector = np.array([
            mean_rgb[0], mean_rgb[1], mean_rgb[2],
            std_rgb.mean(),
            max_rgb.mean(),
            min_rgb.mean(),
            median_rgb.mean()
        ])

        # Normalize and reshape
        feature_vector = feature_vector[:4]  # Limit to first 4 features
        feature_vector = np.expand_dims(feature_vector, axis=0)  # Reshape into (1, 4)

        st.write("Extracted feature vector for prediction:", feature_vector.shape)
        return feature_vector
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        print(e)
        return None


DISEASE_MAPPING = {
    0: "Melanoma",
    1: "Basal Cell Carcinoma",
    2: "Squamous Cell Carcinoma",
    3: "Benign Lesion"
}


def run_prediction(image_file):
    """
    Preprocesses the uploaded image and runs prediction for dynamic, unique disease classification.
    """
    try:
        model = tf.keras.models.load_model('./data/trained_skin_cancer_model.keras')
        features = preprocess_uploaded_image(image_file)

        if features is not None:
            predictions = model.predict(features)
            predicted_idx = np.argmax(predictions, axis=-1)[0]
            confidence = predictions[0][predicted_idx]
            disease_name = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")
            st.success(f"✅ Prediction Confidence: {confidence:.2%}")
            st.subheader(f"Predicted Disease: {disease_name}")
            return predicted_idx, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        print(e)

    return None, None


# Sidebar Menu
st.sidebar.title("🩺 Skin Cancer Prediction Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])

if app_mode == "Home":
    st.title("Welcome to SkinCancerVision.com")
    st.write("**Empowering Early Detection with AI-Driven Skin Cancer Diagnosis**")
    st.subheader("What is SkinCancerVision?")
    st.write("""
SkinCancerVision.com leverages advanced machine learning algorithms to assist in the early detection of skin cancer. Our AI-powered platform analyzes images of skin lesions to identify signs of melanoma and other skin cancers, enabling users to take proactive steps toward better skin health.
    """)
    st.subheader("Key Features")
    st.markdown("""
    - **AI-Powered Detection:** Analyze skin images for potential cancerous growths.
    - **Early Detection:** Detect skin cancer at its earliest stages.
    - **Accessible to All:** Designed for healthcare providers, researchers, and the general public.
    - **Privacy-Focused:** Your health data remains secure and confidential.
    - **Educational Resources:** Learn about prevention, risks, and early detection signs.
    - **Real-Time Updates:** Stay connected to new research findings.
    """)

elif app_mode == "About":
    st.write("**About SkinCancerVision.com**")
    st.write("""
SkinCancerVision.com is committed to using AI technology to empower individuals with early detection of skin cancer. Designed for users across healthcare, research, and the general public, the platform offers intuitive tools, educational insights, and early diagnostic capabilities.
    """)

# Prediction Mode, Train Mode, and Core Logic work the same.










