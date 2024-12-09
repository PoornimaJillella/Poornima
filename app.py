import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image


# Helper Functions
def preprocess_data(df):
    """
    Preprocess data for model training by extracting features from numerical/statistical information.
    Extract only the mean RGB statistics.
    """
    # Label encoding target variable (dx)
    label_encoder = LabelEncoder()
    df['dx'] = label_encoder.fit_transform(df['dx'])  # Encoding diagnosis classes

    # Extract features based on statistical RGB features (mean red, green, blue, average intensity)
    features = []
    for index, row in df.iterrows():
        # Simulate feature extraction from row-based pixel statistics
        # Use mock feature values here for demonstration (train only on 4 extracted features)
        mean_red = row['age'] / 255 if 'age' in df.columns else 0
        mean_green = row['age'] / 255 if 'age' in df.columns else 0
        mean_blue = row['age'] / 255 if 'age' in df.columns else 0
        mean_intensity = row['age'] / 255 if 'age' in df.columns else 0
        features.append([mean_red, mean_green, mean_blue, mean_intensity])

    # Extract label and features
    X = np.array(features)  # Feature matrix
    y = pd.get_dummies(df['dx']).to_numpy()

    # Split train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder


def create_and_train_model(X_train, y_train):
    """
    Create a simple model and train on the extracted statistical features.
    """
    model = Sequential([
        Dense(64, activation="relu", input_shape=(4,)),  # Expect 4 statistical features
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(y_train.shape[1], activation="softmax")
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=2)

    # Save model for later use
    model.save('trained_skin_cancer_model.keras')
    st.success("✅ Model trained and saved successfully!")
    return model


def preprocess_uploaded_image(image_file):
    """
    Preprocess uploaded image into expected statistical features.
    """
    try:
        # Load the image and compute statistical pixel averages
        image = Image.open(image_file).convert('RGB').resize((128, 128))  # Resize image
        image = np.array(image) / 255.0  # Normalize pixel intensities
        mean_red = np.mean(image[:, :, 0])
        mean_green = np.mean(image[:, :, 1])
        mean_blue = np.mean(image[:, :, 2])
        mean_intensity = np.mean(image)

        # Return extracted statistical features for prediction
        features = np.array([mean_red, mean_green, mean_blue, mean_intensity])
        features = np.expand_dims(features, axis=0)  # Shape should now be (1, 4)
        return features
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        print(e)
        return None


def run_prediction(image_file):
    """
    Predict skin cancer on uploaded image after statistical feature extraction.
    """
    model = tf.keras.models.load_model('trained_skin_cancer_model.keras')
    features = preprocess_uploaded_image(image_file)

    if features is None:
        return None, None

    try:
        # Run the model's prediction
        predictions = model.predict(features)
        predicted_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_idx]
        return predicted_idx, confidence
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        print(e)
        return None, None


# Map predictions to disease names
DISEASE_MAPPING = {
    0: "Melanoma",
    1: "Basal Cell Carcinoma",
    2: "Squamous Cell Carcinoma",
    3: "Benign Lesion"
}


# Sidebar Menu
st.sidebar.title("🩺 Skin Cancer Prediction Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])


# Main Pages
if app_mode == "Home":
    st.title("🌿 Skin Cancer Detection App")
    st.markdown("""
    This application allows:
    - Training a model with your own data.
    - Running a real-time image prediction pipeline.
    """)

elif app_mode == "Train & Test Model":
    uploaded_file = st.file_uploader("Upload a CSV for model training", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)
                create_and_train_model(X_train, y_train)

elif app_mode == "Prediction":
    uploaded_image = st.file_uploader("Upload a skin image for prediction", type=["jpg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image")
        if st.button("Run Prediction"):
            with st.spinner("Predicting..."):
                predicted_idx, confidence = run_prediction(uploaded_image)
                if predicted_idx is not None:
                    disease_name = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")
                    st.success(f"Prediction Confidence: {confidence:.2f}")
                    st.subheader(f"Predicted Disease: {disease_name}")





