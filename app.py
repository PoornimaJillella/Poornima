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
    Preprocess the data for model training by encoding labels and preparing features.
    Extract pixel statistics from images directly if needed.
    """
    # Encode the diagnosis classes (dx) into integers
    label_encoder = LabelEncoder()
    df['dx'] = label_encoder.fit_transform(df['dx'])  # Encoding diagnosis classes

    # Handle missing values
    if 'age' in df.columns:
        df['age'].fillna(df['age'].mean(), inplace=True)

    # Extract features
    # Prepare only numerical features from statistics like mean RGB intensities
    # Feature extraction based on statistics (4 features: red mean, green mean, blue mean, and mean intensity)
    X = df.drop(columns=['image_id', 'dx_type', 'dx'], errors='ignore')
    y = pd.get_dummies(df['dx']).to_numpy()

    # Handle NaN values (convert columns to numbers and normalize)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Normalize features by Rescaling (map to 0-1 range)
    scaler = tf.keras.layers.Rescaling(1.0 / 255)
    X = scaler(tf.constant(X)).numpy()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder


def create_and_train_model(X_train, y_train):
    """
    Create a simple model and train on processed data.
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(y_train.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=32, verbose=2)

    # Save the model
    model.save('trained_skin_cancer_model.keras')
    st.success("✅ Model trained and saved successfully!")
    return model


def preprocess_uploaded_image(image_file):
    """
    Preprocess uploaded image into expected statistical features (mean red, green, blue values).
    Ensures the input matches the trained model's expectations.
    """
    try:
        image = Image.open(image_file).convert('RGB').resize((128, 128))  # Normalize image
        image = np.array(image) / 255.0  # Normalize pixel values between 0 and 1
        mean_red = np.mean(image[:, :, 0])
        mean_green = np.mean(image[:, :, 1])
        mean_blue = np.mean(image[:, :, 2])
        mean_intensity = np.mean(image)

        # Package as a feature vector expected by model
        features = np.array([mean_red, mean_green, mean_blue, mean_intensity])  # Model expects these 4 features
        features = np.expand_dims(features, axis=0)  # Make shape (1, 4)
        return features
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        print(e)
        return None


def run_prediction(image_file):
    """
    Perform prediction with uploaded image.
    """
    # Load trained model
    model = tf.keras.models.load_model('trained_skin_cancer_model.keras')

    # Preprocess image
    features = preprocess_uploaded_image(image_file)
    if features is None:
        return None, None

    try:
        # Run prediction
        predictions = model.predict(features)
        predicted_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_idx]
        return predicted_idx, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
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
    st.title("🌿 Skin Cancer Detection Application")
    st.markdown("""
    This application allows:
    - Model training using custom data.
    - Real-time predictions on uploaded images.
    - Predictions from trained models.
    """)

elif app_mode == "Train & Test Model":
    uploaded_file = st.file_uploader("Upload a CSV file to train the model", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:", df.head())
        if st.button("Train Model"):
            with st.spinner("Training..."):
                X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)
                create_and_train_model(X_train, y_train)

elif app_mode == "Prediction":
    uploaded_image = st.file_uploader("Upload image for prediction", type=["jpg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image")
        if st.button("Run Prediction"):
            with st.spinner("Predicting..."):
                predicted_idx, confidence = run_prediction(uploaded_image)
                if predicted_idx is not None:
                    disease_name = DISEASE_MAPPING.get(predicted_idx, "Unknown")
                    st.success(f"Prediction Confidence: {confidence:.2f}")
                    st.subheader(f"Predicted Disease: {disease_name}")




