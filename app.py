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
    Preprocess the data for training: encoding, handling missing values, normalization.
    """
    # Label encoding the diagnosis type
    label_encoder = LabelEncoder()
    df['dx'] = label_encoder.fit_transform(df['dx'])

    # Handle missing data (fill NaN values)
    if 'age' in df.columns:
        df['age'].fillna(df['age'].mean(), inplace=True)

    # Prepare features and target
    X = df.drop(columns=['image_id', 'dx_type', 'dx'], errors='ignore')
    y = pd.get_dummies(df['dx']).to_numpy()

    # Handle NaN values
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Normalize features using tf.keras Rescaling
    scaler = tf.keras.layers.Rescaling(1.0 / 255)
    X = scaler(tf.constant(X)).numpy()

    # Split the data into train/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder


def create_and_train_model(X_train, y_train):
    """
    Define, compile, and train the classification model.
    """
    # Create a neural network model
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(y_train.shape[1], activation="softmax")  # Softmax layer for multi-class classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=32, verbose=2)

    # Save the trained model
    model.save('trained_skin_cancer_model.keras')
    st.success("✅ Model trained and saved successfully!")
    return model


def preprocess_uploaded_image(image_file):
    """
    Preprocess the uploaded image into features expected by the model.
    Normalize pixel intensities and reshape into feature vectors.
    """
    try:
        # Open and preprocess image
        image = Image.open(image_file).convert('RGB').resize((128, 128))  # Resize to 128x128
        image = np.array(image) / 255.0  # Normalize pixel values

        # Flatten pixel intensities into a vector
        image_features = image.reshape(-1)  # Flatten 128x128 image to 1D feature vector
        image_features = np.expand_dims(image_features, axis=0)  # Add batch dimension for prediction
        return image_features
    except Exception as e:
        st.error(f"Error preprocessing the image: {e}")
        print(e)
        return None


def run_prediction(image_file):
    """
    Run prediction using the uploaded image.
    """
    # Load trained model
    model = tf.keras.models.load_model('trained_skin_cancer_model.keras')

    # Preprocess the uploaded image
    features = preprocess_uploaded_image(image_file)
    if features is None:
        return None, None

    # Run prediction
    try:
        predictions = model.predict(features)
        predicted_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_idx]
        return predicted_idx, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        print(e)
        return None, None


# Disease mapping dictionary for human-readable labels
DISEASE_MAPPING = {
    0: "Melanoma",
    1: "Basal Cell Carcinoma",
    2: "Squamous Cell Carcinoma",
    3: "Benign Lesion"
}


# Sidebar menu
st.sidebar.title("🩺 Skin Cancer Prediction Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])


# Main Pages
if app_mode == "Home":
    st.title("🌿 Skin Cancer Detection App")
    st.markdown("""
    This application allows:
    - Model training with your own dataset.
    - Real-time image classification predictions.
    - Instant predictions with trained models.
    """)

elif app_mode == "Train & Test Model":
    st.header("🛠 Train & Test Model")
    uploaded_file = st.file_uploader("Upload your CSV file for model training", type=["csv"])

    if uploaded_file:
        st.info("📊 Dataset loaded. Ready for processing...")
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())

        if st.button("Train Model"):
            with st.spinner("🔄 Training model..."):
                X_train, X_test, y_train, y_test, _ = preprocess_data(df)
                create_and_train_model(X_train, y_train)

elif app_mode == "Prediction":
    st.header("🔮 Image Prediction")
    uploaded_image = st.file_uploader("Upload an image to predict skin cancer", type=["jpg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image")
        if st.button("Run Prediction"):
            with st.spinner("⏳ Predicting..."):
                predicted_idx, confidence = run_prediction(uploaded_image)
                if predicted_idx is not None:
                    disease_name = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")
                    st.success(f"✅ Confidence: {confidence:.2f}")
                    st.subheader(f"Predicted Disease: {disease_name}")

elif app_mode == "About":
    st.header("📖 About")
    st.markdown("""
    This app is designed for educational purposes, leveraging machine learning models to detect skin cancer risk.
    Built with **Streamlit**, **TensorFlow**, and **Python**, it offers:
    - Model training capabilities.
    - Real-time image prediction based on uploaded images.
    """)


