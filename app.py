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


# Helper Functions
def extract_features_from_image(image_path):
    """
    Extract features from an image for training/inference.
    Only uses mean pixel values in R, G, B channels and their combined statistics.
    """
    try:
        image = Image.open(image_path).convert('RGB').resize((128, 128))
        image = np.array(image) / 255.0  # Normalize pixel values to 0-1
        
        # Calculate statistics
        mean_red = np.mean(image[:, :, 0])
        mean_green = np.mean(image[:, :, 1])
        mean_blue = np.mean(image[:, :, 2])
        mean_intensity = np.mean(image)

        return [mean_red, mean_green, mean_blue, mean_intensity]
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def preprocess_data(df):
    """
    Preprocess uploaded CSV data for model training.
    Extract features and split data.
    """
    # Label encoding the target variable
    label_encoder = LabelEncoder()
    df['dx'] = label_encoder.fit_transform(df['dx'])

    # Preprocess image paths to extract features
    features = []
    for index, row in df.iterrows():
        feature_set = extract_features_from_image(row['image_path'])  # Extract image statistics
        if feature_set:
            features.append(feature_set)
        else:
            st.error(f"Could not process image at index {index}: {row['image_path']}")

    X = np.array(features)
    y = pd.get_dummies(df['dx']).to_numpy()

    # Handle missing values
    X = np.nan_to_num(X)  # Replace NaNs with zeros

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder


def create_and_train_model(X_train, y_train, X_test, y_test):
    """
    Define, train, and save a Keras model.
    Uses extracted image features for prediction.
    """
    # Class imbalance handling
    y_train_indices = np.argmax(y_train, axis=1)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_indices),
        y=y_train_indices
    )
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Define the model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(4,)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(y_train.shape[1], activation="softmax")
    ])

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

    # Save the trained model
    model.save('trained_skin_cancer_model.keras')
    st.success("✅ Model trained and saved successfully!")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"🔍 Test Accuracy: {accuracy:.2%}")

    return model


def preprocess_uploaded_image(image_file):
    """
    Extract features from an uploaded image for prediction.
    """
    features = extract_features_from_image(image_file)
    if features:
        features = np.array(features).reshape(1, -1)  # Reshape to match model input
        st.write("Features extracted for inference:", features)
        return features
    else:
        st.error("Failed to process the image.")
        return None


def run_prediction(image_file):
    """
    Load model and perform predictions on the uploaded image.
    """
    model = tf.keras.models.load_model('trained_skin_cancer_model.keras')
    features = preprocess_uploaded_image(image_file)

    if features is not None:
        predictions = model.predict(features)
        predicted_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_idx]

        st.write("Predicted class:", predicted_idx)
        st.write("Prediction confidence:", confidence)

        return predicted_idx, confidence
    else:
        return None, None


# Streamlit App Workflow
st.sidebar.title("🩺 Skin Cancer Prediction Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])

DISEASE_MAPPING = {
    0: "Melanoma",
    1: "Basal Cell Carcinoma",
    2: "Squamous Cell Carcinoma",
    3: "Benign Lesion"
}

if app_mode == "Train & Test Model":
    uploaded_file = st.file_uploader("Upload your CSV with image paths for training", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data loaded:", df.head())
        if st.button("Train Model"):
            with st.spinner("Training..."):
                X_train, X_test, y_train, y_test, le = preprocess_data(df)
                create_and_train_model(X_train, y_train, X_test, y_test)

elif app_mode == "Prediction":
    uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image")
        if st.button("Run Prediction"):
            with st.spinner("Running..."):
                idx, conf = run_prediction(uploaded_image)
                if idx is not None:
                    st.success(f"Predicted Class: {DISEASE_MAPPING[idx]}")
                    st.success(f"Confidence: {conf:.2%}")



