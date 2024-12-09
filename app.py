import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from PIL import Image
import os


# Helper Functions
def load_and_preprocess_data(image_dir, csv_file):
    """
    Preprocess the dataset based on images in a directory and their labels in the CSV.
    Assumes CSV contains paths and labels.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None, None, None, None, None

    images = []
    labels = []

    # Map disease categories
    DISEASE_MAPPING = {
        "Melanoma": 0,
        "Basal Cell Carcinoma": 1,
        "Squamous Cell Carcinoma": 2,
        "Benign Lesion": 3,
    }

    for index, row in df.iterrows():
        try:
            img_path = os.path.join(image_dir, row["image_id"])
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB').resize((128, 128))
                img = np.array(img) / 255.0
                images.append(img)
                if row["dx"] in DISEASE_MAPPING:
                    labels.append(DISEASE_MAPPING[row["dx"]])
                else:
                    st.warning(f"Unrecognized label {row['dx']} for image {row['image_id']}")
            else:
                st.warning(f"Image file {img_path} does not exist.")
        except Exception as e:
            st.error(f"Skipping image {row['image_id']} due to error: {e}")

    # Check if images and labels are parsed correctly
    if len(images) == 0 or len(labels) == 0:
        st.error("No images or labels were processed. Check your CSV and directory paths.")
        return None, None, None, None, None

    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Ensure y is one-hot encoded
    y = tf.keras.utils.to_categorical(y, num_classes=len(DISEASE_MAPPING))

    # Split the data into train and test sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except ValueError as e:
        st.error(f"Data splitting issue: {e}")
        st.info(f"Images shape: {X.shape}, Labels shape: {y.shape}")
        return None, None, None, None, None

    return X_train, X_test, y_train, y_test, DISEASE_MAPPING


def create_and_train_cnn_model(X_train, y_train):
    """
    Create and train a CNN model for classification. Save the model for later use.
    """
    try:
        model = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(y_train.shape[1], activation="softmax")
        ])

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # Train model
        model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=32, verbose=2)

        # Save the model
        model_save_path = os.path.join(os.getcwd(), 'trained_skin_cancer_cnn_model.h5')
        model.save(model_save_path)
        st.success(f"✅ Model trained and saved successfully at {model_save_path}")
        return model_save_path
    except Exception as e:
        st.error(f"Model training failed: {e}")
        print(e)
        return None


def preprocess_uploaded_image(image_file):
    """
    Preprocess the uploaded image into input expected by CNN model.
    """
    try:
        image = Image.open(image_file).convert('RGB').resize((128, 128))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)  # Make it batch-ready
        return image
    except Exception as e:
        st.error(f"Error preprocessing the image: {e}")
        return None


def run_prediction(image_file, model_path):
    """
    Perform a prediction on the uploaded image using the trained CNN model.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        img = preprocess_uploaded_image(image_file)

        if img is None:
            return None, None

        predictions = model.predict(img)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]

        DISEASE_MAPPING = {
            0: "Melanoma",
            1: "Basal Cell Carcinoma",
            2: "Squamous Cell Carcinoma",
            3: "Benign Lesion",
        }

        predicted_disease = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")
        return predicted_disease, confidence
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None


# Sidebar Menu
st.sidebar.title("🩺 Skin Cancer Detection Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])

if app_mode == "Train & Test Model":
    st.header("🛠 Train & Test Model")
    uploaded_csv = st.file_uploader("Upload CSV file with image IDs & labels (dx)", type=["csv"])
    uploaded_image_dir = st.file_uploader("Upload directory (ZIP or Folder Link if applicable)", type=["zip"])

    if uploaded_csv:
        if st.button("Train Model"):
            with st.spinner("Training the model..."):
                X_train, X_test, y_train, y_test, disease_mapping = load_and_preprocess_data(
                    uploaded_image_dir, uploaded_csv
                )
                if X_train is not None and y_train is not None:
                    create_and_train_cnn_model(X_train, y_train)

elif app_mode == "Prediction":
    st.header("🔮 Make Predictions")
    uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Run Prediction"):
            with st.spinner("⏳ Running prediction..."):
                model_path = './trained_skin_cancer_cnn_model.h5'
                predicted_disease, confidence = run_prediction(uploaded_image, model_path)
                if predicted_disease:
                    st.success(f"✅ Prediction Confidence: {confidence:.2f}")
                    st.subheader(f"Predicted Disease: {predicted_disease}")








