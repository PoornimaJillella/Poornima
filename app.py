import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from PIL import Image
import os


# Helper Functions
def load_and_preprocess_data(image_dir, csv_file):
    """
    Preprocess the images and map CSV to image paths and labels
    Ensures path integrity and processes images into arrays
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None, None, None, None, None

    images = []
    labels = []

    # Map diseases dynamically from CSV
    DISEASE_MAPPING = {
        "Melanoma": 0,
        "Basal Cell Carcinoma": 1,
        "Squamous Cell Carcinoma": 2,
        "Benign Lesion": 3
    }

    for index, row in df.iterrows():
        try:
            img_path = os.path.join(image_dir, row["image_id"])
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB').resize((224, 224))  # Resize to work with MobileNetV2
                img = np.array(img) / 255.0  # Normalize
                images.append(img)
                if row["dx"] in DISEASE_MAPPING:
                    labels.append(DISEASE_MAPPING[row["dx"]])
                else:
                    st.warning(f"Unknown disease label for row {row}")
            else:
                st.warning(f"Image not found: {img_path}")
        except Exception as e:
            st.error(f"Skipping {row['image_id']} due to error: {e}")

    # Preprocess the images into numpy arrays
    X = np.array(images)
    y = tf.keras.utils.to_categorical(labels, num_classes=len(DISEASE_MAPPING))  # One-hot encode the classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, DISEASE_MAPPING


def create_cnn_model(input_shape, num_classes):
    """
    Create a CNN architecture using Transfer Learning with MobileNetV2
    """
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze layers for pre-trained feature extraction
    for layer in base_model.layers:
        layer.trainable = False

    # Define the new layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation="softmax")(x)

    # Complete Model
    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    st.success("✅ Model created and ready for training")
    return model


def augment_and_train(model, X_train, y_train, X_test, y_test):
    """
    Use ImageDataGenerator for Data Augmentation and Train the Model
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Fit the generator
    datagen.fit(X_train)

    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        validation_data=(X_test, y_test),
                        epochs=20,
                        verbose=2)

    st.success("✅ Model trained successfully!")
    return history


def preprocess_uploaded_image(image_file):
    """
    Preprocess uploaded image to feed into trained CNN model.
    """
    try:
        image = Image.open(image_file).convert('RGB').resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        st.error(f"Error preprocessing the image: {e}")
        return None


def run_prediction(image_file, model):
    """
    Run prediction on uploaded image
    """
    try:
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
            3: "Benign Lesion"
        }

        return DISEASE_MAPPING.get(predicted_idx, "Unknown"), confidence
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None


# Sidebar Menu
st.sidebar.title("🩺 Skin Cancer Detection Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])

if app_mode == "Train & Test Model":
    st.header("🛠 Train & Test Model")
    uploaded_csv = st.file_uploader("Upload your CSV with image IDs and disease labels (dx)", type=["csv"])
    uploaded_dir = st.text_input("Enter directory path with image files")

    if uploaded_csv and uploaded_dir:
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                X_train, X_test, y_train, y_test, disease_map = load_and_preprocess_data(
                    uploaded_dir, uploaded_csv
                )
                if X_train is not None:
                    model = create_cnn_model(input_shape=(224, 224, 3), num_classes=len(disease_map))
                    history = augment_and_train(model, X_train, y_train, X_test, y_test)
                    st.success("✅ Model Trained Successfully!")

elif app_mode == "Prediction":
    st.header("🔮 Predict with a trained model")
    uploaded_image = st.file_uploader("Upload image for prediction", type=["jpg", "png"])
    if uploaded_image:
        if st.button("Run Prediction"):
            with st.spinner("⏳ Running prediction..."):
                model_path = './trained_skin_cancer_cnn_model.h5'
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path)
                    disease_name, confidence = run_prediction(uploaded_image, model)
                    st.success(f"✅ Disease Prediction: {disease_name} with confidence {confidence:.2f}")









