import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from PIL import Image
import os


# Helper Functions
def load_and_preprocess_data(csv_file, image_dir):
    """
    Load CSV, process image paths and preprocess data.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None, None, None, None, None

    # Map diseases dynamically from CSV
    DISEASE_MAPPING = {
        "Melanoma": 0,
        "Basal Cell Carcinoma": 1,
        "Squamous Cell Carcinoma": 2,
        "Benign Lesion": 3
    }

    images = []
    labels = []
    
    for index, row in df.iterrows():
        try:
            img_path = os.path.join(image_dir, row["image_id"])
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB').resize((224, 224))  # Resize to fit MobileNetV2's expected input
                img = np.array(img) / 255.0  # Normalize
                images.append(img)
                if row["dx"] in DISEASE_MAPPING:
                    labels.append(DISEASE_MAPPING[row["dx"]])
                else:
                    st.warning(f"Skipping unknown disease type {row['dx']}")
            else:
                st.warning(f"Could not find image at path {img_path}")
        except Exception as e:
            st.error(f"Skipping due to processing issue: {e}")

    # Preprocess features
    X = np.array(images)
    y = tf.keras.utils.to_categorical(labels, num_classes=len(DISEASE_MAPPING))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, DISEASE_MAPPING


def create_cnn_model(input_shape, num_classes):
    """
    Create a CNN architecture using Transfer Learning with MobileNetV2
    """
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze pre-trained base model
    for layer in base_model.layers:
        layer.trainable = False

    # Build the new head layers on top of pre-trained features
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation="softmax")(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    st.success("✅ Model created")
    return model


def augment_and_train(model, X_train, y_train, X_test, y_test):
    """
    Train the CNN model with augmentation.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    datagen.fit(X_train)

    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        validation_data=(X_test, y_test),
                        epochs=20,
                        verbose=2)

    st.success("✅ Model successfully trained")
    return history


def preprocess_uploaded_image(image_file):
    """
    Preprocess uploaded image for prediction
    """
    try:
        image = Image.open(image_file).convert('RGB').resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)  # Expand dims for batch
        return image
    except Exception as e:
        st.error(f"Error processing uploaded image: {e}")
        return None


def run_prediction(image_file, model):
    """
    Predict using uploaded image
    """
    try:
        image = preprocess_uploaded_image(image_file)
        if image is None:
            return None, None

        predictions = model.predict(image)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]

        DISEASE_MAPPING = {
            0: "Melanoma",
            1: "Basal Cell Carcinoma",
            2: "Squamous Cell Carcinoma",
            3: "Benign Lesion"
        }

        return DISEASE_MAPPING.get(predicted_idx, "Unknown Disease"), confidence
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None


# Sidebar Menu
st.sidebar.title("🩺 Skin Cancer Detection Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])

# Main Pages
if app_mode == "Train & Test Model":
    st.header("🛠 Train & Test Model")
    uploaded_csv = st.file_uploader("Upload the CSV file with `image_id` & `dx`", type=["csv"])
    uploaded_images = st.file_uploader("Upload all images as a zip (ensure matching IDs)", type=["zip"])

    if uploaded_csv and uploaded_images:
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Extract uploaded images
                import zipfile
                with zipfile.ZipFile(uploaded_images) as zip_ref:
                    zip_ref.extractall("images/")

                # Preprocess and load data
                X_train, X_test, y_train, y_test, disease_map = load_and_preprocess_data(uploaded_csv, "images/")
                if X_train is not None:
                    model = create_cnn_model(input_shape=(224, 224, 3), num_classes=len(disease_map))
                    history = augment_and_train(model, X_train, y_train, X_test, y_test)
                    model.save('trained_skin_cancer_model.h5')
                    st.success("✅ Training Complete & Model Saved")
else:
    if app_mode == "Prediction":
        st.header("🔮 Run Prediction")
        uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "png"])
        if uploaded_image:
            with st.spinner("Running Prediction..."):
                model = tf.keras.models.load_model("trained_skin_cancer_model.h5")
                disease_name, confidence = run_prediction(uploaded_image, model)

                if disease_name:
                    st.success(f"✅ Prediction Confidence: {confidence:.2f}")
                    st.subheader(f"Predicted Disease: {disease_name}")










