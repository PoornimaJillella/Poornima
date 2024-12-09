import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from PIL import Image


# Helper Functions
def preprocess_images(df):
    """
    Preprocess image data for training. Assumes images can be read from their IDs.
    """
    # Process image IDs and labels into arrays
    images = []
    labels = []
    for index, row in df.iterrows():
        try:
            # Load image
            image_path = row['image_id']
            image = Image.open(image_path).convert('RGB').resize((128, 128))
            image = np.array(image) / 255.0  # Normalize pixel values
            images.append(image)
            # Process the corresponding label
            labels.append(row['dx'])
        except Exception as e:
            st.error(f"Could not load image {row['image_id']}: {e}")

    # Convert images and labels to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def create_and_train_model(images, labels):
    """
    Defines, compiles, and trains a CNN model on image data directly.
    Handles class imbalance by computing class weights.
    Saves the trained model after training.
    """
    # Encode the labels numerically
    unique_labels = np.unique(labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_mapping[label] for label in labels])
    
    # One-hot encode
    y_train = tf.keras.utils.to_categorical(encoded_labels, num_classes=len(unique_labels))

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(images, y_train, test_size=0.2, random_state=42)

    # Build the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=16)

    # Save model
    model.save('trained_skin_cancer_model.keras')
    st.success("✅ Model trained and saved successfully!")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"🔍 Test Accuracy: {accuracy:.2%}")

    return model


# Image Preprocessing Function
def preprocess_uploaded_image(image_file):
    """
    Preprocess the uploaded image into a format suitable for CNN prediction.
    Resize and normalize.
    """
    try:
        # Open and resize the image
        image = Image.open(image_file).convert('RGB').resize((128, 128))
        image = np.array(image) / 255.0  # Normalize values
        image = image.reshape(1, 128, 128, 3)  # Reshape for CNN model input
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


# Sidebar Menu
st.sidebar.title("🩺 Skin Cancer Prediction Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])

DISEASE_MAPPING = {
    0: "Melanoma",
    1: "Basal Cell Carcinoma",
    2: "Squamous Cell Carcinoma",
    3: "Benign Lesion"
}


if app_mode == "Home":
    st.title("🌿 Skin Cancer Detection App")
    st.markdown("""
    This app allows:
    - Model training using image data.
    - Testing predictions using uploaded images.
    - Use a pre-trained model for quick predictions.
    """)

elif app_mode == "Train & Test Model":
    st.header("🛠 Train Model with Images")
    uploaded_file = st.file_uploader("Upload your CSV with image paths for training", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data loaded. Preparing to train model...")
        if st.button("Train Model"):
            with st.spinner("🔄 Processing images & training model..."):
                images, labels = preprocess_images(df)
                create_and_train_model(images, labels)

elif app_mode == "Prediction":
    st.header("🔮 Make Predictions")
    uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "png"])
    if uploaded_image:
        if st.button("Run Prediction"):
            with st.spinner("⏳ Running prediction..."):
                preprocessed_image = preprocess_uploaded_image(uploaded_image)
                model = tf.keras.models.load_model('trained_skin_cancer_model.keras')
                predictions = model.predict(preprocessed_image)
                predicted_idx = np.argmax(predictions, axis=1)[0]
                confidence = predictions[0][predicted_idx]

                st.success(f"✅ Confidence: {confidence:.2f}")
                st.write(f"Predicted Disease: {DISEASE_MAPPING.get(predicted_idx)}")


