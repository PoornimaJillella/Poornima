import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from PIL import Image
import os


# Helper Functions
def preprocess_data(df):
    """
    Preprocess data for training. Handles encoding, splits data, and ensures categorical classes are dynamically set.
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

    # Dynamically determine number of classes
    num_classes = len(df['dx'].unique())
    y = tf.keras.utils.to_categorical(y, num_classes)  # Ensure correct one-hot encoding for classes

    # Handle remaining NaN values and convert X to numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()

    # Normalize features
    X = X / 255.0  # Normalize pixel values

    # Split data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Debugging Information
    st.write("Unique classes:", df['dx'].unique())
    st.write("Number of classes:", num_classes)

    return X_train, X_test, y_train, y_test, num_classes


def create_and_train_cnn(X_train, y_train, X_test, y_test, num_classes):
    """
    Defines, compiles, and trains a CNN model capable of handling image inputs directly.
    """
    # Reshape X_train/X_test into image-like data (128x128x3) for CNNs
    X_train = X_train.reshape(-1, 128, 128, 3)  # Reshape data for CNNs
    X_test = X_test.reshape(-1, 128, 128, 3)

    # Define a CNN model suitable for image classification
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")  # Final classification layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=16,
        verbose=2
    )

    # Save the trained model
    model_save_path = './data/trained_skin_cancer_cnn_model.keras'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure directory exists
    model.save(model_save_path)
    st.success(f"✅ Model trained and saved to: {model_save_path}")

    # Evaluate on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"🔍 Test Accuracy: {accuracy:.2%}")

    return model


def preprocess_uploaded_image(image_file):
    """
    Preprocess uploaded image into the shape expected by CNN for prediction.
    """
    try:
        # Open the image, resize, and normalize
        image = Image.open(image_file).convert('RGB').resize((128, 128))
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Reshape to (1, 128, 128, 3)

        st.write("Image shape prepared for prediction:", image.shape)

        return image
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        print(e)
        return None


def run_prediction(image_file):
    """
    Runs prediction with the trained CNN model on preprocessed images.
    """
    # Path to trained CNN model
    model_path = './data/trained_skin_cancer_cnn_model.keras'
    if not os.path.exists(model_path):
        st.error("❌ Trained model not found. Please train the model first.")
        return None, None

    model = tf.keras.models.load_model(model_path)

    # Preprocess the uploaded image
    features = preprocess_uploaded_image(image_file)

    if features is not None:
        try:
            predictions = model.predict(features)
            predicted_idx = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_idx]

            # Map prediction index back to disease name
            disease_name = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")

            st.success(f"✅ Prediction Confidence: {confidence:.2%}")
            st.subheader(f"Predicted Disease: {disease_name}")
            return predicted_idx, confidence
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            print(e)
    else:
        st.error("❌ Failed to preprocess the uploaded image.")











