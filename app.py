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
import tensorflow.keras.backend as K


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
    Defines, compiles, and trains a CNN model for classification with proper class handling.
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

    # Define the CNN model architecture
    num_classes = len(class_weights_dict)
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(128, 128, 3)),  # Input dimensions for images
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")  # Dynamically match the number of classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with class weights
    history = model.fit(
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
    
    # Return the model for predictions
    return model


def preprocess_uploaded_image(image_file, add_noise=True):
    """
    Preprocess the uploaded image into numerical features expected by the model.
    Introduces intentional variance by adding slight random noise for variability in prediction.
    """
    try:
        # Open the image and resize
        image = Image.open(image_file).convert('RGB').resize((128, 128))
        image = np.array(image) / 255.0  # Normalize pixel values

        if add_noise:
            # Add variance/noise for stochastic behavior
            noise = np.random.normal(0, 0.01, image.shape)  # Gaussian noise
            image += noise

        # Ensure the image is reshaped properly for CNNs
        image = np.expand_dims(image, axis=0)  # Shape becomes (1, 128, 128, 3)

        st.write("Processed image shape ready for prediction:", image.shape)

        return image
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
    Preprocesses the image and makes a stochastic prediction with intentional variance.
    """
    # Extract file name without extension
    image_name = os.path.splitext(image_file.name)[0]

    # Load the trained model
    model = tf.keras.models.load_model('./data/trained_skin_cancer_model.keras')

    # Preprocess uploaded image into a format expected by the model
    features = preprocess_uploaded_image(image_file)

    if features is not None:
        try:
            # Predict with the model
            predictions = model.predict(features)

            # Map index to disease
            predicted_idx = np.argmax(predictions, axis=-1)[0]
            confidence = predictions[0][predicted_idx]

            # Map index to disease name
            disease_name = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")

            # Render prediction results
            st.success(f"✅ Prediction Confidence: {confidence:.2%}")
            st.subheader(f"Predicted Disease: {disease_name}")
            st.info(f"Based on uploaded image: {image_name}")

            return predicted_idx, confidence
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            print(e)
    else:
        st.error("Failed to preprocess uploaded image.")
    return None, None


# Streamlit UI
st.title("Skin Cancer Diagnosis Prediction")
st.write("""
This application uses a trained neural network to classify uploaded skin images
into potential disease categories. Upload a skin lesion image to begin.
""")

uploaded_image = st.file_uploader("Upload an image of a skin lesion", type=["jpg", "jpeg", "png"])

if uploaded_image:
    if st.button("Run Prediction"):
        run_prediction(uploaded_image)
