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
    X = X / 255.0  # Normalize pixel values if features are expected image-like

    # Split data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Debugging Information
    st.write("Unique classes:", df['dx'].unique())
    st.write("Number of classes:", num_classes)
    st.write("Encoded y_train shape:", y_train.shape)

    return X_train, X_test, y_train, y_test, label_encoder, num_classes


def create_and_train_model(X_train, y_train, X_test, y_test, num_classes):
    """
    Defines, compiles, and trains a model with dynamic class handling and logs results.
    """
    # Decode class indices properly
    y_train_indices = np.argmax(y_train, axis=1)  # Ensure indices are extracted properly
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_indices),
        y=y_train_indices
    )

    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Debugging Output
    st.write("Class weights computed:", class_weights_dict)

    # Model architecture
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")  # Dynamically match the number of classes
    ])

    # Compile model with appropriate metrics
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with computed class weights
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=16,
        class_weight=class_weights_dict,
        verbose=2
    )

    # Save trained model
    model_save_path = './data/trained_skin_cancer_model.keras'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure directory exists
    model.save(model_save_path)
    st.success(f"✅ Model trained and saved to: {model_save_path}")

    # Evaluate on test data
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"🔍 Test Accuracy: {accuracy:.2%}")

    return model


def preprocess_uploaded_image(image_file):
    """
    Preprocess uploaded image into features suitable for prediction by resizing, normalizing, and reshaping.
    """
    try:
        # Open the image and resize
        image = Image.open(image_file).convert('RGB').resize((128, 128))  # Resize image
        image = np.array(image) / 255.0  # Normalize pixel values
        image = image.reshape(-1)  # Flatten the image into a 1D array

        # Debugging output
        st.write("Image shape for prediction (flattened):", image.shape)

        # Reshape to add batch dimension
        image = np.expand_dims(image, axis=0)  # Make it (1, num_features)

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
    Runs model prediction using preprocessed uploaded image.
    """
    # Extract file name
    image_name = os.path.splitext(image_file.name)[0]

    # Load the trained model
    model_path = './data/trained_skin_cancer_model.keras'
    if not os.path.exists(model_path):
        st.error("❌ Trained model not found. Please train the model first.")
        return None, None

    model = tf.keras.models.load_model(model_path)

    # Preprocess image
    features = preprocess_uploaded_image(image_file)

    if features is not None:
        try:
            # Predict with trained model
            predictions = model.predict(features)
            predicted_idx = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_idx]

            # Map prediction index back to disease name
            disease_name = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")

            # Display results
            st.success(f"✅ Prediction Confidence: {confidence:.2%}")
            st.subheader(f"Predicted Disease: {disease_name}")
            st.info(f"Based on uploaded image: {image_name}")
            return predicted_idx, confidence
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            print(e)
            return None, None
    else:
        st.error("❌ Failed to preprocess the uploaded image.")
        return None, None


# Sidebar Menu
st.sidebar.title("🩺 Skin Cancer Prediction Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])

# Main Pages
if app_mode == "Home":
    st.title("🌿 Skin Cancer Detection App")
    st.markdown("""
    This web app allows you to:
    - Train a model with your own CSV dataset.
    - Predict based on your uploaded image using a trained model.
    - Use real-time prediction features.
    """)

elif app_mode == "Train & Test Model":
    st.header("🛠 Train & Test Model")
    uploaded_file = st.file_uploader("Upload your CSV file for training", type=["csv"])

    if uploaded_file:
        st.info("📊 Dataset loaded. Preparing...")
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())

        if st.button("Train Model"):
            with st.spinner("🔄 Training model..."):
                X_train, X_test, y_train, y_test, label_encoder, num_classes = preprocess_data(df)
                create_and_train_model(X_train, y_train, X_test, y_test, num_classes)

elif app_mode == "Prediction":
    st.header("🔮 Make Predictions")
    uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Run Prediction"):
            with st.spinner("⏳ Running prediction..."):
                run_prediction(uploaded_image)









