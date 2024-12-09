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
    Preprocess data for training. Handles encoding and splits data.
    Normalizes features and extracts target variables correctly.
    """
    # Label encoding target variable dynamically
    label_encoder = LabelEncoder()
    df['dx'] = label_encoder.fit_transform(df['dx'])  # Encode disease types dynamically

    # Handle missing data
    if 'age' in df.columns:
        df['age'].fillna(df['age'].mean(), inplace=True)

    # Prepare features and target
    X = df.drop(columns=['image_id', 'dx_type', 'dx'], errors='ignore')
    y = pd.get_dummies(df['dx']).to_numpy()

    # Handle remaining NaN values
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Normalize features with Rescaling
    scaler = tf.keras.layers.Rescaling(1.0 / 255)
    X = scaler(tf.constant(X)).numpy()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Extract class names dynamically
    class_names = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))

    # Return the splits and class names
    return X_train, X_test, y_train, y_test, class_names, label_encoder


def create_and_train_model(X_train, y_train):
    """
    Defines, compiles, and trains a model.
    If class imbalances exist, adjust by using class weights.
    """
    # Compute class weights if imbalance exists
    class_weights = {}
    class_counts = np.sum(y_train, axis=0)
    for idx, count in enumerate(class_counts):
        if count > 0:
            class_weights[idx] = len(y_train) / (len(class_counts) * count)

    # Define the model architecture
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(y_train.shape[1], activation="softmax")
    ])

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=16, verbose=2, class_weight=class_weights)

    # Save the model
    model.save('trained_skin_cancer_model.keras')
    st.success("✅ Model trained and saved successfully!")
    return model


def preprocess_uploaded_image(image_file):
    """
    Preprocesses the image for prediction by normalizing it as expected during model training.
    """
    try:
        # Open and preprocess the image
        image = Image.open(image_file).convert('RGB').resize((128, 128))  # Resize to model's expected input size
        image = np.array(image) / 255.0  # Normalize to range [0,1]
        image = image.flatten()  # Flattening, depending on model's input expectations
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        print(e)
        return None


def run_prediction(image_file, class_names):
    """
    Run prediction on an uploaded image with dynamic class name mapping.
    """
    model = tf.keras.models.load_model('trained_skin_cancer_model.keras')
    features = preprocess_uploaded_image(image_file)

    if features is not None:
        try:
            # Perform prediction
            predictions = model.predict(features)
            predicted_idx = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_idx]

            # Map index back to class names
            predicted_disease = class_names[predicted_idx]

            return predicted_disease, confidence
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            print(e)
            return None, None
    else:
        st.error("Unable to preprocess image for prediction.")
        return None, None


# Sidebar Menu
st.sidebar.title("🩺 Skin Cancer Prediction Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])


# Main Pages
if app_mode == "Home":
    st.title("🌿 Skin Cancer Detection App")
    st.markdown("""
    This app lets you:
    - Train a model using your own dataset.
    - Predict skin cancer risk using a test image.
    """)

elif app_mode == "Train & Test Model":
    st.header("🛠 Train & Test Model")
    uploaded_file = st.file_uploader("Upload a CSV file with disease data", type=["csv"])

    if uploaded_file:
        st.info("Dataset ready for training...")
        df = pd.read_csv(uploaded_file)
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                X_train, X_test, y_train, y_test, class_names, _ = preprocess_data(df)
                create_and_train_model(X_train, y_train)

elif app_mode == "Prediction":
    st.header("🔮 Run Prediction")
    uploaded_image = st.file_uploader("Upload image for testing", type=["jpg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image")
        if st.button("Run Prediction"):
            with st.spinner("Running predictions..."):
                class_names = pd.read_csv('trained_skin_cancer_model.classes.csv').columns  # dynamically load mapping
                disease_name, confidence = run_prediction(uploaded_image, class_names)
                if disease_name:
                    st.success(f"✅ Prediction Confidence: {confidence:.2f}")
                    st.subheader(f"Predicted Disease: {disease_name}")








