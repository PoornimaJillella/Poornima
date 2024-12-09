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
    Preprocess data for training. Handles encoding and splits data.
    Dynamically determines the number of classes for classification.
    """
    # Label encoding target variable
    label_encoder = LabelEncoder()
    df['dx'] = label_encoder.fit_transform(df['dx'])

    # Handle missing data
    if 'age' in df.columns:
        df['age'].fillna(df['age'].mean(), inplace=True)

    # Prepare features and target
    X = df.drop(columns=['image_id', 'dx_type', 'dx'], errors='ignore')
    y = df['dx']  # Get raw target values
    num_classes = len(df['dx'].unique())  # Dynamically determine number of classes
    y = pd.get_dummies(y).to_numpy()  # One-hot encode using pandas get_dummies

    # Handle remaining NaN values
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Normalize features
    scaler = tf.keras.layers.Rescaling(1.0 / 255)
    X = scaler(tf.constant(X)).numpy()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Debugging output
    st.write("Number of classes determined:", num_classes)
    st.write("Shape of X_train:", X_train.shape)
    st.write("Shape of y_train:", y_train.shape)

    return X_train, X_test, y_train, y_test, num_classes, label_encoder


def create_and_train_model(X_train, y_train, X_test, y_test, num_classes, model_name):
    """
    Defines, compiles, and trains a basic model for classification with proper class handling.
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
    st.write("y_train shape:", y_train.shape)

    # Define the model architecture dynamically
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")  # Adjust number of output neurons dynamically
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with class weights
    try:
        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=10,
            batch_size=16,
            class_weight=class_weights_dict,
            verbose=2
        )
    except Exception as e:
        st.error(f"Error during training: {e}")
        print(e)
        return None

    # Save the model with a dynamic name
    model.save(model_name)
    st.success(f"✅ Model trained and saved as '{model_name}' successfully!")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"🔍 Test Accuracy: {accuracy:.2%}")
    
    return model


def preprocess_uploaded_image(image_file):
    """
    Preprocess the uploaded image into numerical features expected by the model.
    This function computes the mean of R, G, B values and a general mean pixel intensity.
    """
    try:
        # Open the image and resize
        image = Image.open(image_file).convert('RGB').resize((128, 128))  # Resize to expected input dimensions
        image = np.array(image) / 255.0  # Normalize pixel values

        # Extract features
        mean_red = np.mean(image[:, :, 0])
        mean_green = np.mean(image[:, :, 1])
        mean_blue = np.mean(image[:, :, 2])
        mean_intensity = np.mean(image)

        # Create feature array with 4 numerical values
        image_features = np.array([mean_red, mean_green, mean_blue, mean_intensity])
        image_features = np.expand_dims(image_features, axis=0)  # Reshape for prediction

        st.write("Extracted features from image:", image_features)

        return image_features
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


def run_prediction(model_name, image_file):
    """
    Run prediction on an uploaded image using the dynamically saved model's filename.
    """
    # Load the trained model dynamically
    try:
        model = tf.keras.models.load_model(model_name)

        # Preprocess the uploaded image into features expected by the model
        features = preprocess_uploaded_image(image_file)

        if features is not None:
            predictions = model.predict(features)
            predicted_idx = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_idx]

            disease_name = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")
            st.success(f"✅ Prediction Confidence: {confidence:.2%}")
            st.subheader(f"Predicted Disease: {disease_name}")
        else:
            st.error("Failed to process the uploaded image.")
    except Exception as e:
        st.error(f"Error loading or predicting with the model: {e}")
        print(e)


# Sidebar Menu
st.sidebar.title("🩺 Skin Cancer Prediction Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])

# Main Pages
if app_mode == "Train & Test Model":
    st.header("🛠 Train & Test Model")
    uploaded_file = st.file_uploader("Upload your CSV file for training", type=["csv"])

    if uploaded_file:
        st.info("📊 Dataset loaded successfully. Preparing for training...")
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())

        if st.button("Train Model"):
            with st.spinner("🔄 Training model..."):
                X_train, X_test, y_train, y_test, num_classes, label_encoder = preprocess_data(df)
                model_name = f"model_{uploaded_file.name.split('.')[0]}.keras"  # Save with the uploaded filename
                create_and_train_model(X_train, y_train, X_test, y_test, num_classes, model_name)

elif app_mode == "Prediction":
    st.header("🔮 Make Predictions")
    uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "png"])
    uploaded_model_file = st.text_input("Enter Model Filename", value="")  # Allow user to input model name

    if uploaded_image and uploaded_model_file:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Run Prediction"):
            with st.spinner("⏳ Running prediction..."):
                run_prediction(uploaded_model_file, uploaded_image)




