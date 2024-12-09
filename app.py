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


def create_and_train_model(X_train, y_train, X_test, y_test, num_classes):
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

    # Save the model
    model.save('trained_skin_cancer_model.keras')
    st.success("✅ Model trained and saved successfully!")

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


def run_prediction(image_file):
    """
    Run prediction on an uploaded image after preprocessing it into expected numerical features.
    """
    # Load the trained model
    model = tf.keras.models.load_model('trained_skin_cancer_model.keras')

    # Preprocess the uploaded image into features expected by the model
    features = preprocess_uploaded_image(image_file)

    if features is not None:
        try:
            # Predict using the features
            predictions = model.predict(features)
            predicted_idx = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_idx]

            # Map prediction index back to a disease name
            disease_name = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")

            st.success(f"✅ Prediction Confidence: {confidence:.2%}")
            st.subheader(f"Predicted Disease: {disease_name}")
            return predicted_idx, confidence
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            print(e)
            return None, None
    else:
        st.error("Failed to process the uploaded image.")
        return None, None


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
                create_and_train_model(X_train, y_train, X_test, y_test, num_classes)


