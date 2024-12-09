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
    """
    # Label encoding target variable
    label_encoder = LabelEncoder()
    df['dx'] = label_encoder.fit_transform(df['dx'])

    # Handle missing data
    if 'age' in df.columns:
        df['age'].fillna(df['age'].mean(), inplace=True)

    # Prepare features and target
    X = df.drop(columns=['image_id', 'dx_type', 'dx'], errors='ignore')
    y = pd.get_dummies(df['dx']).to_numpy()

    # Handle remaining NaN values
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Normalize features
    scaler = tf.keras.layers.Rescaling(1.0 / 255)
    X = scaler(tf.constant(X)).numpy()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder


def create_and_train_model(X_train, y_train):
    """
    Defines, compiles, and trains a basic model for classification.
    Saves model after training.
    """
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dense(y_train.shape[1], activation="softmax")
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=16, verbose=2)

    # Save the model
    model.save('trained_skin_cancer_model.keras')
    st.success("Model trained and saved successfully!")
    return model


def run_prediction(image_file):
    """
    Run prediction on an uploaded image.
    """
    # Load the trained model
    model = tf.keras.models.load_model('trained_skin_cancer_model.keras')

    # Process the uploaded image
    try:
        # Resize and normalize the image
        image = Image.open(image_file).convert('RGB').resize((128, 128))  # Resize to match model's expected input
        image = np.array(image) / 255.0  # Normalize pixel values between 0 and 1
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict
        predictions = model.predict(image)
        predicted_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_idx]

        return predicted_idx, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        print(e)
        return None, None


# Sidebar Menu
st.sidebar.title("Skin Cancer Prediction Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])


# Main Pages
if app_mode == "Home":
    st.header("🌿 Skin Cancer Detection Dashboard")
    st.markdown("""
    This system allows you to:
    - Train a model with your own CSV dataset.
    - Test your own image to check for skin cancer risk.
    - Use a pre-trained model pipeline.
    """)

elif app_mode == "Train & Test Model":
    st.header("Train & Test Model")
    uploaded_file = st.file_uploader("Upload your CSV file for training", type=["csv"])

    if uploaded_file:
        st.info("Dataset loaded successfully. Preparing for training...")
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)
                create_and_train_model(X_train, y_train)

elif app_mode == "Prediction":
    st.header("Make Predictions")
    uploaded_image = st.file_uploader("Upload a skin image for prediction", type=["jpg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Run Prediction"):
            with st.spinner("Running prediction..."):
                predicted_idx, confidence = run_prediction(uploaded_image)
                if predicted_idx is not None:
                    st.success(f"Prediction Confidence: {confidence:.2f}")
                    st.write(f"Predicted Class Index: {predicted_idx}")

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    This web app uses machine learning techniques to predict skin cancer risk from dermoscopic image data.
    Built with Streamlit & TensorFlow, this application allows model training, testing with custom image data, 
    and leveraging machine learning models for inference.
    """)



