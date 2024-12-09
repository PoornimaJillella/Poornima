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
    st.success("✅ Model trained and saved successfully!")
    return model


def preprocess_uploaded_image(image_file):
    """
    Preprocess the uploaded image into numerical features expected by the model.
    """
    try:
        # Open the image and resize
        image = Image.open(image_file).convert('RGB').resize((128, 128))  # Resize image
        image = np.array(image) / 255.0  # Normalize pixel values to 0-1

        # Extract numerical features: compute mean pixel intensity across each channel (R, G, B)
        mean_red = np.mean(image[:, :, 0])
        mean_green = np.mean(image[:, :, 1])
        mean_blue = np.mean(image[:, :, 2])
        # Extract additional statistics
        image_features = np.array([mean_red, mean_green, mean_blue, np.mean(image)])  # Example: mean pixel values
        image_features = np.expand_dims(image_features, axis=0)  # Reshape for prediction

        return image_features
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        print(e)
        return None


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

            return predicted_idx, confidence
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            print(e)
            return None, None
    else:
        return None, None


# Sidebar Menu
st.sidebar.title("🩺 Skin Cancer Prediction Dashboard")
app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Train & Test Model", "Prediction", "About"])


# Mapping indices to disease names
DISEASE_MAPPING = {
    0: "Melanoma",
    1: "Basal Cell Carcinoma",
    2: "Squamous Cell Carcinoma",
    3: "Benign Lesion"
}


# Main Pages
if app_mode == "Home":
    st.title("🌿 Skin Cancer Detection App")
    st.markdown("""
    This web app allows you to:
    - Train a model with your own CSV dataset.
    - Test your uploaded image to check for skin cancer risk.
    - Use a pre-trained model for instant predictions.
    """)

elif app_mode == "Train & Test Model":
    st.header("🛠 Train & Test Model")
    uploaded_file = st.file_uploader("Upload your CSV file for training", type=["csv"])

    if uploaded_file:
        st.info("📊 Dataset loaded successfully. Preparing for training...")
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())

        if st.button("Train Model"):
            with st.spinner("🔄 Training model..."):
                X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)
                create_and_train_model(X_train, y_train)

elif app_mode == "Prediction":
    st.header("🔮 Make Predictions")
    uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Run Prediction"):
            with st.spinner("⏳ Running prediction..."):
                predicted_idx, confidence = run_prediction(uploaded_image)
                if predicted_idx is not None:
                    disease_name = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")
                    st.success(f"✅ Prediction Confidence: {confidence:.2f}")
                    st.subheader(f"Predicted Disease: {disease_name}")

elif app_mode == "About":
    st.header("📖 About This App")
    st.markdown("""
    This web application uses machine learning techniques to predict skin cancer risk from dermoscopic image data.
    It was built using *Streamlit, **TensorFlow, and **Python*, and allows:
    - Model training with your own labeled datasets.
    - Testing using your uploaded image for prediction.
    - Real-time predictions from trained models.
    """)








