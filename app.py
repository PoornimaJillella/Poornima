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

    # Define the model architecture
    num_classes = len(class_weights_dict)
    model = Sequential([
        Dense(64, activation="relu", input_shape=(4,)),  # Input shape matches feature vector length
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")  # Dynamically match the number of classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with class weights
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=16,
        class_weight=class_weights_dict,
        verbose=2
    )

    # Ensure directory exists for saving the model
    os.makedirs('./data', exist_ok=True)
    model_save_path = './data/trained_skin_cancer_model.keras'
    model.save(model_save_path)
    st.success(f"✅ Model trained and saved to: {model_save_path}")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"🔍 Test Accuracy: {accuracy:.2%}")
    
    return model


def preprocess_uploaded_image(image_file):
    """
    Preprocess the uploaded image into numerical features expected by the model.
    This converts the uploaded image into a 4-dimensional vector expected by the trained model.
    """
    try:
        # Open the image and resize
        image = Image.open(image_file).convert('RGB').resize((128, 128))
        image = np.array(image) / 255.0  # Normalize pixel values

        # Flatten or average features across the image dimensions
        features = image.mean(axis=(0, 1))  # Mean across spatial dimensions for RGB
        if features.shape[0] != 3:
            st.error(f"Unexpected feature shape: {features.shape}")
            return None
        # Add a constant alpha-like channel
        alpha_channel = np.array([0.5])  # Example value; adjust normalization as needed
        features_with_alpha = np.concatenate((features, alpha_channel))  # Shape should now match 4

        st.write("Processed feature vector ready for prediction:", features_with_alpha.shape)

        return np.expand_dims(features_with_alpha, axis=0)  # Reshape into shape (1, 4)
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


def run_prediction(image_file, model):
    """
    Preprocesses the image, runs it through the trained model, and returns the predicted class.
    Introduces variance using Monte Carlo Dropout.
    """
    features = preprocess_uploaded_image(image_file)

    if features is not None:
        try:
            # Perform prediction with stochastic inference (Monte Carlo Dropout)
            n_samples = 100
            f = K.function([model.input, K.learning_phase()], [model.output])
            predictions = np.array([f([features, 1])[0] for _ in range(n_samples)])
            mean_prediction = predictions.mean(axis=0)
            predicted_idx = np.argmax(mean_prediction, axis=1)[0]
            confidence = mean_prediction[0][predicted_idx]

            # Map index to disease
            disease_name = DISEASE_MAPPING.get(predicted_idx, "Unknown Disease")

            # Display the prediction and confidence
            st.success(f"✅ Prediction Confidence: {confidence:.2%}")
            st.subheader(f"Predicted Disease: {disease_name}")
            st.info(f"Based on uploaded image.")
            return predicted_idx, confidence
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            print(e)
    else:
        st.error("Failed to preprocess uploaded image.")
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
                model = create_and_train_model(X_train, y_train, X_test, y_test)

elif app_mode == "Prediction":
    st.header("🔮 Make Predictions")
    uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Run Prediction"):
            with st.spinner("⏳ Running prediction..."):
                if os.path.exists('./data/trained_skin_cancer_model.keras'):
                    model = tf.keras.models.load_model('./data/trained_skin_cancer_model.keras')
                    run_prediction(uploaded_image, model)
                else:
                    st.error("Model not trained. Train a model first.")

elif app_mode == "About":
    st.header("📖 About This App")
    st.markdown("""
    This web application uses machine learning techniques to predict skin cancer risk from dermoscopic image data.
    Built with Streamlit and TensorFlow, it allows:
    - Model training with custom datasets.
    - Real-time predictions using uploaded images.
    - Dynamic visualization and prediction results based on input data.
    """)















