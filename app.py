import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from PIL import Image
import matplotlib.pyplot as plt


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


def create_and_train_model(X_train, y_train, X_test, y_test):
    """
    Defines, compiles, and trains a CNN model for image classification.
    Handles class imbalance by computing class weights.
    Saves the trained model after training.
    """
    # Reshape for CNN processing (convolutional models expect 4D inputs)
    X_train = X_train.reshape((-1, 128, 128, 3))
    X_test = X_test.reshape((-1, 128, 128, 3))

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
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(y_train.shape[1], activation="softmax")  # Output layer
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
    model.save('trained_skin_cancer_model.keras')
    st.success("✅ Model trained and saved successfully!")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"🔍 Test Accuracy: {accuracy:.2%}")

    # Plotting training history
    st.write("Training History:")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    st.pyplot(plt)

    return model


def preprocess_uploaded_image(image_file):
    """
    Preprocess the uploaded image into numerical features expected by the model.
    Normalize image size and scale to match the training process.
    """
    try:
        # Open the image and resize
        image = Image.open(image_file).convert('RGB').resize((128, 128))  # Resize to expected input dimensions
        image = np.array(image) / 255.0  # Normalize pixel values to 0-1
        image = image.reshape(1, 128, 128, 3)  # Reshape for prediction

        st.write("Image shape being sent to model:", image.shape)  # Debugging step
        return image
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
                create_and_train_model(X_train, y_train, X_test, y_test)

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
