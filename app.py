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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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

    return X_train, X_test, y_train, y_test, label_encoder, df


def plot_class_distribution(df):
    """
    Visualize class distribution to identify imbalances in the dataset.
    """
    class_counts = df['dx'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    class_counts.plot(kind='bar', ax=ax, color="orange")
    ax.set_title("Class Distribution in Uploaded Dataset")
    ax.set_xlabel("Disease Types")
    ax.set_ylabel("Number of Cases")
    st.pyplot(fig)


def create_and_train_model(X_train, y_train, X_test, y_test):
    """
    Defines, compiles, and trains a basic model for classification.
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

    # Model architecture
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dense(y_train.shape[1], activation="softmax")
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=16,
        class_weight=class_weights_dict,
        verbose=2
    )

    # Plotting training history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title("Model Accuracy Over Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title("Loss Over Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    st.pyplot(fig)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"🔍 Model Test Accuracy: {accuracy:.2%}")
    st.success(f"🔍 Model Test Loss: {loss:.4f}")

    # Confusion matrix visualization
    y_test_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_test_pred)
    st.write("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)
    ax.set_xticks(np.arange(len(cm)))
    ax.set_yticks(np.arange(len(cm)))
    ax.set_xticklabels([f"Class {i}" for i in range(len(cm))])
    ax.set_yticklabels([f"Class {i}" for i in range(len(cm))])
    st.pyplot(fig)

    # Save the model
    model.save('trained_skin_cancer_model.keras')
    st.success("✅ Model trained and saved successfully!")
    return model


def run_prediction(image_file):
    """
    Run prediction on an uploaded image after preprocessing it into expected numerical features.
    """
    # Load the trained model
    model = tf.keras.models.load_model('trained_skin_cancer_model.keras')

    # Preprocess the uploaded image
    features = preprocess_uploaded_image(image_file)

    if features is not None:
        try:
            predictions = model.predict(features)
            predicted_idx = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_idx]
            return predicted_idx, confidence
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            return None, None
    else:
        return None, None


# Sidebar Menu
st.sidebar.title("🩺 Skin Cancer Prediction Dashboard")
app_mode = st.sidebar.radio("Choose an option", ["Home", "Train Model", "Make Prediction", "About"])

DISEASE_MAPPING = {
    0: "Melanoma",
    1: "Basal Cell Carcinoma",
    2: "Squamous Cell Carcinoma",
    3: "Benign Lesion"
}

# Main Pages
if app_mode == "Train Model":
    uploaded_file = st.file_uploader("Upload CSV for Training", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset Preview", df.head())
        if st.button("Visualize Class Distribution"):
            plot_class_distribution(df)
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                X_train, X_test, y_train, y_test, _, df = preprocess_data(df)
                create_and_train_model(X_train, y_train, X_test, y_test)

