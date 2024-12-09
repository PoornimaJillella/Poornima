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
    ax.set_xlabel("Disease Type")
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

    # Training history visualization
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


# Home Page
if st.sidebar.selectbox("Navigation", ["Home", "Train Model", "Prediction", "About"]) == "Home":
    st.title("🌿 Skin Cancer Detection Dashboard")
    st.markdown("""
    **Welcome to the Skin Cancer Detection App!** 🚀
    This app uses machine learning to:
    - Train and test models using dermoscopic images or CSV data.
    - Allow you to upload images to determine the likelihood of a skin cancer diagnosis.
    - Visualize insights about data imbalances, model accuracy, and more.
    """)

    st.markdown("---")
    st.subheader("How to Use:")
    st.write("""
    1. **Train Model**:
        Upload CSV data containing images and labels to train your custom model.
    2. **Prediction**:
        Upload an image and predict skin cancer risk using our trained model.
    3. **Explore Insights**:
        Visualize model performance graphs and confusion matrices.
    """)
    st.markdown("---")
    st.write("Select the options from the sidebar to get started!")



