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
def extract_features_from_image(image_path):
    """
    Extract simple features from image. Modify this function to compute actual meaningful features.
    For simplicity, this function uses average pixel intensity features.
    """
    try:
        # Open the image and resize
        image = Image.open(image_path).convert('RGB').resize((128, 128))
        image = np.array(image) / 255.0  # Normalize pixel values
        
        # Calculate statistical features
        mean_red = np.mean(image[:, :, 0])
        mean_green = np.mean(image[:, :, 1])
        mean_blue = np.mean(image[:, :, 2])
        mean_intensity = np.mean(image)

        # Return feature vector
        return [mean_red, mean_green, mean_blue, mean_intensity]
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


def preprocess_data(df):
    """
    Preprocess uploaded CSV data for model training.
    Extract features and split data.
    """
    # Debugging: View column names in the uploaded CSV
    st.write("Column names in uploaded CSV:", df.columns)

    # Label encoding the target variable
    label_encoder = LabelEncoder()
    df['dx'] = label_encoder.fit_transform(df['dx'])

    # Dynamically look for the first column related to images
    image_column = None
    for col in df.columns:
        if 'image' in col.lower():
            image_column = col
            break

    if not image_column:
        st.error("Could not find a column with image paths in the uploaded CSV. Make sure your CSV has an image path column.")
        return None, None, None, None, None

    st.write(f"Using column '{image_column}' for image paths.")

    # Preprocess image paths to extract features
    features = []
    for index, row in df.iterrows():
        feature_set = extract_features_from_image(row[image_column])  # Extract image statistics
        if feature_set:
            features.append(feature_set)
        else:
            st.error(f"Could not process image at index {index}: {row[image_column]}")

    X = np.array(features)
    y = pd.get_dummies(df['dx']).to_numpy()

    # Handle missing values
    X = np.nan_to_num(X)  # Replace NaNs with zeros

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder


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

    # Define the model architecture
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dense(y_train.shape[1], activation="softmax")  # Adjust number of output neurons to match number of classes
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
        image = np.array(image) / 255.0  # Normalize pixel values to 0-1
        
        # Calculate mean pixel intensities as features
        mean_red = np.mean(image[:, :, 0])
        mean_green = np.mean(image[:, :, 1])
        mean_blue = np.mean(image[:, :, 2])
        mean_intensity = np.mean(image)  # General mean pixel intensity
        
        # Create feature array with 4 numerical values
        image_features = np.array([mean_red, mean_green, mean_blue, mean_intensity])
        image_features = np.expand_dims(image_features, axis=0)  # Reshape for prediction

        return image_features
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        print(e)
        return None


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
                X_train, X_test, y_train, y_test, le = preprocess_data(df)
                if X_train is not None:
                    create_and_train_model(X_train, y_train, X_test, y_test)

elif app_mode == "About":
    st.header("📖 About This App")
    st.markdown("""
    This web application uses machine learning techniques to predict skin cancer risk from dermoscopic image data.
    It allows:
    - Model training with your own labeled datasets.
    - Testing using your uploaded image for prediction.
    - Real-time predictions from trained models.
    """)




