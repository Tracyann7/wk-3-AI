"""
MNIST Digit Classifier - Streamlit Web Application

This app allows users to:
1. Draw digits on a canvas
2. Upload digit images
3. Get real-time predictions with confidence scores
4. View model architecture and performance metrics
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #2ca02c;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .confidence-high {
        color: #2ca02c;
        font-weight: bold;
        font-size: 24px;
    }
    .confidence-medium {
        color: #ff7f0e;
        font-weight: bold;
        font-size: 24px;
    }
    .confidence-low {
        color: #d62728;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the pre-trained MNIST model with caching"""
    try:
        model = keras.models.load_model('../models/mnist_model.h5')
        return model, None
    except Exception as e:
        return None, str(e)


def preprocess_image(image, target_size=(28, 28)):
    """
    Preprocess image for model prediction

    Args:
        image: PIL Image or numpy array
        target_size: Target size for the model (28, 28)

    Returns:
        Preprocessed image ready for prediction
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif image.shape[2] == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize to 28x28
    image = cv2.resize(image, target_size)

    # Invert if needed (model expects white digit on black background)
    if np.mean(image) > 127:
        image = 255 - image

    # Normalize to [0, 1]
    image = image.astype('float32') / 255.0

    # Reshape for model input (1, 28, 28, 1)
    image = image.reshape(1, 28, 28, 1)

    return image


def make_prediction(model, image):
    """
    Make prediction on preprocessed image

    Returns:
        predicted_digit, confidence, all_probabilities
    """
    # Get predictions
    predictions = model.predict(image, verbose=0)[0]

    # Get predicted digit and confidence
    predicted_digit = np.argmax(predictions)
    confidence = predictions[predicted_digit] * 100

    return predicted_digit, confidence, predictions


def plot_probability_distribution(probabilities):
    """Create a bar plot of prediction probabilities"""
    fig, ax = plt.subplots(figsize=(10, 4))

    digits = range(10)
    colors = ['#2ca02c' if i == np.argmax(probabilities) else '#1f77b4'
              for i in digits]

    bars = ax.bar(digits, probabilities, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.1f}%',
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Digit', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(digits)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    return fig


def main():
    # Header
    st.markdown('<p class="main-header">🔢 MNIST Handwritten Digit Classifier</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Load model
    with st.spinner("Loading model..."):
        model, error = load_model()

    if error:
        st.error(f"Error loading model: {error}")
        st.info("Please make sure you've trained the model first by running the Task 2 notebook!")
        st.stop()

    st.success("✓ Model loaded successfully!")

    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["Digit Prediction", "Model Info", "About"]
    )

    if page == "Digit Prediction":
        show_prediction_page(model)
    elif page == "Model Info":
        show_model_info(model)
    else:
        show_about_page()


def show_prediction_page(model):
    """Main prediction page"""
    st.markdown('<p class="sub-header">Draw or Upload a Digit</p>', unsafe_allow_html=True)

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Use Sample Images"],
        horizontal=True
    )

    if input_method == "Upload Image":
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload an image of a handwritten digit",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a single digit (0-9)"
        )

        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True)

            # Preprocess and predict
            processed_image = preprocess_image(image)

            with col2:
                st.markdown("### Preprocessed Image")
                st.image(processed_image.reshape(28, 28),
                        use_container_width=True,
                        clamp=True,
                        channels="GRAY")

            # Make prediction
            with st.spinner("Analyzing..."):
                predicted_digit, confidence, probabilities = make_prediction(
                    model, processed_image
                )

            # Display results
            display_prediction_results(predicted_digit, confidence, probabilities)

    else:
        # Sample images
        st.info("Sample images will be loaded from the MNIST test dataset")

        # Load sample MNIST images
        try:
            from tensorflow.keras.datasets import mnist
            (_, _), (X_test, y_test) = mnist.load_data()

            # Random sample
            if st.button("🎲 Get Random Sample", type="primary"):
                st.session_state.sample_idx = np.random.randint(0, len(X_test))

            if 'sample_idx' not in st.session_state:
                st.session_state.sample_idx = np.random.randint(0, len(X_test))

            idx = st.session_state.sample_idx
            sample_image = X_test[idx]
            true_label = y_test[idx]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Sample Digit")
                st.image(sample_image, use_container_width=True, clamp=True)
                st.info(f"True Label: **{true_label}**")

            # Preprocess and predict
            processed_image = sample_image.reshape(1, 28, 28, 1).astype('float32') / 255.0

            predicted_digit, confidence, probabilities = make_prediction(
                model, processed_image
            )

            with col2:
                st.markdown("### Prediction")
                display_prediction_results(predicted_digit, confidence, probabilities)

                # Check if correct
                if predicted_digit == true_label:
                    st.success("✓ Correct prediction!")
                else:
                    st.error(f"✗ Incorrect! True label: {true_label}")

        except Exception as e:
            st.error(f"Error loading sample images: {e}")


def display_prediction_results(predicted_digit, confidence, probabilities):
    """Display prediction results with styling"""
    # Confidence styling
    if confidence >= 90:
        conf_class = "confidence-high"
    elif confidence >= 70:
        conf_class = "confidence-medium"
    else:
        conf_class = "confidence-low"

    # Prediction box
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown(f"### Predicted Digit: <span class='{conf_class}'>{predicted_digit}</span>",
                unsafe_allow_html=True)
    st.markdown(f"### Confidence: <span class='{conf_class}'>{confidence:.2f}%</span>",
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Probability distribution
    st.markdown("### Probability Distribution")
    fig = plot_probability_distribution(probabilities)
    st.pyplot(fig)
    plt.close()

    # Detailed probabilities
    with st.expander("📊 View Detailed Probabilities"):
        prob_df = pd.DataFrame({
            'Digit': range(10),
            'Probability': probabilities,
            'Percentage': [f"{p*100:.2f}%" for p in probabilities]
        }).sort_values('Probability', ascending=False)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)


def show_model_info(model):
    """Display model architecture and information"""
    st.markdown('<p class="sub-header">Model Information</p>', unsafe_allow_html=True)

    # Model summary
    st.markdown("### 📐 Model Architecture")

    # Create a string buffer to capture model summary
    import io
    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + '\n'))
    summary_string = buffer.getvalue()

    st.code(summary_string, language='text')

    # Model statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        total_params = model.count_params()
        st.metric("Total Parameters", f"{total_params:,}")

    with col2:
        num_layers = len(model.layers)
        st.metric("Number of Layers", num_layers)

    with col3:
        st.metric("Input Shape", "28×28×1")

    # Layer details
    st.markdown("### 🔍 Layer Details")
    layer_data = []
    for layer in model.layers:
        layer_data.append({
            'Layer Name': layer.name,
            'Type': layer.__class__.__name__,
            'Output Shape': str(layer.output_shape),
            'Parameters': layer.count_params()
        })

    layer_df = pd.DataFrame(layer_data)
    st.dataframe(layer_df, use_container_width=True, hide_index=True)

    # Model performance
    st.markdown("### 📊 Expected Performance")
    st.info("""
    **Model Performance Metrics:**
    - Test Accuracy: ~98%+
    - Training Time: ~5-10 minutes (with GPU)
    - Inference Time: <10ms per image
    - Model Size: ~2.5 MB

    **Architecture Highlights:**
    - 2 Convolutional blocks with BatchNormalization
    - MaxPooling for spatial dimension reduction
    - Dropout layers for regularization
    - Dense layers for classification
    """)


def show_about_page():
    """Display about information"""
    st.markdown('<p class="sub-header">About This Application</p>', unsafe_allow_html=True)

    st.markdown("""
    ## 🎯 Project Overview

    This is a web application for classifying handwritten digits using a Convolutional Neural Network (CNN)
    trained on the MNIST dataset.

    ### 🔬 Technology Stack

    - **Framework**: TensorFlow/Keras
    - **Web App**: Streamlit
    - **Image Processing**: OpenCV, Pillow
    - **Visualization**: Matplotlib, Seaborn

    ### 📚 MNIST Dataset

    - **Training samples**: 60,000 images
    - **Test samples**: 10,000 images
    - **Image size**: 28×28 pixels (grayscale)
    - **Classes**: 10 digits (0-9)

    ### 🏗️ Model Architecture

    The CNN model consists of:
    1. **Convolutional Layers**: Extract spatial features
    2. **Batch Normalization**: Stabilize training
    3. **MaxPooling**: Reduce dimensionality
    4. **Dropout**: Prevent overfitting
    5. **Dense Layers**: Final classification

    ### ✨ Features

    - 📤 Upload custom digit images
    - 🎲 Test on random MNIST samples
    - 📊 View prediction confidence scores
    - 📈 Visualize probability distributions
    - 🔍 Explore model architecture

    ### 👥 Team Members

    This project was developed as part of an AI Tools and Frameworks assignment.

    ### 📝 License

    This project is created for educational purposes.

    ---

    ### 📖 How to Use

    1. **Navigate** to the "Digit Prediction" page
    2. **Choose** input method (Upload or Sample)
    3. **Get** instant predictions with confidence scores
    4. **Explore** model architecture in "Model Info"

    ### 🚀 Future Improvements

    - Add drawing canvas for freehand digit input
    - Support batch predictions
    - Add model comparison feature
    - Implement attention visualization
    - Export prediction reports

    ---

    Made with ❤️ using Streamlit
    """)


if __name__ == "__main__":
    main()
