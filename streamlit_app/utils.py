"""
Utility functions for the MNIST Digit Classifier Streamlit app
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def preprocess_canvas_image(canvas_data, target_size=(28, 28)):
    """
    Preprocess image from drawing canvas

    Args:
        canvas_data: Canvas image data
        target_size: Target size (28, 28)

    Returns:
        Preprocessed image
    """
    # Convert to grayscale
    if len(canvas_data.shape) == 3:
        gray = cv2.cvtColor(canvas_data, cv2.COLOR_RGB2GRAY)
    else:
        gray = canvas_data

    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours to crop the digit
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Get bounding box of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)

        # Crop to bounding box
        cropped = binary[y:y+h, x:x+w]

        # Make square by adding padding
        max_dim = max(w, h)
        square = np.zeros((max_dim, max_dim), dtype=np.uint8)

        # Center the digit
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

        # Resize to target size
        resized = cv2.resize(square, target_size)
    else:
        # If no contours found, just resize
        resized = cv2.resize(binary, target_size)

    # Invert (model expects white digit on black background)
    inverted = 255 - resized

    # Normalize
    normalized = inverted.astype('float32') / 255.0

    # Reshape for model
    processed = normalized.reshape(1, 28, 28, 1)

    return processed


def create_confidence_meter(confidence):
    """
    Create a visual confidence meter

    Args:
        confidence: Confidence score (0-100)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 2))

    # Determine color based on confidence
    if confidence >= 90:
        color = '#2ca02c'  # Green
        label = 'High Confidence'
    elif confidence >= 70:
        color = '#ff7f0e'  # Orange
        label = 'Medium Confidence'
    else:
        color = '#d62728'  # Red
        label = 'Low Confidence'

    # Create horizontal bar
    ax.barh([0], [confidence], height=0.5, color=color, alpha=0.7)
    ax.barh([0], [100], height=0.5, color='lightgray', alpha=0.3, zorder=0)

    # Add text
    ax.text(confidence/2, 0, f'{confidence:.1f}%',
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(50, -0.8, label, ha='center', va='top',
            fontsize=12, fontweight='bold', color=color)

    # Styling
    ax.set_xlim([0, 100])
    ax.set_ylim([-1, 1])
    ax.axis('off')

    plt.tight_layout()
    return fig


def batch_predict(model, images):
    """
    Make predictions on multiple images

    Args:
        model: Trained Keras model
        images: List of preprocessed images

    Returns:
        List of (predicted_digit, confidence) tuples
    """
    results = []

    for img in images:
        predictions = model.predict(img, verbose=0)[0]
        predicted_digit = np.argmax(predictions)
        confidence = predictions[predicted_digit] * 100
        results.append((predicted_digit, confidence, predictions))

    return results


def get_misclassified_examples(model, X_test, y_test, num_examples=10):
    """
    Find misclassified examples from test set

    Args:
        model: Trained model
        X_test: Test images
        y_test: True labels
        num_examples: Number of examples to return

    Returns:
        List of misclassified examples
    """
    # Normalize if needed
    if X_test.max() > 1:
        X_test_norm = X_test.astype('float32') / 255.0
    else:
        X_test_norm = X_test

    # Reshape if needed
    if len(X_test_norm.shape) == 3:
        X_test_norm = X_test_norm.reshape(-1, 28, 28, 1)

    # Make predictions
    predictions = model.predict(X_test_norm, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    # Find misclassified
    misclassified_idx = np.where(predicted_labels != y_test)[0]

    # Select random subset
    if len(misclassified_idx) > num_examples:
        selected_idx = np.random.choice(
            misclassified_idx, num_examples, replace=False
        )
    else:
        selected_idx = misclassified_idx

    results = []
    for idx in selected_idx:
        results.append({
            'image': X_test[idx],
            'true_label': y_test[idx],
            'predicted_label': predicted_labels[idx],
            'confidence': predictions[idx][predicted_labels[idx]] * 100
        })

    return results


def augment_image(image, augmentation_type='rotate'):
    """
    Apply augmentation to an image

    Args:
        image: Input image (28x28)
        augmentation_type: Type of augmentation

    Returns:
        Augmented image
    """
    if augmentation_type == 'rotate':
        angle = np.random.randint(-10, 10)
        rows, cols = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        augmented = cv2.warpAffine(image, M, (cols, rows))

    elif augmentation_type == 'shift':
        shift_x = np.random.randint(-2, 3)
        shift_y = np.random.randint(-2, 3)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        augmented = cv2.warpAffine(image, M, image.shape[::-1])

    elif augmentation_type == 'zoom':
        zoom_factor = np.random.uniform(0.9, 1.1)
        rows, cols = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 0, zoom_factor)
        augmented = cv2.warpAffine(image, M, (cols, rows))

    else:
        augmented = image

    return augmented


def visualize_feature_maps(model, image, layer_name):
    """
    Visualize feature maps from a specific layer

    Args:
        model: Trained model
        image: Input image
        layer_name: Name of layer to visualize

    Returns:
        Feature maps
    """
    # Create a model that outputs the specified layer
    layer_output = model.get_layer(layer_name).output
    feature_model = tf.keras.Model(inputs=model.input, outputs=layer_output)

    # Get feature maps
    feature_maps = feature_model.predict(image, verbose=0)

    return feature_maps


def calculate_model_metrics(y_true, y_pred):
    """
    Calculate comprehensive model metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

    return metrics


def format_large_number(num):
    """Format large numbers with commas"""
    return f"{num:,}"


def get_model_size(model):
    """
    Calculate model size in MB

    Args:
        model: Keras model

    Returns:
        Model size in MB
    """
    import tempfile
    import os

    # Save model to temporary file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        model.save(tmp.name)
        size_bytes = os.path.getsize(tmp.name)
        os.unlink(tmp.name)

    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def create_comparison_plot(original, preprocessed):
    """
    Create side-by-side comparison of original and preprocessed images

    Args:
        original: Original image
        preprocessed: Preprocessed image

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Preprocessed
    if len(preprocessed.shape) == 4:
        preprocessed = preprocessed.reshape(28, 28)
    axes[1].imshow(preprocessed, cmap='gray')
    axes[1].set_title('Preprocessed (28×28)', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    return fig
