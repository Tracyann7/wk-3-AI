"""
FIXED TENSORFLOW CODE - Solution to Debugging Challenge

This is the corrected version with explanations for each bug fix.
All issues have been resolved and the code now runs successfully.

Task: Train a simple neural network on MNIST dataset
Result: Code runs without errors and achieves >95% accuracy
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("="*70)
print("FIXED VERSION - All Bugs Corrected")
print("="*70)

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(f"Original data shapes: X_train={X_train.shape}, y_train={y_train.shape}")

# ✅ FIX #1: Correct data preprocessing
# - Added channel dimension for Conv2D layers
# - Normalized pixel values to [0, 1] range
print("\n📝 FIX #1: Proper data preprocessing")
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
print(f"   Fixed: X_train shape = {X_train.shape}")
print(f"   Fixed: Pixel range = [{X_train.min():.2f}, {X_train.max():.2f}]")
print("   Explanation: Conv2D needs 4D input (batch, height, width, channels)")
print("                Normalization helps with training stability")

# ✅ FIX #2: Correct label encoding
# - Keep labels as integers (sparse format)
# - OR one-hot encode them (but then use categorical_crossentropy)
# - Here we keep them sparse for efficiency
print("\n📝 FIX #2: Correct label encoding")
print(f"   Using sparse labels: {y_train[:5]}")
print(f"   Shape: {y_train.shape}")
print("   Explanation: Sparse labels (integers) are more memory efficient")
print("                Use with 'sparse_categorical_crossentropy' loss")

# Alternative approach (commented out):
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)
# # Then use 'categorical_crossentropy' loss

# ✅ FIX #3: Correct output layer dimension
# - Changed from 9 to 10 neurons for 10 digit classes
print("\n📝 FIX #3: Correct model architecture")
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # ✅ Fixed: 10 outputs for 10 classes
])
print("   Fixed: Output layer has 10 neurons (one per digit class)")
print("   Explanation: Must match number of classes in classification task")

# ✅ FIX #4: Correct loss function
# - Using sparse_categorical_crossentropy with integer labels
# - Matches our label format (sparse, not one-hot)
print("\n📝 FIX #4: Matching loss function to label format")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # ✅ Fixed: Matches integer labels
    metrics=['accuracy']
)
print("   Fixed: Using 'sparse_categorical_crossentropy'")
print("   Explanation: This loss works with integer labels (0-9)")
print("                Use 'categorical_crossentropy' for one-hot encoded labels")

# Print model summary
print("\n📐 Model Architecture:")
print("="*70)
model.summary()
print("="*70)

# ✅ FIX #5: Data now properly shaped and typed
# - Input shape (?, 28, 28, 1) matches first Conv2D layer
# - Data type is float32 (normalized)
print("\n🏋️ Training model...")
print("   Data shapes match model expectations")
print("   X_train shape:", X_train.shape, "matches input_shape=(28, 28, 1)")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_data=(X_test, y_test),
    verbose=1
)

# ✅ FIX #6: Correct variable names for loss and accuracy
# - test_loss contains loss value
# - test_acc contains accuracy value
print("\n📊 Evaluation Results:")
print("="*70)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"   Test Loss: {test_loss:.4f}")        # ✅ Fixed: Correct label
print(f"   Test Accuracy: {test_acc:.4f}")     # ✅ Fixed: Correct label
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print("="*70)

if test_acc > 0.95:
    print("   ✅ SUCCESS: Achieved >95% accuracy!")
else:
    print(f"   ⚠️ Achieved {test_acc*100:.2f}% accuracy (may need more epochs)")

# ✅ FIX #7: Use argmax to get predicted class labels
# - predictions are probabilities for each class
# - use np.argmax to get the class with highest probability
print("\n🔮 Making Predictions:")
print("="*70)
predictions = model.predict(X_test[:5], verbose=0)
predicted_classes = np.argmax(predictions, axis=1)  # ✅ Fixed: Get class labels
print("   Prediction probabilities shape:", predictions.shape)
print("   Predicted classes:", predicted_classes)
print("   True labels:", y_test[:5])
print("   Match:", np.array_equal(predicted_classes, y_test[:5]))
print("="*70)

# Display detailed predictions
print("\n📋 Detailed Predictions for First 5 Samples:")
for i in range(5):
    pred_class = predicted_classes[i]
    confidence = predictions[i][pred_class] * 100
    true_label = y_test[i]
    correct = "✓" if pred_class == true_label else "✗"
    print(f"   Sample {i+1}: Predicted={pred_class}, True={true_label}, "
          f"Confidence={confidence:.1f}% {correct}")

# ✅ FIX #8: Correct preprocessing with proper dimensions
# - Ensure batch dimension is included
# - Keep preprocessing consistent
print("\n🔧 Custom Preprocessing Example:")
print("="*70)

def preprocess_single_image(image):
    """
    Properly preprocess a single image for prediction

    Args:
        image: Single image (28, 28) or (28, 28, 1)

    Returns:
        Preprocessed image with shape (1, 28, 28, 1)
    """
    # Ensure correct shape
    if len(image.shape) == 2:  # If (28, 28)
        image = np.expand_dims(image, axis=-1)  # Add channel: (28, 28, 1)

    # Ensure batch dimension
    if len(image.shape) == 3:  # If (28, 28, 1)
        image = np.expand_dims(image, axis=0)   # Add batch: (1, 28, 28, 1)

    # Normalize if needed
    if image.max() > 1:
        image = image.astype('float32') / 255.0

    return image

# Test preprocessing
sample = X_test[0]  # Shape: (28, 28, 1)
print(f"   Original sample shape: {sample.shape}")

# Remove extra dimensions for testing
sample_2d = sample.reshape(28, 28)
print(f"   2D sample shape: {sample_2d.shape}")

# Preprocess correctly
processed = preprocess_single_image(sample_2d)
print(f"   Preprocessed shape: {processed.shape}")  # ✅ Fixed: (1, 28, 28, 1)

# Now prediction works!
sample_pred = model.predict(processed, verbose=0)
predicted_digit = np.argmax(sample_pred)
confidence = sample_pred[0][predicted_digit] * 100

print(f"   Prediction successful!")
print(f"   Predicted digit: {predicted_digit}")
print(f"   Confidence: {confidence:.2f}%")
print("="*70)

# Summary of all fixes
print("\n" + "="*70)
print("SUMMARY OF BUG FIXES")
print("="*70)
print("""
✅ FIX #1: Data Preprocessing
   Problem: Missing channel dimension, no normalization
   Solution: Reshape to (n, 28, 28, 1) and normalize to [0, 1]

✅ FIX #2: Label Encoding
   Problem: Inconsistency between label format and loss function
   Solution: Keep sparse labels (integers) for efficiency

✅ FIX #3: Model Architecture
   Problem: Output layer had 9 neurons instead of 10
   Solution: Changed to 10 neurons for 10 digit classes

✅ FIX #4: Loss Function
   Problem: Using categorical_crossentropy with one-hot labels
   Solution: Use sparse_categorical_crossentropy with integer labels

✅ FIX #5: Input Shape Mismatch
   Problem: Data shape didn't match Conv2D input requirements
   Solution: Properly reshaped data with channel dimension

✅ FIX #6: Variable Name Confusion
   Problem: Printing loss as accuracy and vice versa
   Solution: Correctly labeled loss and accuracy values

✅ FIX #7: Prediction Output
   Problem: Printing probabilities instead of class labels
   Solution: Use np.argmax() to get predicted class

✅ FIX #8: Inconsistent Preprocessing
   Problem: Missing batch dimension in single image prediction
   Solution: Proper preprocessing function with all dimensions
""")

print("="*70)
print("🎉 All bugs fixed! Code runs successfully!")
print("="*70)

# Save the model
model.save('../models/debugged_mnist_model.h5')
print("\n💾 Model saved to: ../models/debugged_mnist_model.h5")
