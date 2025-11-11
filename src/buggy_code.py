"""
BUGGY TENSORFLOW CODE - Debugging Challenge

This code contains several common TensorFlow errors that need to be fixed.
Try to identify and fix all bugs before looking at the solution!

Task: Train a simple neural network on MNIST dataset
Expected: Code should run without errors and achieve >95% accuracy
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# BUG #1: Incorrect data preprocessing
# Images need to be normalized and reshaped properly
X_train = X_train.reshape(-1, 28, 28)  # Missing channel dimension
X_test = X_test.reshape(-1, 28, 28)    # Missing channel dimension
# Not normalizing the pixel values!

# BUG #2: Wrong label encoding
# Using sparse labels with categorical_crossentropy loss
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# BUG #3: Dimension mismatch in model architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    # BUG: Wrong number of output units
    layers.Dense(9, activation='softmax')  # Should be 10 for 10 digits!
])

# BUG #4: Incorrect loss function for the label encoding
# Using categorical_crossentropy but should match label format
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Wrong! We one-hot encoded labels
    metrics=['accuracy']
)

# Print model summary
print("Model Architecture:")
model.summary()

# BUG #5: Wrong data types being passed to fit()
# Need to ensure data types are correct
print("\nTraining model...")
history = model.fit(
    X_train,  # Not matching the input shape expected by first layer!
    y_train,
    epochs=5,
    batch_size=128,
    validation_data=(X_test, y_test)
)

# BUG #6: Incorrect evaluation metric interpretation
# Test the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_loss:.4f}")  # Printing loss instead of accuracy!
print(f"Test loss: {test_acc:.4f}")         # Printing accuracy instead of loss!

# BUG #7: Wrong prediction format
# Make predictions on a few samples
predictions = model.predict(X_test[:5])
print("\nPredictions shape:", predictions.shape)
print("Predicted classes:", predictions)  # Should use argmax to get class labels!
print("True labels:", y_test[:5])

# BUG #8: Tensor shape mismatch when adding custom preprocessing
# Try to add a preprocessing step
def preprocess(image):
    # BUG: Inconsistent dimensions
    image = tf.expand_dims(image, -1)  # Add channel dimension
    image = image / 255.0
    return image

# This will cause issues because we're trying to apply it inconsistently
sample = X_train[0]
processed = preprocess(sample)
print("\nProcessed shape:", processed.shape)  # Will work here

# But trying to predict with inconsistent preprocessing will fail
sample_pred = model.predict(processed)  # Missing batch dimension!

print("\n❌ If you see errors, congratulations! You found the bugs!")
print("Now try to fix them all!")
