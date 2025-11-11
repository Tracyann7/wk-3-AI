# TensorFlow Debugging Challenge - Comprehensive Guide

## Overview

This document provides detailed explanations of common TensorFlow errors and their solutions. The buggy code contains 8 intentional mistakes that reflect real-world issues developers encounter.

---

## Bug #1: Incorrect Data Preprocessing

### The Problem

```python
# ❌ BUGGY CODE
X_train = X_train.reshape(-1, 28, 28)  # Missing channel dimension
X_test = X_test.reshape(-1, 28, 28)    # Missing channel dimension
# Not normalizing the pixel values!
```

### Why It's Wrong

1. **Missing Channel Dimension**: Conv2D layers in TensorFlow/Keras expect 4D input:
   - Dimension 0: Batch size
   - Dimension 1: Height
   - Dimension 2: Width
   - Dimension 3: Channels (1 for grayscale, 3 for RGB)

2. **No Normalization**: MNIST pixels range from 0-255, which can cause:
   - Slow convergence
   - Numerical instability
   - Poor gradient flow
   - Lower final accuracy

### The Error Message

```
ValueError: Input 0 of layer "conv2d" is incompatible with the layer:
expected min_ndim=4, found ndim=3. Full shape received: (None, 28, 28)
```

### The Fix

```python
# ✅ FIXED CODE
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
```

**Explanation:**
- `.reshape(-1, 28, 28, 1)`: Adds channel dimension (1 for grayscale)
- `.astype('float32')`: Converts to float for division
- `/ 255.0`: Normalizes pixel values to [0, 1] range

### Key Lesson

**Always match your input dimensions to what your first layer expects!**

---

## Bug #2: Label Encoding Mismatch

### The Problem

```python
# ❌ BUGGY CODE
y_train = keras.utils.to_categorical(y_train, 10)  # One-hot encoding
y_test = keras.utils.to_categorical(y_test, 10)

# Later...
model.compile(
    loss='sparse_categorical_crossentropy',  # Expects integer labels!
    ...
)
```

### Why It's Wrong

**Mismatch between label format and loss function:**
- `to_categorical()` creates one-hot encoded labels: `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]`
- `sparse_categorical_crossentropy` expects integer labels: `2`

This combination will cause shape mismatch errors during training.

### The Error Message

```
ValueError: Shapes (None, 10) and (None, 10, 10) are incompatible
```

### The Solution (Two Approaches)

**Option 1: Use Sparse Labels (Recommended - More Efficient)**
```python
# ✅ Keep labels as integers
# y_train stays as [0, 1, 2, ..., 9]

model.compile(
    loss='sparse_categorical_crossentropy',  # ✅ Matches integer labels
    ...
)
```

**Option 2: Use One-Hot Encoded Labels**
```python
# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.compile(
    loss='categorical_crossentropy',  # ✅ Matches one-hot labels
    ...
)
```

### Key Lesson

**Match your loss function to your label format:**
- **Sparse labels (integers)** → `sparse_categorical_crossentropy`
- **One-hot labels** → `categorical_crossentropy`

---

## Bug #3: Wrong Output Layer Dimension

### The Problem

```python
# ❌ BUGGY CODE
layers.Dense(9, activation='softmax')  # Only 9 outputs for 10 classes!
```

### Why It's Wrong

- MNIST has **10 digit classes** (0-9)
- Output layer has only **9 neurons**
- Model cannot predict class 9 (or class 0, depending on how you see it)
- Causes shape mismatch with labels

### The Error Message

```
InvalidArgumentError: Received a label value of 9 which is outside the
valid range of [0, 9). Label values: 0 3 5 9 1
```

### The Fix

```python
# ✅ FIXED CODE
layers.Dense(10, activation='softmax')  # 10 outputs for 10 classes
```

**Explanation:**
- Output neurons must match the number of classes
- Each neuron represents one class's probability
- Softmax ensures probabilities sum to 1

### Key Lesson

**Always match output layer size to the number of classes!**

| Task Type | Output Layer Size |
|-----------|-------------------|
| Binary Classification | 1 (sigmoid) or 2 (softmax) |
| Multi-class Classification | Number of classes (softmax) |
| Regression | 1 (linear/relu) |
| Multi-output Regression | Number of outputs (linear/relu) |

---

## Bug #4: Loss Function Selection

### The Problem

```python
# ❌ BUGGY CODE
# Labels are one-hot encoded
y_train = keras.utils.to_categorical(y_train, 10)

# But using wrong loss function
model.compile(
    loss='sparse_categorical_crossentropy',  # Expects integers!
    ...
)
```

### Why It's Wrong

**Loss function doesn't match label format:**

| Label Format | Loss Function |
|--------------|---------------|
| Integers (e.g., `2`) | `sparse_categorical_crossentropy` |
| One-hot (e.g., `[0,0,1,0,0,0,0,0,0,0]`) | `categorical_crossentropy` |

Using the wrong combination causes shape mismatches and training errors.

### The Error Message

```
ValueError: Shapes (32, 10, 1) and (32, 10) are incompatible
```

### The Fix

Match loss function to label format (as shown in Bug #2).

### Loss Function Decision Tree

```
Do you have class labels?
│
├─ YES → Are they integers (0, 1, 2, ...) or one-hot encoded?
│        │
│        ├─ Integers → sparse_categorical_crossentropy
│        └─ One-hot → categorical_crossentropy
│
└─ NO (continuous values) → Use regression loss
                           ├─ Mean Squared Error (MSE)
                           ├─ Mean Absolute Error (MAE)
                           └─ Huber Loss
```

### Key Lesson

**Always ensure your loss function matches your label encoding!**

---

## Bug #5: Input Shape Mismatch

### The Problem

```python
# ❌ BUGGY CODE
X_train = X_train.reshape(-1, 28, 28)  # 3D: (batch, height, width)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)),  # Expects 4D!
    ...
])
```

### Why It's Wrong

- Conv2D layers require 4D input: `(batch, height, width, channels)`
- Data is 3D: `(batch, height, width)`
- Shape mismatch causes immediate error

### The Error Message

```
ValueError: Input 0 of layer "conv2d" is incompatible with the layer:
expected min_ndim=4, found ndim=3
```

### The Fix

See Bug #1 - properly reshape data with channel dimension.

### Common Shape Requirements

| Layer Type | Expected Input Shape |
|------------|---------------------|
| Dense | 2D: `(batch, features)` |
| Conv1D | 3D: `(batch, timesteps, channels)` |
| Conv2D | 4D: `(batch, height, width, channels)` |
| LSTM | 3D: `(batch, timesteps, features)` |

### Debugging Tip

```python
# Print shapes at each step
print("Original shape:", X_train.shape)
X_train = X_train.reshape(-1, 28, 28, 1)
print("Reshaped:", X_train.shape)
print("First layer expects:", model.layers[0].input_shape)
```

### Key Lesson

**Always verify input shapes match layer expectations!**

---

## Bug #6: Variable Name Confusion

### The Problem

```python
# ❌ BUGGY CODE
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_loss:.4f}")  # Wrong variable!
print(f"Test loss: {test_acc:.4f}")       # Wrong variable!
```

### Why It's Wrong

- Variable names are swapped
- Confusing for readers and debugging
- May lead to incorrect interpretations of model performance

### The Fix

```python
# ✅ FIXED CODE
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss:.4f}")      # Correct!
print(f"Test accuracy: {test_acc:.4f}")   # Correct!
```

### Key Lesson

**Use descriptive variable names and double-check print statements!**

---

## Bug #7: Incorrect Prediction Format

### The Problem

```python
# ❌ BUGGY CODE
predictions = model.predict(X_test[:5])
print("Predicted classes:", predictions)  # Shows probabilities, not classes!
```

### Output

```
Predicted classes: [[0.01, 0.02, 0.89, 0.03, ...], ...]  # Probabilities!
```

### Why It's Wrong

- `model.predict()` returns probabilities for each class
- For a 10-class problem, output shape is `(n_samples, 10)`
- We need to convert probabilities to class labels using `argmax`

### The Fix

```python
# ✅ FIXED CODE
predictions = model.predict(X_test[:5])
predicted_classes = np.argmax(predictions, axis=1)
print("Predicted classes:", predicted_classes)  # [2, 0, 9, 0, 3]
print("True labels:", y_test[:5])
```

### Understanding the Output

```python
# Example for one sample
prediction = [0.01, 0.02, 0.89, 0.03, 0.01, 0.01, 0.01, 0.01, 0.00, 0.01]
#             0     1     2     3     4     5     6     7     8     9
#
# argmax returns index of maximum value: 2
# So predicted class is 2
```

### Key Lesson

**Use `np.argmax()` to convert probabilities to class predictions!**

---

## Bug #8: Batch Dimension Missing

### The Problem

```python
# ❌ BUGGY CODE
sample = X_train[0]  # Shape: (28, 28, 1)
prediction = model.predict(sample)  # Missing batch dimension!
```

### The Error Message

```
ValueError: Input 0 of layer "sequential" is incompatible with the layer:
expected shape=(None, 28, 28, 1), found shape=(28, 28, 1)
```

### Why It's Wrong

- Models expect batches, even for single predictions
- Input shape should be: `(batch_size, height, width, channels)`
- Single image shape is: `(height, width, channels)`
- Missing the first dimension (batch size)

### The Fix

```python
# ✅ FIXED CODE Method 1: Add batch dimension
sample = X_train[0]  # Shape: (28, 28, 1)
sample = np.expand_dims(sample, axis=0)  # Shape: (1, 28, 28, 1)
prediction = model.predict(sample)

# ✅ FIXED CODE Method 2: Use array slicing
prediction = model.predict(X_train[0:1])  # Automatically has batch dimension

# ✅ FIXED CODE Method 3: Reshape
sample = X_train[0].reshape(1, 28, 28, 1)
prediction = model.predict(sample)
```

### Complete Preprocessing Function

```python
def preprocess_single_image(image):
    """Properly preprocess a single image for prediction"""

    # Handle different input shapes
    if len(image.shape) == 2:  # (28, 28)
        image = np.expand_dims(image, axis=-1)  # → (28, 28, 1)

    if len(image.shape) == 3:  # (28, 28, 1)
        image = np.expand_dims(image, axis=0)   # → (1, 28, 28, 1)

    # Normalize if needed
    if image.max() > 1:
        image = image.astype('float32') / 255.0

    return image
```

### Key Lesson

**Always include the batch dimension, even for single predictions!**

---

## Common TensorFlow Error Patterns

### 1. Shape Mismatches

**Symptoms:**
- `ValueError: ... is incompatible ...`
- `Shape ... and shape ... are incompatible`

**Debugging:**
```python
print("Layer input shape:", model.layers[0].input_shape)
print("Data shape:", X.shape)
model.summary()  # Review all layer shapes
```

### 2. Type Errors

**Symptoms:**
- `TypeError: ... must be a tensor`
- `Cannot convert ... to Tensor`

**Solution:**
- Convert to numpy arrays: `np.array(data)`
- Ensure correct dtypes: `.astype('float32')`

### 3. Label Mismatches

**Symptoms:**
- `Label value ... is outside valid range`
- Shape mismatches during training

**Solution:**
- Verify label encoding matches loss function
- Check label range (0 to num_classes-1)

### 4. Memory Errors

**Symptoms:**
- `ResourceExhaustedError: OOM`
- System becomes unresponsive

**Solutions:**
```python
# Reduce batch size
model.fit(X, y, batch_size=32)  # Instead of 128

# Use data generators
train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=32)

# Clear session
keras.backend.clear_session()
```

---

## Best Practices for Avoiding These Bugs

### 1. Always Print Shapes

```python
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
```

### 2. Use model.summary()

```python
model.summary()
# Review:
# - Input shapes
# - Output shapes
# - Parameter counts
# - Layer configurations
```

### 3. Test on Small Batch First

```python
# Test with 10 samples before full training
model.fit(X_train[:10], y_train[:10], epochs=1)
```

### 4. Implement Shape Assertions

```python
assert X_train.shape[1:] == (28, 28, 1), "Wrong input shape!"
assert len(np.unique(y_train)) == 10, "Should have 10 classes!"
assert y_train.min() >= 0 and y_train.max() < 10, "Labels out of range!"
```

### 5. Use Meaningful Variable Names

```python
# ❌ Bad
a, b = model.evaluate(X, y)

# ✅ Good
test_loss, test_accuracy = model.evaluate(X_test, y_test)
```

---

## Debugging Checklist

Before running your code, verify:

- [ ] Input data shape matches first layer expectations
- [ ] Data is normalized (if required)
- [ ] Labels are in correct format (integers or one-hot)
- [ ] Loss function matches label encoding
- [ ] Output layer size matches number of classes
- [ ] Batch dimension is included in all inputs
- [ ] Variable names correctly reflect their contents
- [ ] Predictions are converted from probabilities to classes (if needed)

---

## Summary

| Bug | Problem | Fix |
|-----|---------|-----|
| #1 | Missing channel dimension | Add channel: `.reshape(-1, 28, 28, 1)` |
| #2 | Label encoding mismatch | Match labels to loss function |
| #3 | Wrong output size | Use correct number of output neurons |
| #4 | Wrong loss function | Match loss to label format |
| #5 | Input shape mismatch | Ensure input matches layer expectations |
| #6 | Variable name confusion | Use descriptive names correctly |
| #7 | Wrong prediction format | Use `np.argmax()` for class labels |
| #8 | Missing batch dimension | Add batch dim: `np.expand_dims(x, 0)` |

---

## Additional Resources

- **TensorFlow Documentation**: https://www.tensorflow.org/api_docs
- **Keras Guide**: https://keras.io/guides/
- **Common Errors**: https://www.tensorflow.org/guide/effective_tf2
- **Stack Overflow**: Search for specific error messages
- **TensorFlow Forums**: https://discuss.tensorflow.org/

---

**Remember**: Debugging is a skill that improves with practice. These bugs represent common mistakes that even experienced developers make. Understanding them makes you a better ML engineer!
