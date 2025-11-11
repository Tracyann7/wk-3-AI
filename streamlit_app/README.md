# MNIST Digit Classifier - Web Application

A user-friendly web application for classifying handwritten digits using a CNN trained on the MNIST dataset.

## Features

- 📤 Upload custom digit images
- 🎲 Test on random MNIST samples
- 📊 Real-time predictions with confidence scores
- 📈 Interactive probability distribution visualization
- 🔍 Model architecture exploration
- 🎨 Clean and intuitive user interface

## Installation

### Prerequisites

- Python 3.8 or higher
- Trained MNIST model (from Task 2 notebook)

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the trained model exists at `../models/mnist_model.h5`
   - Run the Task 2 notebook first if you haven't already

## Usage

### Running the App

From the `streamlit_app` directory:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Application

#### 1. Digit Prediction Page

**Upload Image:**
- Click "Browse files" to upload a digit image
- Supported formats: PNG, JPG, JPEG
- The app will automatically preprocess and classify the digit

**Use Sample Images:**
- Click "Get Random Sample" to test on MNIST test images
- See the true label and compare with predictions
- Generate new samples as many times as you want

#### 2. Model Info Page

View detailed information about the CNN model:
- Complete architecture summary
- Layer-by-layer breakdown
- Parameter counts
- Expected performance metrics

#### 3. About Page

Learn more about:
- Project overview
- Technology stack
- MNIST dataset details
- Model architecture
- How to use the app

## Project Structure

```
streamlit_app/
├── app.py              # Main Streamlit application
├── utils.py            # Utility functions
├── requirements.txt    # App-specific dependencies
└── README.md           # This file
```

## Features Breakdown

### Image Preprocessing

The app automatically:
- Converts images to grayscale
- Resizes to 28×28 pixels
- Inverts colors (white digit on black background)
- Normalizes pixel values to [0, 1]

### Prediction Display

For each prediction, you get:
- **Predicted Digit**: The model's classification
- **Confidence Score**: Percentage confidence (0-100%)
- **Probability Distribution**: Bar chart showing probabilities for all digits
- **Detailed Probabilities**: Complete breakdown in table format

### Confidence Color Coding

- 🟢 **Green (90-100%)**: High confidence - very reliable
- 🟠 **Orange (70-89%)**: Medium confidence - generally reliable
- 🔴 **Red (0-69%)**: Low confidence - uncertain prediction

## Model Requirements

The app expects a trained Keras model saved at:
```
../models/mnist_model.h5
```

The model should:
- Accept input shape: (None, 28, 28, 1)
- Output 10 probabilities (one per digit class)
- Be compiled with categorical crossentropy loss

## Customization

### Changing Model Path

Edit the `load_model()` function in `app.py`:

```python
@st.cache_resource
def load_model():
    model = keras.models.load_model('path/to/your/model.h5')
    return model, None
```

### Modifying Styles

The app uses custom CSS in `app.py`. Modify the `st.markdown()` section at the top to change colors, fonts, and layouts.

### Adding Features

The `utils.py` file contains helper functions for:
- Image preprocessing
- Batch predictions
- Feature map visualization
- Model metrics calculation

Feel free to extend these or add your own!

## Troubleshooting

### Model Not Found

**Error**: "Error loading model: ..."

**Solution**:
1. Run the Task 2 notebook (`task2_mnist_deep_learning.ipynb`)
2. Ensure the model is saved to `../models/mnist_model.h5`
3. Check that the path in `app.py` is correct

### Poor Predictions

**Issue**: Model gives incorrect predictions

**Possible Causes**:
- Image quality is poor
- Digit is not centered
- Multiple digits in image
- Unusual handwriting style

**Solutions**:
- Use clearer images
- Ensure one digit per image
- Center the digit in the frame
- Try different samples

### Streamlit Issues

**Error**: "Streamlit command not found"

**Solution**:
```bash
pip install streamlit --upgrade
```

**Error**: "Port already in use"

**Solution**:
```bash
streamlit run app.py --server.port 8502
```

## Performance

- **Inference Time**: <10ms per image
- **Model Accuracy**: ~98% on MNIST test set
- **Supported Images**: Any size (auto-resized to 28×28)
- **Concurrent Users**: Streamlit handles multiple users efficiently

## Future Enhancements

Potential improvements:
- [ ] Drawing canvas for freehand input
- [ ] Batch prediction from multiple files
- [ ] Model comparison feature
- [ ] Prediction history tracking
- [ ] Export predictions to CSV
- [ ] Attention/saliency map visualization
- [ ] Mobile-responsive design improvements

## Technologies Used

- **Streamlit**: Web framework
- **TensorFlow/Keras**: Deep learning
- **OpenCV**: Image processing
- **Matplotlib/Seaborn**: Visualization
- **Pillow**: Image handling
- **NumPy/Pandas**: Data manipulation

## Contributing

This is an educational project. Suggestions and improvements are welcome!

## License

Created for educational purposes as part of an AI Tools assignment.

## Acknowledgments

- MNIST dataset: Yann LeCun et al.
- Streamlit team for the amazing framework
- TensorFlow/Keras for deep learning tools

---

**Made with ❤️ using Streamlit**

For questions or issues, please contact the project team.
