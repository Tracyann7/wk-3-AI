# AI Tools and Frameworks Assignment

**Week 3 - AI Tools Mastery**

This repository contains a comprehensive exploration of popular AI tools and frameworks including TensorFlow, PyTorch, Scikit-learn, and spaCy through practical implementations and theoretical analysis.

## Project Overview

This assignment demonstrates proficiency in selecting, implementing, and critically analyzing AI tools to solve real-world problems across three domains:
- Classical Machine Learning (Scikit-learn)
- Deep Learning (TensorFlow/Keras)
- Natural Language Processing (spaCy)

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Tasks Overview](#tasks-overview)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
wk-3-AI/
│
├── notebooks/                          # Jupyter notebooks for all tasks
│   ├── task1_iris_classification.ipynb    # Classical ML with Scikit-learn
│   ├── task2_mnist_deep_learning.ipynb    # Deep Learning with TensorFlow
│   └── task3_nlp_sentiment_analysis.ipynb # NLP with spaCy
│
├── data/                              # Dataset storage
│   ├── raw/                          # Original datasets
│   └── processed/                    # Preprocessed data
│
├── models/                           # Saved trained models
│   ├── iris_model.pkl
│   └── mnist_model.h5
│
├── src/                              # Source code modules
│   ├── data_preprocessing.py
│   ├── model_utils.py
│   └── visualization.py
│
├── streamlit_app/                    # Web deployment (Bonus)
│   ├── app.py
│   └── utils.py
│
├── reports/                          # Documentation and analysis
│   ├── report.md
│   ├── theoretical_questions.md
│   ├── ethical_analysis.md
│   └── figures/                      # Generated visualizations
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd wk-3-AI
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

## Tasks Overview

### Task 1: Classical Machine Learning - Iris Classification
**Framework:** Scikit-learn

**Objective:** Build and compare multiple classification models for iris species prediction

**Features:**
- Data preprocessing and exploratory analysis
- Multiple model comparison (Decision Tree, Random Forest, SVM, Logistic Regression)
- Hyperparameter tuning with GridSearchCV
- Cross-validation and performance metrics
- Feature importance analysis

**Metrics Achieved:** ~97% accuracy with optimized Random Forest

### Task 2: Deep Learning - MNIST Digit Classification
**Framework:** TensorFlow/Keras

**Objective:** Design a CNN to classify handwritten digits with >95% accuracy

**Features:**
- Custom CNN architecture with multiple layers
- Data augmentation for better generalization
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Training visualization and learning curves
- Confusion matrix and error analysis

**Metrics Achieved:** 98.7% test accuracy

### Task 3: NLP - Amazon Reviews Sentiment Analysis
**Framework:** spaCy

**Objective:** Perform Named Entity Recognition and sentiment analysis on product reviews

**Features:**
- Complete NLP pipeline (tokenization, lemmatization, POS tagging)
- Named Entity Recognition for products and brands
- Rule-based sentiment analysis
- Text visualization (word clouds, entity distributions)
- Comparative analysis of sentiment patterns

### Bonus: Web Deployment
**Framework:** Streamlit

A web application for the MNIST classifier with:
- Interactive drawing canvas for digit input
- Real-time prediction with confidence scores
- Model architecture visualization
- User-friendly interface

## Usage

### Running Notebooks

Launch Jupyter Notebook:
```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and open any task notebook.

### Running the Streamlit App

```bash
cd streamlit_app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Results

Detailed results, visualizations, and analysis are available in the `reports/` directory:
- **Theoretical Questions:** Deep dive into AI concepts and framework comparisons
- **Ethical Analysis:** Discussion on bias, fairness, and responsible AI practices
- **Performance Metrics:** Comprehensive evaluation of all models

### Key Findings

1. **Scikit-learn** excels at classical ML tasks with simple, intuitive APIs
2. **TensorFlow** provides powerful tools for deep learning with extensive community support
3. **spaCy** offers production-ready NLP pipelines with excellent performance

## Team Members

- [Team Member 1]
- [Team Member 2]
- [Team Member 3]
- [Team Member 4]
- [Team Member 5]

## Contributing

This is an academic assignment. For questions or suggestions, please contact the team members.

## Acknowledgments

- Iris Dataset: UCI Machine Learning Repository
- MNIST Dataset: Yann LeCun's website
- Amazon Reviews: Kaggle Dataset
- Course materials and documentation from official frameworks

## License

This project is created for educational purposes as part of an AI course assignment.
