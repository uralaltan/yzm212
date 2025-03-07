# Naive Bayes Binary Classification Project

## Table of Contents

- [Problem Description](#problem-description)
- [Dataset](#dataset)
- [Method](#method)
- [Running the Project](#running-the-project)
    - [Using a Virtual Environment (Preferred)](#using-a-virtual-environment-preferred-method)
    - [Direct Execution](#direct-execution-alternative-method)
- [Results](#results)
- [Discussion](#discussion)
    - [Performance Metrics Selection](#performance-metrics-selection)
    - [Implementation Comparison](#implementation-comparison)

This project implements a binary classification task using Naive Bayes algorithm. Two different implementations are
provided:

1. Scikit-learn's GaussianNB
2. Custom implementation of Gaussian Naive Bayes algorithm

## Problem Description

The task is to classify breast cancer tumors as malignant (0) or benign (1) using the Breast Cancer Wisconsin dataset.

## Dataset

The Breast Cancer Wisconsin dataset from scikit-learn is used for this project, which includes:

- 569 samples
- 30 features
- 2 classes (malignant and benign)

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe
characteristics of the cell nuclei present in the image. The features include measurements like:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal dimension

## Method

### Data Analysis and Preprocessing

- Explored class distribution to understand the dataset balance
- Checked feature distributions and correlations
- Identified and handled missing values (none were found in this dataset)
- Split the data into 70% training and 30% testing sets

### Implementation

1. **Scikit-learn GaussianNB**:
    - Used the built-in GaussianNB class
    - Measured training and prediction time
    - Evaluated performance using a confusion matrix and metrics

2. **Custom GaussianNB**:
    - Implemented Gaussian Naive Bayes from scratch
    - Applied the algorithm on the same train-test split
    - Measured training and prediction time
    - Evaluated performance using the same metrics

## Running the Project

### Using a Virtual Environment (Preferred Method)

Using a virtual environment is the preferred way to run this project as it isolates dependencies and avoids conflicts
with other projects.

#### On Windows:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python run_pipeline.py

# Deactivate the virtual environment when done
deactivate
```

#### On macOS and Linux:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python run_pipeline.py

# Deactivate the virtual environment when done
deactivate
```

### Direct Execution (Alternative Method)

If you prefer not to use a virtual environment, you can run the code directly:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python run_pipeline.py
```

## Results

The performance of both implementations is compared based on:

- Accuracy
- Precision
- Recall
- F1 Score
- Training time
- Prediction time

| Metric        | Scikit-learn | Custom Implementation |
|---------------|--------------|-----------------------|
| Accuracy      | ~0.95        | ~0.94                 |
| Precision     | ~0.96        | ~0.94                 |
| Recall        | ~0.95        | ~0.93                 |
| F1 Score      | ~0.95        | ~0.94                 |
| Training Time | <0.01s       | ~0.05s                |
| Testing Time  | <0.01s       | ~0.02s                |

The confusion matrices show similar classification performance with slight differences:

- Both models correctly classify most samples
- The custom implementation has slightly more false positives and false negatives

## Discussion

### Performance Metrics Selection

The choice of evaluation metrics depends on both the problem context and the class distribution:

1. **Accuracy**: Works well when classes are balanced (as in our dataset). However, for imbalanced datasets, accuracy
   can be misleading.

2. **Precision**: Important when the cost of false positives is high. In cancer diagnosis, false positives may lead to
   unnecessary procedures.

3. **Recall**: Critical when the cost of false negatives is high. In cancer diagnosis, missing a malignant tumor (false
   negative) can be life-threatening.

4. **F1 Score**: Balances precision and recall, useful when both false positives and false negatives have significant
   consequences.

For the breast cancer classification problem:

- Class distribution is reasonably balanced (212 malignant, 357 benign)
- False negatives (missing cancer) are more dangerous than false positives
- Therefore, while all metrics are reported, recall should be given more weight in the evaluation

### Implementation Comparison

1. **Performance**: The scikit-learn implementation slightly outperforms the custom implementation in all metrics,
   likely due to optimizations in the library.

2. **Execution Time**: The scikit-learn implementation is significantly faster in both training and testing, benefiting
   from compiled C/C++ code and optimizations.

3. **Trade-offs**: The custom implementation provides educational value and transparency at the cost of performance and
   speed.

Future improvements could include:

- Implementing feature selection to improve model performance
- Testing different Naive Bayes variants (Multinomial, Bernoulli) for comparison
- Exploring hyperparameter tuning to optimize performance