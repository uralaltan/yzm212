# Machine Learning Homework Repository

This repository contains implementations of various machine learning algorithms for the YZM212 Machine Learning course.

## Project Structure

```
├── 1.naiveBayes
│   ├── compareModels.py
│   ├── naiveBayesScikitLearn.py
│   ├── naiveBayes.py
│   ├── runPipeline.py
│   └── README.md
├── 2.logisticRegression
│   ├── logisticRegressionScikitLearn.py
│   ├── logisticRegression.py
│   ├── compareModels.py
│   ├── data
│   │   └── diabetes.csv
│   └── README.md
├── 3.eigenVectorsValues
│   ├── EigenVectorsValues.ipynb
│   ├── README1.md.
│   ├── README2.md.
│   └── README3.md
├── 4.linearRegression
│   ├── linearRegressionWLSE.py
│   ├── linearRegressionWSLearn.py
│   ├── images
│   │   ├── data_visualization.png
│   │   ├── lse_regression_results.png
│   │   ├── lse_residuals.png
│   │   ├── sklearn_regression_results.png
│   │   ├── sklearn_residuals.png
│   │   └── model_comparison.png
│   └── README.md
├── .gitignore
├── README.md
└── requirements.txt
```

## Requirements

See `requirements.txt` for a list of dependencies.

## Installation

```bash
pip install -r requirements.txt
```

## Contents

### 1. Naive Bayes Classifier

Implementation of Naive Bayes algorithm from scratch and comparison with scikit-learn's implementation.

### 2. Logistic Regression

Implementation of Logistic Regression algorithm from scratch and comparison with scikit-learn's implementation.

### 3. Eigenvectors and Eigenvalues

Exploration of eigenvectors and eigenvalues in the context of machine learning applications.

### 4. Linear Regression

Implementation of Linear Regression using Least Square Estimation (LSE) from scratch and comparison with scikit-learn's
implementation. The study includes model training, evaluation and visualization of results.