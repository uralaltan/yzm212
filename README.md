# Machine Learning Homework Repository

This repository contains implementations of various machine learning algorithms for the YZM212 Machine Learning course.

## Project Structure

```
├── eigenVectorsValues/
│   ├── eigenVectorsValues.py
│   ├── README1.md
│   ├── README2.md
│   └── README3.md
├── forwardAndBackwardPropagation/
│   ├── neuralNetwork.py
│   ├── README.md
│   └── images/
│       ├── confusion_matrix.png
│       ├── regression_predictions.png
│       ├── training_history_classification.png
│       └── training_history_regression.png
├── linearRegression/
│   ├── linearRegressionWlse.py
│   ├── linearRegressionWsLearn.py
│   ├── lse_cost.txt
│   ├── sklearn_cost.txt
│   ├── README.md
│   └── images/
│       ├── data_visualization.png
│       ├── lse_regression_results.png
│       ├── lse_residuals.png
│       ├── model_comparison.png
│       ├── sklearn_regression_results.png
│       └── sklearn_residuals.png
├── logisticRegression/
│   ├── compareModels.py
│   ├── dataPreprocessing.py
│   ├── logisticRegression.py
│   ├── logisticRegressionScikitLearn.py
│   ├── modelComparison.csv
│   ├── README.md
│   ├── data/
│   │   └── diabetes.csv
│   └── images/
│       ├── confusionMatricesComparison.png
│       ├── metricsComparison.png
│       └── timeComparison.png
├── naiveBayes/
│   ├── compareModels.py
│   ├── modelComparison.csv
│   ├── naiveBayes.py
│   ├── naiveBayesScikitLearn.py
│   ├── runPipeline.py
│   ├── README.md
│   ├── data/
│   │   ├── custom_results.npy
│   │   ├── sklearn_results.npy
│   │   ├── X_test.npy
│   │   ├── X_train.npy
│   │   ├── y_test.npy
│   │   └── y_train.npy
│   └── images/
│       ├── classDistribution.png
│       ├── confusionMatricesComparison.png
│       ├── customConfusionMatrix.png
│       ├── featureDistributions.png
│       ├── metricsComparison.png
│       ├── sklearnConfusionMatrix.png
│       └── timeComparison.png
├── main.py
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

Implementation of Linear Regression using Least Square Estimation (LSE) from scratch and comparison with scikit-learn's implementation. The study includes model training, evaluation and visualization of results.

### 5. Forward and Backward Propagation Neural Network

From-scratch implementation of a feedforward neural network using forward and backward propagation algorithms. Supports both classification and regression tasks with multiple activation functions (ReLU, Sigmoid, Tanh, Linear, Softmax) and gradient-based optimization with mini-batch support.