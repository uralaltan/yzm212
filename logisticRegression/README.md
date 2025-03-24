# Logistic Regression Implementation

## Problem Definition

This project implements logistic regression for binary classification using both scikit-learn and a custom implementation. The goal is to compare the two implementations in terms of performance metrics and execution time.

## Data

The Pima Indians Diabetes dataset is used for this project, which contains medical data of female patients of Pima Indian heritage. The task is to predict whether a patient has diabetes based on several medical measurements.

Key features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: Class variable (0 or 1) indicating diabetes

The dataset contains 768 instances with 8 features (plus the outcome variable) and no missing values.

## Method

### Custom Logistic Regression

The custom implementation uses the maximum likelihood estimation (MLE) approach with gradient descent optimization:

- **Cost function**: Cross-entropy loss
- **Optimization method**: Batch gradient descent
- **Probability estimation**: Sigmoid function

Steps:

1. Initialize weights randomly.
2. Compute predicted probabilities using the sigmoid function.
3. Calculate the cost using cross-entropy loss.
4. Update weights using gradient descent.
5. Repeat until convergence or a set number of iterations.

### Scikit-learn Implementation

The scikit-learn implementation uses the following:

- `LogisticRegression` class with the `'lbfgs'` solver.
- Default L2 regularization.
- Maximum iterations set to 1000.

## Results

| Metric           | Custom Model | Scikit-learn Model |
|------------------|--------------|--------------------|
| Accuracy         | 0.636364     | 0.714286           |
| Precision        | 0.375000     | 0.608696           |
| Recall           | 0.055556     | 0.518519           |
| F1 Score         | 0.096774     | 0.560000           |
| Fit Time (s)     | 0.243206     | 0.010690           |
| Predict Time (s) | 0.000007     | 0.000428           |

Visualizations are saved in the project's `images` directory.

## Discussion

### Theoretical Comparison

The two implementations differ in several aspects:

1. **Optimization**: Scikit-learn uses the LBFGS optimizer, which is generally more efficient than traditional gradient descent. LBFGS approximates the Hessian matrix to determine optimal step directions, leading to faster convergence.
2. **Regularization**: Scikit-learn applies L2 regularization by default to prevent overfitting. The custom implementation does not include any form of regularization.
3. **Numerical Stability**: Scikit-learn handles edge cases and numerical instability more robustly compared to the custom implementation.
4. **Additional Features**: Scikit-learn offers advanced features such as scaling, feature selection, and cross-validation, which are not present in the custom approach.

### Performance Metrics Choice

Given the moderate class imbalance in the diabetes dataset (with approximately 35% positive cases), metrics like precision, recall, and F1-score provide more insight than accuracy alone. In the context of medical diagnosis, missing a positive case (i.e., a false negative) can have serious implications; hence, recall is a particularly important metric.

### Time Performance

Scikit-learn's implementation significantly outperforms the custom model in terms of execution time, both in fitting and prediction. This is attributed to its optimized C/C++ backend, advanced optimization algorithms, and efficient memory management. However, the custom implementation offers educational value by providing insights into the underlying mechanics of logistic regression.

