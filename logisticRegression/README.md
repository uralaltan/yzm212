# Logistic Regression Implementation

## Problem Definition

This project implements logistic regression for binary classification using both scikit-learn and a custom
implementation. The goal is to compare the two implementations in terms of performance metrics and execution time.

## Data

The Pima Indians Diabetes dataset is used for this project, which contains medical data of female patients of Pima
Indian heritage. The task is to predict whether a patient has diabetes based on several medical measurements.

Key features:

- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age (years)
- Outcome: Class variable (0 or 1) indicating diabetes

The dataset contains 768 instances with 8 features and no missing values.

## Method

### Custom Logistic Regression

The custom implementation uses the maximum likelihood estimation (MLE) approach with gradient descent optimization:

- Cost function: Cross-entropy loss
- Optimization method: Batch gradient descent
- Sigmoid function for probability estimation

Steps:

1. Initialize weights randomly
2. Compute predicted probabilities using the sigmoid function
3. Calculate the cost using cross-entropy loss
4. Update weights using gradient descent
5. Repeat until convergence or maximum iterations

### Scikit-learn Implementation

The scikit-learn implementation uses the following:

- LogisticRegression class with 'lbfgs' solver
- Default L2 regularization
- Maximum iterations set to 1000

## Results

| Metric           | Custom Model | Scikit-learn Model |
|------------------|--------------|--------------------|
| Accuracy         | 0.XX         | 0.XX               |
| Precision        | 0.XX         | 0.XX               |
| Recall           | 0.XX         | 0.XX               |
| F1 Score         | 0.XX         | 0.XX               |
| Fit Time (s)     | 0.XX         | 0.XX               |
| Predict Time (s) | 0.XX         | 0.XX               |

(Note: The actual values will be filled in after running the code)

## Discussion

### Theoretical Comparison

The two implementations differ in several aspects:

1. **Optimization**: Scikit-learn uses the LBFGS optimizer which is more efficient than traditional gradient descent, as
   it approximates the Hessian matrix to find optimal step directions.
2. **Regularization**: Scikit-learn applies L2 regularization by default, which helps prevent overfitting. Our custom
   implementation doesn't include regularization.
3. **Numerical Stability**: Scikit-learn handles edge cases and numerical instability better than the custom
   implementation.
4. **Features**: Scikit-learn offers additional features like scaling, feature selection, and cross-validation.

### Performance Metrics Choice

The choice of performance metrics is influenced by:

1. **Class Imbalance**: If classes are imbalanced, accuracy may not be the best metric. Precision, recall, and F1-score
   are more informative in such cases.
2. **Application Context**: For medical diagnosis like diabetes detection, false negatives (missing a diabetic patient)
   might be more problematic than false positives. Therefore, recall might be a more important metric.

For the diabetes dataset, which has some class imbalance (approximately 35% positive cases), the F1 score provides a
good balance between precision and recall.

### Time Performance

Scikit-learn's implementation generally outperforms the custom implementation in terms of execution time due to:

1. Optimized C/C++ backend
2. Advanced optimization algorithms
3. Efficient memory management

However, the custom implementation provides educational value in understanding the underlying mathematics and algorithm.