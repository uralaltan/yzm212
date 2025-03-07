# naiveBayesScikitLearn.py
# Implementation of Naive Bayes using scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
import time
import os

warnings.filterwarnings("ignore", category=FutureWarning)

# Create directories
os.makedirs('images', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load the dataset
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Dataset exploration
print("Dataset shape:", X.shape)
print("\nFeature names:")
print(X.columns.tolist())
print("\nDistribution of classes:")
print(y.value_counts())
print("\nExample data:")
print(X.head())

# Check for missing values
print("\nMissing values per feature:")
print(X.isnull().sum())

# Visualize class distribution
plt.figure(figsize=(8, 6))
y.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution')
plt.xlabel('Class (0: Malignant, 1: Benign)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig('images/classDistribution.png')
plt.close()

# Visualization of feature distributions
plt.figure(figsize=(12, 8))
sns.boxplot(data=X)
plt.title('Feature Distributions')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('images/featureDistributions.png')
plt.close()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Naive Bayes model
model = GaussianNB()

# Train the model and measure the training time
train_start_time = time.time()
model.fit(X_train, y_train)
train_end_time = time.time()
train_time = train_end_time - train_start_time
print(f"\nTraining time: {train_time:.6f} seconds")

# Test the model and measure the testing time
test_start_time = time.time()
y_pred = model.predict(X_test)
test_end_time = time.time()
test_time = test_end_time - test_start_time
print(f"Testing time: {test_time:.6f} seconds")

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Create and visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Scikit-learn GaussianNB')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('images/sklearnConfusionMatrix.png')
plt.close()

# Save results for comparison
sklearn_results = {
    'train_time': train_time,
    'test_time': test_time,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'confusion_matrix': cm
}

# Save the train-test split for use in the custom implementation
np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

# Save the results for comparison
np.save('data/sklearn_results.npy', sklearn_results)

print("\nSckit-learn GaussianNB implementation complete.")
print("Results saved for comparison with custom implementation.")
