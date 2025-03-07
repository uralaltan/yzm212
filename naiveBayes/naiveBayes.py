# NaiveBayes.py
# Custom implementation of Gaussian Naive Bayes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time

# Custom implementation of Gaussian Naive Bayes
class CustomGaussianNB:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.means = None
        self.variances = None

    def fit(self, X, y):
        """
        Train the Gaussian Naive Bayes model

        Parameters:
        X (DataFrame or array-like): Training features
        y (Series or array-like): Target variable
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize parameters
        self.class_priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))

        # Calculate class priors and likelihood parameters
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[i] = len(X_c) / n_samples
            self.means[i, :] = X_c.mean(axis=0)
            self.variances[i, :] = X_c.var(axis=0) + 1e-9

    def _calculate_likelihood(self, X):
        """
        Calculate the Gaussian likelihood P(x|y) for all classes

        Parameters:
        X (DataFrame or array-like): Features for which to calculate likelihood

        Returns:
        likelihoods (array): Likelihood probabilities for each sample and class
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        likelihoods = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            # For each feature, calculate the Gaussian probability
            for j in range(n_features):
                # Gaussian probability density function
                exponent = -0.5 * ((X.iloc[:, j] - self.means[i, j]) ** 2) / self.variances[i, j]
                likelihood = np.exp(exponent) / np.sqrt(2 * np.pi * self.variances[i, j])

                # Get log-likelihood to avoid numerical underflow
                # Use log-sum technique: add logs instead of multiplying probabilities
                likelihoods[:, i] += np.log(likelihood + 1e-10)

        # Add log of class priors
        for i in range(n_classes):
            likelihoods[:, i] += np.log(self.class_priors[i])

        return likelihoods

    def predict(self, X):
        """
        Predict the class for each sample in X

        Parameters:
        X (DataFrame or array-like): Features to predict

        Returns:
        predictions (array): Predicted class for each sample
        """
        likelihoods = self._calculate_likelihood(X)
        # Return the class with the highest log-likelihood for each sample
        return self.classes[np.argmax(likelihoods, axis=1)]


# Load the pre-processed data and train-test split from the scikit-learn implementation
X_train = np.load('data/X_train.npy', allow_pickle=True)
X_test = np.load('data/X_test.npy', allow_pickle=True)
y_train = np.load('data/y_train.npy', allow_pickle=True)
y_test = np.load('data/y_test.npy', allow_pickle=True)

# Convert numpy arrays back to pandas DataFrames/Series
# This is necessary because our custom implementation uses DataFrame indexing
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

# Initialize the custom Gaussian Naive Bayes model
custom_model = CustomGaussianNB()

# Train the model and measure the training time
train_start_time = time.time()
custom_model.fit(X_train, y_train)
train_end_time = time.time()
train_time = train_end_time - train_start_time
print(f"\nCustom GaussianNB training time: {train_time:.6f} seconds")

# Test the model and measure the testing time
test_start_time = time.time()
y_pred = custom_model.predict(X_test)
test_end_time = time.time()
test_time = test_end_time - test_start_time
print(f"Custom GaussianNB testing time: {test_time:.6f} seconds")

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nCustom Model Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Create and visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix - Custom GaussianNB')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('images/customConfusionMatrix.png')
plt.close()

# Save results for comparison
custom_results = {
    'train_time': train_time,
    'test_time': test_time,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'confusion_matrix': cm
}

# Save the custom results
np.save('data/custom_results.npy', custom_results)

# Load scikit-learn results for comparison
sklearn_results = np.load('data/sklearn_results.npy', allow_pickle=True).item()

# Create comparison table
print("\nComparison of Scikit-learn vs Custom Implementation:")
comparison = pd.DataFrame({
    'Scikit-learn': [
        sklearn_results['train_time'],
        sklearn_results['test_time'],
        sklearn_results['accuracy'],
        sklearn_results['precision'],
        sklearn_results['recall'],
        sklearn_results['f1']
    ],
    'Custom': [
        custom_results['train_time'],
        custom_results['test_time'],
        custom_results['accuracy'],
        custom_results['precision'],
        custom_results['recall'],
        custom_results['f1']
    ]
}, index=['Training Time (s)', 'Testing Time (s)', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

print(comparison)

# Save the comparison table as CSV
comparison.to_csv('modelComparison.csv')

# Visualize the comparison
plt.figure(figsize=(10, 6))
comparison.iloc[2:].plot(kind='bar')
plt.title('Performance Metrics Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/metricsComparison.png')
plt.close()

# Compare training and testing times
plt.figure(figsize=(10, 5))
comparison.iloc[:2].plot(kind='bar')
plt.title('Time Comparison')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('images/timeComparison.png')
plt.close()

print("\nCustom Gaussian Naive Bayes implementation complete.")
print("Comparison with scikit-learn implementation completed and saved.")
