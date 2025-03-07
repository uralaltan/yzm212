# compareModels.py
# Script to compare scikit-learn and custom implementations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure images directory exists
os.makedirs('images', exist_ok=True)

# Load results from both implementations
try:
    sklearn_results = np.load('data/sklearn_results.npy', allow_pickle=True).item()
    custom_results = np.load('data/custom_results.npy', allow_pickle=True).item()

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

    # Visualize the comparison of performance metrics
    plt.figure(figsize=(10, 6))
    comparison.iloc[2:].plot(kind='bar')
    plt.title('Performance Metrics Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/metricsComparison.png')
    plt.close()

    # Visualize the comparison of training and testing times
    plt.figure(figsize=(10, 5))
    comparison.iloc[:2].plot(kind='bar')
    plt.title('Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('images/timeComparison.png')
    plt.close()

    # Compare confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(sklearn_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix - Scikit-learn GaussianNB')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    sns.heatmap(custom_results['confusion_matrix'], annot=True, fmt='d', cmap='Reds', ax=axes[1])
    axes[1].set_title('Confusion Matrix - Custom GaussianNB')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('images/confusionMatricesComparison.png')
    plt.close()

    print("Comparison visualizations saved.")

except FileNotFoundError:
    print("Results files not found. Please run both implementations first.")