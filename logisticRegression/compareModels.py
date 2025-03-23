import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import time
import os

from dataPreprocessing import load_and_preprocess_data, display_data_info
from logisticRegression import LogisticRegression as CustomLogisticRegression


def compare_models(data_path='data/diabetes.csv'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'images')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(data_path)
    print("Dataset Information:")
    display_data_info(df)

    X_train_bias, X_test_bias, y_train, y_test, X_train, X_test = load_and_preprocess_data(data_path)

    y_train_flat = y_train.ravel()
    y_test_flat = y_test.ravel()

    print("\n--- Custom Logistic Regression ---")
    custom_model = CustomLogisticRegression(learning_rate=0.001, n_iterations=10000)
    custom_model.fit(X_train_bias, y_train)
    custom_metrics = custom_model.evaluate(X_test_bias, y_test)

    print("\n--- Scikit-learn Logistic Regression ---")
    sklearn_model = LogisticRegression(solver='lbfgs', max_iter=10000, random_state=42)

    start_time = time.time()
    sklearn_model.fit(X_train, y_train_flat)
    sklearn_fit_time = time.time() - start_time

    start_time = time.time()
    sklearn_preds = sklearn_model.predict(X_test)
    sklearn_predict_time = time.time() - start_time

    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    sklearn_cm = confusion_matrix(y_test_flat, sklearn_preds)
    sklearn_metrics = {
        'confusion_matrix': sklearn_cm,
        'accuracy': accuracy_score(y_test_flat, sklearn_preds),
        'precision': precision_score(y_test_flat, sklearn_preds),
        'recall': recall_score(y_test_flat, sklearn_preds),
        'f1_score': f1_score(y_test_flat, sklearn_preds),
        'fit_time': sklearn_fit_time,
        'predict_time': sklearn_predict_time
    }

    print("\n--- Performance Comparison ---")

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Fit Time (s)', 'Predict Time (s)'],
        'Custom Model': [
            custom_metrics['accuracy'],
            custom_metrics['precision'],
            custom_metrics['recall'],
            custom_metrics['f1_score'],
            custom_metrics['fit_time'],
            custom_metrics['predict_time']
        ],
        'Scikit-learn Model': [
            sklearn_metrics['accuracy'],
            sklearn_metrics['precision'],
            sklearn_metrics['recall'],
            sklearn_metrics['f1_score'],
            sklearn_metrics['fit_time'],
            sklearn_metrics['predict_time']
        ]
    })

    print(metrics_df)

    metrics_df.to_csv(os.path.join(current_dir, 'modelComparison.csv'), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(custom_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
    ax1.set_title('Custom Model - Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    sns.heatmap(sklearn_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2)
    ax2.set_title('Scikit-learn Model - Confusion Matrix')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusionMatricesComparison.png'))
    plt.close()

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    custom_values = [custom_metrics[metric.lower().replace(' ', '_')] for metric in metric_names]
    sklearn_values = [sklearn_metrics[metric.lower().replace(' ', '_')] for metric in metric_names]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(metric_names))
    width = 0.35

    plt.bar(x - width / 2, custom_values, width, label='Custom Model')
    plt.bar(x + width / 2, sklearn_values, width, label='Scikit-learn Model')

    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metric_names)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'metricsComparison.png'))
    plt.close()

    time_metrics = ['Fit Time (s)', 'Predict Time (s)']
    custom_times = [custom_metrics['fit_time'], custom_metrics['predict_time']]
    sklearn_times = [sklearn_metrics['fit_time'], sklearn_metrics['predict_time']]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(time_metrics))
    width = 0.35

    plt.bar(x - width / 2, custom_times, width, label='Custom Model')
    plt.bar(x + width / 2, sklearn_times, width, label='Scikit-learn Model')

    plt.xlabel('Metrics')
    plt.ylabel('Time (seconds)')
    plt.title('Time Performance Comparison')
    plt.xticks(x, time_metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'timeComparison.png'))
    plt.close()

    print(f"\nVisualizations saved in: {output_dir}")
    return metrics_df


if __name__ == "__main__":
    compare_models()
