import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from dataPreprocessing import load_and_preprocess_data


def evaluate_sklearn_model(model, X_test, y_test):
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fit_time': model.fit_time,
        'predict_time': predict_time
    }

    return metrics


def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


if __name__ == "__main__":
    X_train_bias, X_test_bias, y_train, y_test, X_train, X_test = load_and_preprocess_data('data/diabetes.csv')

    y_train_flat = y_train.ravel()
    y_test_flat = y_test.ravel()

    sklearn_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)

    start_time = time.time()
    sklearn_model.fit(X_train, y_train_flat)
    sklearn_model.fit_time = time.time() - start_time

    metrics = evaluate_sklearn_model(sklearn_model, X_test, y_test_flat)

    print("\nScikit-learn Logistic Regression Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Fit Time: {metrics['fit_time']:.4f} seconds")
    print(f"Predict Time: {metrics['predict_time']:.4f} seconds")

    plot_confusion_matrix(metrics['confusion_matrix'])

    print("\nModel Coefficients:")
    feature_names = ['Bias'] + list(X_train.columns)
    coefficients = np.concatenate([[sklearn_model.intercept_[0]], sklearn_model.coef_[0]])
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    print(coef_df)
