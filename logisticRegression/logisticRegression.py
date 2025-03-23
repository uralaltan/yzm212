import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from dataPreprocessing import load_and_preprocess_data


class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.fit_time = None
        self.predict_time = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.weights)
        y_pred = self.sigmoid(z)
        cost = -1 * np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
        return cost

    def fit(self, X, y, verbose=False):
        start_time = time.time()

        m = X.shape[0]
        self.weights = np.random.randn(X.shape[1], 1) * 0.01

        costs = []
        for i in range(self.n_iterations):
            z = np.dot(X, self.weights)
            y_pred = self.sigmoid(z)

            cost = self.compute_cost(X, y)

            gradient = np.dot(X.T, (y_pred - y)) / m

            self.weights -= self.learning_rate * gradient

            if i % 100 == 0:
                costs.append(cost)
                if verbose:
                    print(f"Cost at iteration {i}: {cost}")

        self.fit_time = time.time() - start_time

        if verbose:
            plt.figure(figsize=(10, 6))
            plt.plot(range(0, self.n_iterations, 100), costs)
            plt.xlabel('Iterations')
            plt.ylabel('Cost')
            plt.title('Cost vs. Iterations')
            plt.show()

        return self

    def predict_proba(self, X):
        start_time = time.time()

        z = np.dot(X, self.weights)
        probabilities = self.sigmoid(z)

        self.predict_time = time.time() - start_time

        return probabilities

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)

        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        metrics = {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fit_time': self.fit_time,
            'predict_time': self.predict_time
        }

        return metrics

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()


if __name__ == "__main__":
    X_train_bias, X_test_bias, y_train, y_test, _, _ = load_and_preprocess_data('data/diabetes.csv')

    model = LogisticRegression(learning_rate=0.001, n_iterations=1000)
    model.fit(X_train_bias, y_train, verbose=True)

    metrics = model.evaluate(X_test_bias, y_test)

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Fit Time: {metrics['fit_time']:.4f} seconds")
    print(f"Predict Time: {metrics['predict_time']:.4f} seconds")

    model.plot_confusion_matrix(metrics['confusion_matrix'])
