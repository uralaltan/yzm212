import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
import warnings

warnings.filterwarnings("ignore")


class NeuralNetwork:

    def __init__(
        self,
        layers,
        activation="relu",
        output_activation="sigmoid",
        task="classification",
        learning_rate=0.01,
    ):
        self.layers = layers
        self.activation = activation
        self.output_activation = output_activation
        self.task = task
        self.learning_rate = learning_rate
        self.weights, self.biases = [], []
        self.history = {"loss": [], "accuracy": []}
        self._initialize_parameters()

    def _initialize_parameters(self):
        np.random.seed(42)
        for fan_in, fan_out in zip(self.layers[:-1], self.layers[1:]):
            if self.task == "regression":
                w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))
            else:
                w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            self.weights.append(w)
            self.biases.append(np.zeros((1, fan_out)))

    @staticmethod
    def _activation(z, kind):
        if kind == "relu":
            return np.maximum(0, z)
        if kind == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        if kind == "tanh":
            return np.tanh(z)
        if kind == "softmax":
            e = np.exp(z - np.max(z, axis=1, keepdims=True))
            return e / np.sum(e, axis=1, keepdims=True)
        if kind == "linear":
            return z
        raise ValueError(f"Unknown activation: {kind}")

    @staticmethod
    def _activation_grad(z, kind):
        if kind == "relu":
            return (z > 0).astype(float)
        if kind == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            return s * (1 - s)
        if kind == "tanh":
            return 1 - np.tanh(z) ** 2
        if kind == "linear":
            return np.ones_like(z)
        raise ValueError(f"Unknown activation: {kind}")

    def _forward(self, X):
        activations, z_vals = [X], []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            z_vals.append(z)
            kind = self.output_activation if i == len(self.weights) - 1 else self.activation
            activations.append(self._activation(z, kind))
        return activations, z_vals

    def _backward(self, y, activations, z_vals):
        m = y.shape[0]
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        if self.task == "classification" and self.output_activation in {"sigmoid", "softmax"}:
            dz = activations[-1] - y
        else:
            dz = activations[-1] - y

        for i in reversed(range(len(self.weights))):
            grads_w[i] = activations[i].T @ dz / m
            grads_b[i] = np.sum(dz, axis=0, keepdims=True) / m
            if i:
                dz = (dz @ self.weights[i].T) * self._activation_grad(z_vals[i - 1], self.activation)
        return grads_w, grads_b

    def _update(self, grads_w, grads_b):
        max_norm = 5.0
        for i in range(len(self.weights)):
            grads_w[i] = np.clip(grads_w[i], -max_norm, max_norm)
            grads_b[i] = np.clip(grads_b[i], -max_norm, max_norm)
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def _loss(self, y_true, y_pred):
        if self.task == "classification":
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            if self.output_activation == "sigmoid":
                if y_true.ndim == 1:
                    y_true = y_true.reshape(-1, 1)
                return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            else:
                return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X, y, epochs=1000, batch_size=None, verbose=True):
        if batch_size is None:
            batch_size = X.shape[0]

        if self.task == "classification":
            if self.output_activation == "softmax":
                n_classes = len(np.unique(y))
                y_encoded = np.eye(n_classes)[y.astype(int)]
            else:
                y_encoded = y.reshape(-1, 1)
        else:
            y_encoded = y.reshape(-1, 1)

        best_loss, patience, stale = float("inf"), 50, 0

        for epoch in range(epochs):
            perm = np.random.permutation(X.shape[0])
            X_shuffled, y_shuffled, y_encoded_shuffled = X[perm], y[perm], y_encoded[perm]

            epoch_loss, epoch_acc, batches = 0, 0, 0
            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                Xb, yb, yb_encoded = X_shuffled[start:end], y_shuffled[start:end], y_encoded_shuffled[start:end]

                acts, zs = self._forward(Xb)
                loss = self._loss(yb_encoded, acts[-1])
                epoch_loss += loss

                if self.task == "classification":
                    if self.output_activation == "sigmoid":
                        preds = (acts[-1] > 0.5).astype(int).ravel()
                        truth = yb
                    else:
                        preds = np.argmax(acts[-1], axis=1)
                        truth = yb
                    epoch_acc += np.mean(preds == truth)

                grads_w, grads_b = self._backward(yb_encoded, acts, zs)
                self._update(grads_w, grads_b)
                batches += 1

            avg_loss = epoch_loss / batches
            self.history["loss"].append(avg_loss)
            if self.task == "classification":
                avg_acc = epoch_acc / batches
                self.history["accuracy"].append(avg_acc)

            if self.task == "regression":
                if avg_loss < best_loss:
                    best_loss, stale = avg_loss, 0
                else:
                    stale += 1
                    if stale >= patience:
                        if verbose:
                            print(f"Early stop at epoch {epoch}")
                        break

            if verbose and epoch % 50 == 0:
                if self.task == "classification":
                    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
                else:
                    print(f"Epoch {epoch}: Loss={avg_loss:.4f}")

    def predict(self, X):
        acts, _ = self._forward(X)
        if self.task == "classification":
            if self.output_activation == "sigmoid":
                return (acts[-1] > 0.5).astype(int).ravel()
            return np.argmax(acts[-1], axis=1)
        return acts[-1].ravel()

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba only for classification.")
        return self._forward(X)[0][-1]

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds) if self.task == "classification" else r2_score(y, preds)


def create_classification_dataset():
    return make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )


def create_regression_dataset():
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        effective_rank=5,
        noise=0.01,
        random_state=42,
    )
    return X, (y - y.mean()) / y.std()


def plot_training_history(hist, task="classification"):
    fig, ax = plt.subplots(1, 2 if task == "classification" else 1, figsize=(14, 4))

    if task == "classification":
        ax[0].plot(hist["loss"])
        ax[0].set_title("Loss")
        ax[1].plot(hist["accuracy"])
        ax[1].set_title("Accuracy")
        for a in ax:
            a.set_xlabel("Epoch")
            a.grid(True)
    else:
        ax.plot(hist["loss"])
        ax.set_title("Loss (MSE)")
        ax.set_xlabel("Epoch")
        ax.grid(True)

    plt.tight_layout()
    
    filename = f"images/training_history_{task}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {filename}")
    plt.show()


def plot_conf_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    filename = "images/confusion_matrix.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {filename}")
    plt.show()


def main():
    Xc, yc = create_classification_dataset()
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)
    scaler_c = StandardScaler().fit(Xc_train)
    Xc_train, Xc_test = scaler_c.transform(Xc_train), scaler_c.transform(Xc_test)

    clf = NeuralNetwork([20, 16, 8, 1], activation="relu", output_activation="sigmoid", task="classification")
    print("\nTraining classification network …")
    clf.fit(Xc_train, yc_train, epochs=1000, verbose=True)

    yc_pred = clf.predict(Xc_test)
    print(f"Test accuracy: {accuracy_score(yc_test, yc_pred):.4f}")
    plot_training_history(clf.history, task="classification")
    plot_conf_matrix(yc_test, yc_pred)

    Xr, yr = create_regression_dataset()
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    scaler_r = StandardScaler().fit(Xr_train)
    Xr_train, Xr_test = scaler_r.transform(Xr_train), scaler_r.transform(Xr_test)

    reg = NeuralNetwork([10, 12, 6, 1], activation="tanh", output_activation="linear", task="regression", learning_rate=0.01)
    print("\nTraining regression network …")
    reg.fit(Xr_train, yr_train, epochs=1000, verbose=True)

    yr_pred = reg.predict(Xr_test)
    print(f"Test MSE: {mean_squared_error(yr_test, yr_pred):.4f}")
    print(f"Test R² : {r2_score(yr_test, yr_pred):.4f}")
    plot_training_history(reg.history, task="regression")

    plt.figure(figsize=(6, 5))
    plt.scatter(yr_test, yr_pred, alpha=0.6)
    plt.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Regression: Predicted vs Actual")
    plt.grid(True)
    
    filename = "images/regression_predictions.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Regression predictions plot saved to {filename}")
    plt.show()


if __name__ == "__main__":
    main()
