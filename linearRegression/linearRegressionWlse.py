import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

if not os.path.exists('images'):
    os.makedirs('images')

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print("\nFirst few rows of the dataset:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

X = df[['lstat']].values
y = df['medv'].values

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5)
plt.title('Median Home Value vs % Lower Status Population')
plt.xlabel('Lower Status Population (%)')
plt.ylabel('Median Home Value ($1000s)')
plt.grid(True)
plt.savefig('images/data_visualization.png')
plt.show()

np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
X_train = X[indices[:train_size]]
y_train = y[indices[:train_size]]
X_test = X[indices[train_size:]]
y_test = y[indices[train_size:]]

print(f"Training data size: {X_train.shape[0]}")
print(f"Testing data size: {X_test.shape[0]}")


def least_squares_estimation(X, y):
    X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))

    X_transpose = X_with_intercept.T
    X_transpose_X = np.dot(X_transpose, X_with_intercept)
    X_transpose_X_inv = np.linalg.inv(X_transpose_X)
    X_transpose_y = np.dot(X_transpose, y)
    beta = np.dot(X_transpose_X_inv, X_transpose_y)

    return beta


beta = least_squares_estimation(X_train, y_train)
intercept, slope = beta

print(f"Intercept (β₀): {intercept:.4f}")
print(f"Slope (β₁): {slope:.4f}")


def predict(X, beta):
    X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
    return np.dot(X_with_intercept, beta)


y_train_pred = predict(X_train, beta)

y_test_pred = predict(X_test, beta)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)


train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r_squared(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r_squared(y_test, y_test_pred)

print("\nTraining Results:")
print(f"Mean Squared Error (MSE): {train_mse:.4f}")
print(f"R² Score: {train_r2:.4f}")

print("\nTest Results:")
print(f"Mean Squared Error (MSE): {test_mse:.4f}")
print(f"R² Score: {test_r2:.4f}")

X_range = np.linspace(min(X.flatten()), max(X.flatten()), 100).reshape(-1, 1)
y_range_pred = predict(X_range, beta)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.5, label='Training Data')
plt.plot(X_range, y_range_pred, color='red', label='Regression Line')
plt.title('Linear Regression - Training Data')
plt.xlabel('Lower Status Population (%)')
plt.ylabel('Median Home Value ($1000s)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.5, label='Test Data')
plt.plot(X_range, y_range_pred, color='red', label='Regression Line')
plt.title('Linear Regression - Test Data')
plt.xlabel('Lower Status Population (%)')
plt.ylabel('Median Home Value ($1000s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('images/lse_regression_results.png')
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, y_train_pred - y_train)
plt.axhline(y=0, color='red', linestyle='-')
plt.title('Residuals - Training Data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, y_test_pred - y_test)
plt.axhline(y=0, color='red', linestyle='-')
plt.title('Residuals - Test Data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)

plt.tight_layout()
plt.savefig('images/lse_residuals.png')
plt.show()

with open('lse_cost.txt', 'w') as f:
    f.write(f"LSE Train MSE: {train_mse}\n")
    f.write(f"LSE Test MSE: {test_mse}\n")
    f.write(f"LSE Train R²: {train_r2}\n")
    f.write(f"LSE Test R²: {test_r2}\n")
