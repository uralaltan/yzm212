import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os

if not os.path.exists('images'):
    os.makedirs('images')

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print("\nFirst few rows of the dataset:")
print(df.head())

X = df[['lstat']].values
y = df['medv'].values

np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data size: {X_train.shape[0]}")
print(f"Testing data size: {X_test.shape[0]}")

model = LinearRegression()
model.fit(X_train, y_train)

intercept = model.intercept_
slope = model.coef_[0]

print(f"Intercept (β₀): {intercept:.4f}")
print(f"Slope (β₁): {slope:.4f}")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTraining Results:")
print(f"Mean Squared Error (MSE): {train_mse:.4f}")
print(f"R² Score: {train_r2:.4f}")

print("\nTest Results:")
print(f"Mean Squared Error (MSE): {test_mse:.4f}")
print(f"R² Score: {test_r2:.4f}")

X_range = np.linspace(min(X.flatten()), max(X.flatten()), 100).reshape(-1, 1)
y_range_pred = model.predict(X_range)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.5, label='Training Data')
plt.plot(X_range, y_range_pred, color='red', label='Regression Line')
plt.title('Linear Regression with Scikit-Learn - Training Data')
plt.xlabel('Lower Status Population (%)')
plt.ylabel('Median Home Value ($1000s)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.5, label='Test Data')
plt.plot(X_range, y_range_pred, color='red', label='Regression Line')
plt.title('Linear Regression with Scikit-Learn - Test Data')
plt.xlabel('Lower Status Population (%)')
plt.ylabel('Median Home Value ($1000s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('images/sklearn_regression_results.png')
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
plt.savefig('images/sklearn_residuals.png')
plt.show()

try:
    with open('lse_cost.txt', 'r') as f:
        lse_costs = f.readlines()

    lse_train_mse = float(lse_costs[0].split(": ")[1])
    lse_test_mse = float(lse_costs[1].split(": ")[1])
    lse_train_r2 = float(lse_costs[2].split(": ")[1])
    lse_test_r2 = float(lse_costs[3].split(": ")[1])

    comparison_data = {
        'Model': ['LSE', 'Scikit-Learn'],
        'Train MSE': [lse_train_mse, train_mse],
        'Test MSE': [lse_test_mse, test_mse],
        'Train R²': [lse_train_r2, train_r2],
        'Test R²': [lse_test_r2, test_r2]
    }

    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    models = comparison_df['Model']
    train_mse_values = comparison_df['Train MSE']
    test_mse_values = comparison_df['Test MSE']

    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width / 2, train_mse_values, width, label='Train MSE')
    plt.bar(x + width / 2, test_mse_values, width, label='Test MSE')

    plt.xlabel('Model')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, axis='y')

    plt.subplot(1, 2, 2)
    train_r2_values = comparison_df['Train R²']
    test_r2_values = comparison_df['Test R²']

    plt.bar(x - width / 2, train_r2_values, width, label='Train R²')
    plt.bar(x + width / 2, test_r2_values, width, label='Test R²')

    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.title('R² Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('images/model_comparison.png')
    plt.show()

except FileNotFoundError:
    print("\nLSE cost file not found. Run linearRegressionWLSE.py first to make comparison.")

with open('sklearn_cost.txt', 'w') as f:
    f.write(f"Scikit-Learn Train MSE: {train_mse}\n")
    f.write(f"Scikit-Learn Test MSE: {test_mse}\n")
    f.write(f"Scikit-Learn Train R²: {train_r2}\n")
    f.write(f"Scikit-Learn Test R²: {test_r2}\n")
