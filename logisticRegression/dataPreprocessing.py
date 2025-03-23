import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(file_path, test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)

    if df.isnull().sum().any():
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)

    return X_train_bias, X_test_bias, y_train, y_test, X_train, X_test


def display_data_info(df):
    print(f"Dataset shape: {df.shape}")
    print("\nFeature data types:")
    print(df.dtypes)

    print("\nClass distribution:")
    if "Outcome" in df.columns:
        print(df["Outcome"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSummary statistics:")
    print(df.describe())
