from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import openml
from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
import pandas as pd


XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def load_heart_disease_data() -> Dataset:
    """Loads the Heart Disease dataset from UCI ML repository and handles NaN values."""
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Convert target variable to binary (0 or 1)
    y = np.where(y > 0, 1, 0)

    # Impute NaN values in features
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X)


    # Calculate the split index for an 80:20 split
    split_index = int(len(X_imputed) * 0.8)
    x_train, y_train = X_imputed[:split_index], y[:split_index]
    x_test, y_test = X_imputed[split_index:], y[split_index:]
    # Check feature consistency
    assert x_train.shape[1] == x_test.shape[1], "Mismatch in the number of features between train and test sets"
    return (x_train, y_train), (x_test, y_test)

def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros. Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch.
    """
    n_classes = 2  # Assuming binary classification for heart disease
    n_features = 13  # Number of features in heart disease dataset
    model.classes_ = np.array([0, 1])  # Update based on your dataset's class labels

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros(n_classes)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )