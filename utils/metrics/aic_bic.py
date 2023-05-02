import pandas as pd
from numpy import log
import numpy as np


def linear_likelihood(y: pd.Series, y_hat: pd.Series) -> float:
    """
    Calculates the likelihood of for a any regression model. If likehood function is not given linear model is assumed.

    Args:
        y (pandas.Series): Observed values.
        y_hat (pandas.Series): Predicted values.

    Returns:
        float: The likelihood value.
    """
    assert len(y) == len(y_hat), "y and y_hat must have the same length."
    n = len(y)
    sse = ((y - y_hat) ** 2).sum()
    likelihood = (sse / n) ** (-n / 2)
    return likelihood

def linear_log_likelihood(y: pd.Series, y_hat: pd.Series) -> float:
    """
    Calculates the likelihood of for a any regression model. If likehood function is not given linear model is assumed.

    Args:
        y (pandas.Series): Observed values.
        y_hat (pandas.Series): Predicted values.

    Returns:
        float: The likelihood value.
    """
    assert len(y) == len(y_hat), "y and y_hat must have the same length."
    n = len(y)
    sse = ((y - y_hat) ** 2).sum()
    log_likelihood =  (-n / 2)*log(sse / n)
    return log_likelihood


def aic(
    y: pd.Series, y_hat: pd.Series, k: int, log_likehood_function: callable = None
) -> float:
    """
    Calculates the Akaike Information Criterion (AIC) for any regression model. If likehood function is not given linear model is assumed.

    Args:
        y (pandas.Series): Observed values.
        y_hat (pandas.Series): Predicted values.
        k (int): The number of model parameters.
        likehood_function (callable): The function used to calculate the likelihood.
                                      If None, the linear_likelihood function is used.

    Returns:
        float: The AIC value.
    """
    assert len(y) == len(y_hat), "y and y_hat must have the same length."
    assert isinstance(k, int) and k >= 0, "k must be a non-negative integer."
    if log_likehood_function is not None:
        assert callable(log_likehood_function), "likehood_function must be callable."
    else:
        log_likehood_function = linear_log_likelihood
    log_likehood_value = log_likehood_function(y, y_hat)
    k = float(k)
    return 2 * (k - log_likehood_value)


def bic(
    y: pd.Series, y_hat: pd.Series, k: int, log_likehood_function: callable = None
) -> float:
    """
    Calculates the Bayesian information criterion for a any regression model. If likehood function is not given linear model is assumed.

    Args:
        y (pandas.Series): Observed values.
        y_hat (pandas.Series): Predicted values.
        k (int): The number of model parameters.
        likehood_function (callable): The function used to calculate the likelihood.
                                      If None, the linear_likelihood function is used.

    Returns:
        float: The BIC value.
    """
    assert len(y) == len(y_hat), "y and y_hat must have the same length."
    assert isinstance(k, int) and k >= 0, "k must be a non-negative integer."
    if log_likehood_function is not None:
        assert callable(log_likehood_function), "likehood_function must be callable."
    else:
        log_likehood_function = linear_log_likelihood
    log_likehood_value = log_likehood_function(y, y_hat)
    k = float(k)
    n = len(y)
    k = float(k)
    return 2 * (k * log(n) - log_likehood_value)


def aicc(
    y: pd.Series, y_hat: pd.Series, k: int, log_likehood_function: callable = None
) -> float:
    """
    Calculates the corrected Akaike Information Criterion (AICc) for a linear model.

    Args:
        y (pandas.Series): Observed values.
        y_hat (pandas.Series): Predicted values.
        k (int): The number of model parameters.
        likehood_function (callable): The function used to calculate the likelihood.
                                      If None, the linear_likelihood function is used.

    Returns:
        float: The AICc value.
    """
    assert len(y) == len(y_hat), "y and y_hat must have the same length."
    assert isinstance(k, int) and k >= 0, "k must be a non-negative integer."
    n = len(y)
    aic_value = aic(y, y_hat, k, log_likehood_function)
    return aic_value + (2 * k * (k + 1)) / (n - k - 1)


def aic_scorer(estimator, X, y):
    """Calculate AIC score for a given model."""
    n, p = X.shape
    y_pred = estimator.predict(X)
    rss = np.sum((y - y_pred) ** 2)
    aic = n * np.log(rss / n) + 2 * p
    return aic


def bic_scorer(estimator, X, y):
    """Calculate BIC score for a given model."""
    n, p = X.shape
    y_pred = estimator.predict(X)
    rss = np.sum((y - y_pred) ** 2)
    bic = n * np.log(rss / n) + p * np.log(n)
    return bic


def aicc_scorer(estimator, X, y):
    """Calculate AICc score for a given model."""
    n, p = X.shape
    y_pred = estimator.predict(X)
    rss = np.sum((y - y_pred) ** 2)
    aic = n * np.log(rss / n) + 2 * p
    aicc = aic + 2 * p * (p + 1) / (n - p - 1)
    return aicc