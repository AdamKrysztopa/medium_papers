from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestRegressor

from sklearn.base import TransformerMixin, BaseEstimator    
from sklearn.feature_selection import mutual_info_regression, f_regression


def f_regression_scores(X: pd.DataFrame, y: pd.Series):
    return pd.Series(f_regression(X, y)[0], index=X.columns)


def pearson_corr(X):
    if isinstance(X, np.ndarray):
        return pd.DataFrame(data=X).corr().abs().clip(0.00001)
    elif isinstance(X, pd.DataFrame):
        return X.corr().abs().clip(0.00001)


def rf_features(X: pd.DataFrame, y: pd.Series, iterations: int = 100, metric: str = 'mean') -> pd.Series:
    """
    Calculate feature importances using Random Forest regression.

    Parameters:
    X (pd.DataFrame): Input features.
    y (pd.DataFrame): Target variable.
    iterations (int, optional): Number of iterations. Default is 100.
    metric (str, optional): The aggregation function to apply to the feature importance scores. Default is 'mean'.

    Returns:
    pd.Series: A pandas Series containing the feature importance scores.

    Raises:
    AssertionError: If the inputs are not valid.

    """

    # Assert input types
    assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame"
    assert isinstance(y, pd.Series), "y should be a pandas DataFrame"
    assert isinstance(iterations, int), "iterations should be an integer"
    assert isinstance(metric, str), "metric should be a string"

    # Assert input shapes
    assert X.shape[0] == y.shape[0], "X and y should have the same number of samples"

    data = []
    for ind in range(iterations):
        rf = RandomForestRegressor()
        rf.fit(X=X, y=y)
        data.append(pd.DataFrame(rf.feature_importances_).T)
    data = pd.concat(data)
    data.index = list(range(iterations))
    return data.aggregate([metric]).T.squeeze()


class MRMR(BaseEstimator, TransformerMixin):
    """
    A transformer that selects the top `n_features` using the Maximum Relevance Minimum Redundancy (MRMR).
    Implemented is general version that takes importance_function and penalty_function functions.
    Basing on those features are being selected.

    Parameters
    ----------
    n_features : int, default=5
    The number of features to select.
    importance_function : callable, default=mutual_info_regression
    The function used to calculate the scores of the features.
    This function takes two arguments: X, y and returns an array of scores for each feature.
    penalty_function : callable, default=pearson_corr
    The function used to calculate the correlation between features.
    This function takes a single argument: X and returns a pandas DataFrame of pairwise correlations.

    Attributes
    ----------
    n_features : int
    The number of features to select.
    importance_function : callable
    The function used to calculate the scores of the features.
    penalty_function : callable
    The function used to calculate the correlation between features.
    _selected_idx : list or None
    The indices of the selected features. None before calling `fit`.

    Methods
    -------
    fit(X, y)
    Fit the MRMR feature selector on the input data.
    transform(X)
    Transform the input data by selecting the top `n_features` features.
    get_params(deep=True)
    Gets parameters details
    set_params(**params)
    Sets parameters
    check_is_fitted()
    check if transformer is fitted

    Notes
    -----
    The Maximum Relevance Minimum Redundancy (MRMR) algorithm selects features based on their relevance to the
    target variable and their redundancy with respect to each other. The algorithm selects features iteratively,
    adding the feature that has the highest relevance with the target and the lowest redundancy with respect to
    the already selected features.

    Examples
    --------
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from mrmr_tansofmers import MRMR
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    mrmr = MRMR(n_features=3)
    pipeline = make_pipeline(mrmr, LinearRegression())
    scores = cross_val_score(pipeline, X, y, cv=5)
    print(scores.mean())
    """

    def __init__(self, **kwargs):

        kwargs = dict(kwargs)

        # todo add random state and pass it to the importance_function and penalty_function if they take randomn state
        for name, default_value in (
                # Param name, default param value, param conversion function
            ('n_features', 5),
            ('importance_function', mutual_info_regression),
            ('penalty_function', pearson_corr),
            ('random_state', None)
         ):
            setattr(self, name, kwargs.setdefault(name, default_value))
            del kwargs[name]

        if kwargs:
            raise ValueError('Invalid arguments: %s' % ', '.join(kwargs))

        self._selected_idx = None


    def fit(self, X: Union[pd.DataFrame, ArrayLike], y: Union[pd.Series, ArrayLike]):

        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        The training input samples.
        y : array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in
        regression).
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(data=X)

        if isinstance(y, np.ndarray):
            y = pd.Series(data=y)
        
        self._selected_features = X.columns.to_list()

        assert X.shape[0] == y.shape[0], "X and y should have the same number of samples"

        selected = []
        not_selected = list(X.columns)

        nom = self.importance_function(X, y)
        if not isinstance(nom, pd.Series):
            nom = pd.Series(nom, index=X.columns)

        denom = self.penalty_function(X)

        for i in range(self.n_features):
            score = nom.loc[not_selected] / denom.loc[not_selected, selected].mean(axis=1).fillna(0.001)
            best = score.index[score.argmax()]
            selected.append(best)
            not_selected.remove(best)

        self._selected_idx = selected


        return self

    def transform(self, X):

        assert self.check_is_fitted(), 'Model is not fitted'

        if isinstance(X, np.ndarray):
            return X[:, self._selected_idx]
        else:
            return X.loc[:, self._selected_idx]

    def get_params(self, deep=True):

        return {'n_features': self.n_features,
                'importance_function': self.importance_function,
                'penalty_function': self.penalty_function}

    def set_params(self, **params):

        for param, value in params.items():
            setattr(self, param, value)
        return self

    def check_is_fitted(self):

        return True if self._selected_idx is not None else False
    
    def get_feature_names(self):
        
        return self._selected_idx
        


class ModMRMR(MRMR):
    """
    Modification of the MRMR algorithm to reduce multicollinearity between selected features.

    This implementation of the ModMRMR algorithm selects features based on their mutual information with
    the target variable while also considering their correlation with each other.
    The algorithm iteratively selects features that maximize
    the score:

    The manin different between original and modified approach is on the penalty function,
    which is more ruthless in the case of the same and very close components.

    penalty = (1-max(similarity))/(mean(similarity)

    Parameters
    ----------
    n_features : int, optional (default=None)
    The number of features to select. If None, all features will be selected.

    Attributes
    ----------
    _selected_idx : list
    The indices of the selected features.

    Methods
    -------
    fit(X, y)
    Fit the ModMRMR feature selector on the input data and select the desired number of features.
    """

    def fit(self, X: Union[pd.DataFrame, ArrayLike], y: Union[pd.Series, ArrayLike]):

        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        The training input samples.
        y : array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in
        regression).
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(data=X)

        if isinstance(y, np.ndarray):
            y = pd.Series(data=y)

        assert X.shape[0] == y.shape[0], "X and y should have the same number of samples"



        selected = []
        not_selected = list(X.columns)

        nom = self.importance_function(X, y)
        if not isinstance(nom, pd.Series):
            nom = pd.Series(nom, index=X.columns)

        denom = self.penalty_function(X)

        for i in range(self.n_features):
            score = (nom.loc[not_selected] / denom.loc[not_selected, selected].mean(axis=1).fillna(0.001)) * (
                    1 - denom.loc[not_selected, selected].max(axis=1).fillna(0.0))

            best = score.index[score.argmax()]
            selected.append(best)
            not_selected.remove(best)

        self._selected_idx = selected

        return self
