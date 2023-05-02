from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from typing import Any, List
import warnings


class IsBinary(BaseEstimator, TransformerMixin):
    def __init__(self, select_binary: bool = True):
        self.select_binary = select_binary
        self._selected_cols = None

    def fit(self, X, y=None):
        self._selected_cols = X.apply(lambda x: len(x.unique()))
        self._selected_cols = (
            list(self._selected_cols[self._selected_cols == 2].index)
            if self.select_binary
            else list(self._selected_cols[self._selected_cols > 2].index)
        )
        return self

    def transform(self, X, y=None):
        assert self.check_is_fitted(), "transformer is not fitted"
        return X[self._selected_cols]

    def check_is_fitted(self):
        return True if self._selected_cols is not None else False


class IsSpatial(BaseEstimator, TransformerMixin):
    def __init__(self, select_spatial: bool = True, spatial_cols: List[Any] = None):
        if spatial_cols is None:
            self.spatial_cols = ["Latitude", "Longitude"]
        else:
            self.spatial_cols = spatial_cols
        self.select_spatial = select_spatial

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(
            X, pd.DataFrame
        ), f"this transforemer works on pd.DataFrame, but insted got {type(X)}"

        if self.select_spatial:
            return X[self.spatial_cols]
        else:
            return X.drop(columns=self.spatial_cols, errors='ignore')


class SpatialPCA(BaseEstimator, TransformerMixin):
    def __init__(self, spatial_cols=None):
        if spatial_cols is None:
            self.spatial_cols = ["Latitude", "Longitude"]
        else:
            self.spatial_cols = spatial_cols

        self.pca = pca = PCA(n_components=1)

    def fit(self, X, y=None):
        self.pca.fit(X=X[self.spatial_cols])

        return self

    def transform(self, X, y=None):
        return pd.DataFrame(
            data=self.pca.transform(X[self.spatial_cols]), columns=["spatial_pca"]
        )


class KeepColsNames(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns
        return self

    def transform(self, X, y=None):
        if self.columns is not None and not isinstance(X, pd.DataFrame):
            return pd.DataFrame(data=X, columns=self.columns)
        else:
            return X


class JoinKeepColsNames(BaseEstimator, TransformerMixin):
    def __init__(self, transformers: List[Any]):
        self.columns = None
        self.transformers = transformers

    def fit(self, X, y=None):

        self.columns = []

        for tr in self.transformers:
            if tr.columns is not None:
                self.columns += tr.columns.to_list()
        if len(self.columns) == 0:
            self.columns = None
        return self

    def transform(self, X, y=None):
        if self.columns is not None and not isinstance(X, pd.DataFrame):
            return pd.DataFrame(data=X, columns=self.columns)
        else:
            return X


class PrintCols(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            print(X.columns.to_list())
        else:
            #rise warning not print
            warnings.warn("X is not a pd.DataFrame object")
        return X

