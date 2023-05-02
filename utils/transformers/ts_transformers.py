
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class TimeSeriesDiff(BaseEstimator, TransformerMixin):
    """
    To fullfill the requirements of sklearn transformers,
    this transformer takes a pandas series as input and returns a numpy array as output 
    and keeps the length of the input series and output array the same.
    
    If needed to reduce the length of the series, limit it using first_index_to_use attribute.
    """
    def __init__(self, lag: int = 1):
        self.lag = lag
        self.leading_zeros_index = None
        self.first_values = None
        self.indexes = None
        self.first_index_to_use = None
        
    def fit(self, X, y=None):
        
        if isinstance(X, pd.Series):
            self.leading_zeros_index = X.index.get_loc(X.fillna(0).ne(0).idxmax())
            self.first_values = X[:self.leading_zeros_index+self.lag]
            self.indexes = X.index
        else: 
            self.leading_zeros_index = np.argmax(X != 0)
            self.first_values = X[:self.leading_zeros_index+self.lag]
            self.first_values = pd.Series(self.first_values)
        
        self.first_index_to_use = len(self.first_values)
        
        return self
        
    def transform(self, X, y=None):
        
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
            X_tr = pd.concat([X[:self.leading_zeros_index],X[self.leading_zeros_index:].diff(self.lag).fillna(0)]).to_numpy()
        else:
            X_tr = pd.concat([X[:self.leading_zeros_index],X[self.leading_zeros_index:].diff(self.lag).fillna(0)])
            X_tr.index = self.indexes
        return X_tr
        
    def inverse_transform(self, X, y=None):
        
        if isinstance(X, pd.Series):
            X = X.copy()
            is_pandas = True
        else:
            X = pd.Series(X)
            is_pandas = False

        X = pd.concat([self.first_values[-self.lag:], X[self.leading_zeros_index+self.lag:]],ignore_index=True)
        X_sumed = pd.Series(index= X.index)
        
        for lag in range(self.lag):
            range_ = range(0+lag,X.shape[0], self.lag)
            X_sumed.iloc[range_] = X.iloc[range_].cumsum()
        
        X_sumed = pd.concat([self.first_values[:self.leading_zeros_index], X_sumed],ignore_index=True)
        # X_sumed.index = self.indexes
        
        if is_pandas:
            return X_sumed
        else:
            return X_sumed.to_numpy()
        

class StatsforecastFormat(BaseEstimator, TransformerMixin):
    
    def __init__(self, unique_id, period = None,start_date = None) -> None:
        super().__init__()
        self.period = period
        self.start_date = start_date
        self.unique_id = unique_id
        
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        if not X.index.inferred_type == 'datetime64':
            assert self.period is not None, "period is not defined"
            assert self.start_date is not None, "start_date is not defined"
            X.index = pd.date_range(start=self.start_date, periods=len(X), freq=self.period)
            X.index.name = 'ds'
        else:
            X.index.name = 'ds'
        X.name = 'y'
        X = X.reset_index()
        X['unique_id'] = self.unique_id
        
        return X[['unique_id','ds','y']]
        