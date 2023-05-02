import numpy as np
import pandas as pd
from typing import Any, List

from sklearn.model_selection import _BaseKFold
from sklearn.utils.validation import indexable, _num_samples
"""
https://nander.cc/writing-custom-cross-validation-methods-grid-search
"""

class TimeSeriesSlidingWindow(_BaseKFold):
    
    def __init__(self,train_size: int, n_splits=5, *, min_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.min_train_size = min_train_size
        
        
    def split(self, X, y=None, groups=None):
        
        # method splits the data into training and test sets
        # assuming sliding window
        # if min_train_size is not None, then the first split will be treated as a expanding window
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap
        test_size = (
            self.test_size if self.test_size is not None else n_samples // n_folds
        )
        
                # Make sure we have enough samples for the given split parameters
        if n_folds > n_samples:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater"
                f" than the number of samples={n_samples}."
            )
        if n_samples - gap - (test_size * n_splits) <= 0:
            raise ValueError(
                f"Too many splits={n_splits} for number of samples"
                f"={n_samples} with test_size={test_size} and gap={gap}."
            )

        indices = np.arange(n_samples)
        if self.min_train_size is not None:
            test_starts = range(max(n_samples - n_splits * test_size + self.gap,self.min_train_size+self.gap), n_samples, test_size)
        else:
            test_starts = range(n_samples - n_splits * test_size, n_samples, test_size)
        
        
        for test_start in test_starts:
            train_end = test_start - gap
            if self.min_train_size:
                yield (
                    indices[:train_end],
                    indices[test_start : test_start + test_size],
                )
            else:
                yield (
                    indices[:train_end],
                    indices[test_start : test_start + test_size],
                )
