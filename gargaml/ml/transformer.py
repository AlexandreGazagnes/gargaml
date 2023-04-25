from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np 
import pandas as pd

import logging, os, sys


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Custom Lop1p Tranformer based on skew"""

    def __init__(self, skew_threshold=1):
        """init"""

        self.skew_threshold = skew_threshold
        self.columns = []

    def fit(self, X, y=None):
        """fit method"""

        tmp = X.skew().round(2)
        tmp = tmp[tmp > self.skew_threshold]
        self.columns = tmp.index

        return self

    def transform(self, X, y=None):
        """transform method"""

        # copy  X
        X_copy = X.copy()

        # cols
        np_cols = self.columns
        raw_cols = [i for i in X_copy.columns if i not in self.columns]

        # sep 2 df
        X_log1p = X_copy.loc[:, np_cols]
        X_not = X_copy.loc[:, raw_cols]

        # compute log
        X_log1p = np.log1p(X_log1p)

        # merge 2 df
        _X = pd.concat([X_log1p, X_not], axis=1)

        return _X



#####################################
#####################################
#####################################
#####################################
