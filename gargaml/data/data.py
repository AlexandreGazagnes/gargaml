import os, sys, logging, random, secrets
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

@dataclass
class XY:
    """ """

    X: pd.DataFrame
    y: pd.DataFrame
    name: str

    @property
    def y_log(self) -> float:
        return np.log1p(self.y)


@dataclass
class DataClass:
    """ """
    
    train: XY
    test: XY
    X: pd.DataFrame
    y: pd.Series
    df: pd.DataFrame

    @property
    def y_log(self) -> float:
        return np.log1p(self.y)

    @property
    def name(self) -> str:
        return self.train.name


def make_data(df: pd.DataFrame, y_name: str, data_name: str="data", test_size: float = 0.33, val_size:int=0.00,
              ):
    """ """

    # split X, y
    y = df[y_name]
    X = df.drop(columns=y_name)

    # test train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    train = XY(X_train, y_train, data_name)
    test = XY(X_test, y_test, data_name)

    # data
    data = DataClass(train, test, X, y, df)

    return data