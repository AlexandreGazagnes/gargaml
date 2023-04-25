import os, sys, logging, random, secrets
from dataclasses import dataclass

from math import ceil

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class XY:
    """ """

    X: pd.DataFrame
    y: pd.DataFrame
    name: str = ""

    @property
    def y_log(self) -> float:
        return np.log1p(self.y)


@dataclass
class TVT:
    """ """

    train: pd.DataFrame
    test: pd.DataFrame
    val: pd.DataFrame
    name: str = ""


def DataClass():
    """ """

    def __init__(
        self,
        df: pd.DataFrame,
        y_name: str,
        data_name="",
        drop_cols=None,
        test_size=0.33,
        val_size=0.00,
        shuffle=True,
    ):
        """ """

        if not isinstance(df, pd.DataFrame):
            raise AttributeError()

        assert 0.33 >= test_size >= 0.01
        assert 0.33 >= val_size >= 0.0

        # split X, y
        self.shuffle = shuffle
        self.data_name = data_name
        self.drop_cols = [] if not drop_cols else drop_cols
        self.y_name = y_name
        self.test_size = test_size
        self.val_size = val_size
        self.train_size = 1 - test_size - val_size

        # _X, _y
        _df = _df.copy() if not shuffle else _df.sample(frac=1.0)
        self.X = _df.drop(columns=self.drop_cols + self.y_name)
        self.y = _df[y_name].copy()
        del _df

        # idxs = self._X.index
        self.size = len(self._X)
        self.n_train = ceil(self.size * self.train_size)
        self.n_test = ceil(self.size * self.test_size)
        self.n_val = ceil(self.size * self.val_size)

        while (self.n_train + self.n_test + self.n_val) != self.size:
            self.n_train -= 1

    def _start_stop(self, _type):
        """ """

        if _type == "train":
            _start = 0
            _stop = self.n_train

        elif _type == "test":
            if self.val_size:
                _start = self.n_train
                _stop = self.n_train + self.n_test
            else:
                _start = self.n_train
                _stop = None

        elif _type == "val":
            if self.val_size:
                _start = self.n_train + self.n_test
                _stop = None
            else:
                _start = _stop = -1
        else:
            raise AttributeError("sdfghj")

        return _start, _stop

    @property
    def train(self):
        """faaire data.train.X"""

        _start, _stop = self._start_stop("train")

        xy = XY(
            self.X.iloc[_start:_stop, :],
            self.y.iloc[_start:_stop],
            name=self.data_name,
        )

        return xy

    @property
    def test(self):
        """faaire data.test.X"""

        _start, _stop = self._start_stop("test")

        xy = XY(
            self.X.iloc[_start:_stop, :],
            self.y.iloc[_start:_stop],
            name=self.data_name,
        )

        return xy

    @property
    def val(self):
        """faaire data.val.X OU data.val.y"""

        _start, _stop = self._start_stop("val")

        xy = XY(
            self.X.iloc[_start:_stop, :],
            self.y.iloc[_start:_stop],
            name=self.data_name,
        )

        return xy

    @property
    def X(self):
        """faire data.X.train ou data.X.test"""

        _start, _stop = self._start_stop("train")
        X_train = self.X.iloc[: self.train_size]

        _start, _stop = self._start_stop("test")
        X_test = self.X.iloc[_start:_stop]

        _start, _stop = self._start_stop("val")
        X_val = self.X.iloc[_start:_stop]

        # tvt
        tvt = TVT(
            X_train,
            X_test,
            X_val,
            name=self.data_name,
        )

        return tvt

    @property
    def y(self):
        _start, _stop = self._start_stop("train")
        y_train = self.y.iloc[: self.train_size]

        _start, _stop = self._start_stop("test")
        y_test = self.y.iloc[_start:_stop]

        _start, _stop = self._start_stop("val")
        y_val = self.y.iloc[_start:_stop]

        # tvt
        tvt = TVT(
            y_train,
            y_test,
            y_val,
            name=self.data_name,
        )

        return tvt

    @property
    def y_log(self):
        pass

        _start, _stop = self._start_stop("train")
        y_log_train = self.y_log.iloc[: self.train_size]

        _start, _stop = self._start_stop("test")
        y_log_test = self.y_log.iloc[_start:_stop]

        _start, _stop = self._start_stop("val")
        y_log_val = self.y_log.iloc[_start:_stop]

        # tvt
        tvt = TVT(
            y_log_train,
            y_log_test,
            y_log_val,
            name=self.data_name,
        )

        return tvt
