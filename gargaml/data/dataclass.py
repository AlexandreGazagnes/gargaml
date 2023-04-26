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
    y: pd.Series
    name: str = ""

    @property
    def y_log(self) -> float:
        return np.log1p(self.y)


@dataclass
class XX(pd.DataFrame):
    """ """

    def __init__(
        self,
        _X: pd.DataFrame,
        train_tuple: list,
        test_tuple: list = [-1, -1],
        val_tuple: list = [-1, -1],
        name: str = "",
    ):
        super().__init__(_X)

        self._train_tuple = train_tuple
        self._test_tuple = test_tuple
        self._val_tuple = val_tuple

        self.train = self.iloc[self._train_tuple[0] : self._train_tuple[1], :]
        self.test = self.iloc[self._test_tuple[0] : self._test_tuple[1], :]
        self.val = self.iloc[self._val_tuple[0] : self._val_tuple[1], :]
        self.name = name


@dataclass
class YY(pd.Series):
    """ """

    def __init__(
        self,
        _y: pd.Series,
        train_tuple: list,
        test_tuple: list = [-1, -1],
        val_tuple: list = [-1, -1],
        name: str = "",
    ):
        super().__init__(_y)

        self._train_tuple = train_tuple
        self._test_tuple = test_tuple
        self._val_tuple = val_tuple

        self.train = self.iloc[self._train_tuple[0] : self._train_tuple[1]]
        self.test = self.iloc[self._test_tuple[0] : self._test_tuple[1]]
        self.val = self.iloc[self._val_tuple[0] : self._val_tuple[1]]
        self.name = name


class DataClass:
    """
    Ml Data Class : X/y and Train/Test/Val spliter
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y_name: str,
        data_name="",
        drop_cols=None,
        test_size=0.33,
        val_size=0.00,
        drop_target_nan = True,
        shuffle=True,
    ):
        """
        df: pd.DataFrame,
        y_name: str,
        data_name="",
        drop_cols=None,
        test_size=0.33,
        val_size=0.00,
        shuffle=True,
        """

        if not isinstance(df, pd.DataFrame):
            raise AttributeError()

        assert 0.33 >= test_size >= 0.01
        assert 0.33 >= val_size >= 0.0

        ############################


        # TO DOO DROP y Nan


        ############################

        # split X, y
        self.shuffle = shuffle
        self.data_name = data_name
        self.drop_cols = [] if not drop_cols else drop_cols
        self.y_name = y_name
        self.test_size = test_size
        self.val_size = val_size
        self.drop_target_nan = drop_target_nan
        self.train_size = round(1 - test_size - val_size, 2)

        # _X, _y
        _df = df.copy() if not shuffle else df.sample(frac=1.0)
        self._X = _df.drop(columns=self.drop_cols + [self.y_name]).copy()
        self._y = _df[y_name].copy()
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
            self._X.iloc[_start:_stop, :],
            self._y.iloc[_start:_stop],
            name=self.data_name,
        )

        return xy

    @property
    def test(self):
        """faaire data.test.X"""

        _start, _stop = self._start_stop("test")

        xy = XY(
            self._X.iloc[_start:_stop, :],
            self._y.iloc[_start:_stop],
            name=self.data_name,
        )

        return xy

    @property
    def val(self):
        """faaire data.val.X OU data.val.y"""

        _start, _stop = self._start_stop("val")

        xy = XY(
            self._X.iloc[_start:_stop, :],
            self._y.iloc[_start:_stop],
            name=self.data_name,
        )

        return xy

    @property
    def X(self):
        """faire data.X.train ou data.X.test"""

        train_tuple = self._start_stop("train")
        print(train_tuple)

        test_tuple = self._start_stop("test")
        print(test_tuple)

        val_tuple = self._start_stop("val")
        print(val_tuple)

        # tvt
        x = XX(
            self._X,
            train_tuple,
            test_tuple,
            val_tuple,
            name=self.data_name,
        )

        return x

    @property
    def y(self):
        """ """

        train_tuple = self._start_stop("train")
        test_tuple = self._start_stop("test")
        val_tuple = self._start_stop("val")

        # tvt
        y = YY(
            self._y,
            train_tuple,
            test_tuple,
            val_tuple,
            name=self.data_name,
        )

        return y

    @property
    def y_log(self):
        pass

        train_tuple = self._start_stop("train")
        test_tuple = self._start_stop("test")
        val_tuple = self._start_stop("val")

        # tvt
        y_log = YY(
            self.y_log,
            train_tuple,
            test_tuple,
            val_tuple,
            name=self.data_name,
        )

        return y_log

    def __repr__(self) -> str:
        return f"""
DataClass(X={len(self.X)}, train.X={len(self.train.X)}, X.train={len(self.X.train)}
X_train_size={len(self.X.train)}, X_test_size={len(self.X.test)}, X_val_size={len(self.X.val)}
y_train_size={len(self.y.train)}, y_test_size={len(self.y.test)}, y_val_size={len(self.y.val)} )
"""


# df = pd.DataFrame({"a" : range(20), "b" : range(20)})


# class DF(pd.DataFrame) :

#     def __init__(self, df, sta=1, sto=10) -> None:
#         super().__init__(df.copy())

#         self.sta = sta
#         self.sto = sto

#         self.train = self.iloc[self.sta : self.sto]
#         self.stop = self.iloc[:self.sto : , :]

# from IPython.display import display


# _df = DF(df)
# display(_df)


# display(_df.train)
