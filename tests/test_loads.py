import pytest

# from sklearn.datasets import load_boston, load_iris
# import pandas as pd
# import numpy as np
# import random

from gargaml.loads import Loads


def test_boston():
    X, y = Loads.ames()
    df = Loads.ames(X_y=False)
    # with pytest.raises(ValueError):
    #     X, y = _boston(X_y=False)


#     nan_rate = 0.01
#     df = _boston(X_y=False, nan_rate=nan_rate)
#     N = df.shape[0] * df.shape[1]
#     n = int(nan_rate * N)
#     nan_numb = df.isna().sum().sum()
#     assert n + 1 > nan_numb > n - 10


def test_iris():
    X, y = Loads.iris()
    df = Loads.iris(X_y=False)
    # with pytest.raises(ValueError):
    #     X, y = _iris(X_y=False)


#     nan_rate = 0.01
#     df = _iris(X_y=False, nan_rate=nan_rate)
#     N = df.shape[0] * df.shape[1]
#     n = int(nan_rate * N)
#     nan_numb = df.isna().sum().sum()
#     assert n + 1 > nan_numb > n - 10


def test_seattle():
    df = Loads.seattle()


def test_hr():
    df = Loads.hr()


def test_titanic():
    df = Loads.titanic()


# def tests():

#     _ = Loads.boston()
#     _ = Loads.iris()
