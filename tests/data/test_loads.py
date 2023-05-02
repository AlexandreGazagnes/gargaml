import pytest

# from sklearn.datasets import load_boston, load_iris
# import pandas as pd
# import numpy as np
# import random

from gargaml.data import Load


def test_boston():
    X, y = Load.ames()
    df = Load.ames(X_y=False)
    # with pytest.raises(ValueError):
    #     X, y = _boston(X_y=False)


#     nan_rate = 0.01
#     df = _boston(X_y=False, nan_rate=nan_rate)
#     N = df.shape[0] * df.shape[1]
#     n = int(nan_rate * N)
#     nan_numb = df.isna().sum().sum()
#     assert n + 1 > nan_numb > n - 10


def test_iris():
    X, y = Load.iris()
    df = Load.iris(X_y=False)
    # with pytest.raises(ValueError):
    #     X, y = _iris(X_y=False)


#     nan_rate = 0.01
#     df = _iris(X_y=False, nan_rate=nan_rate)
#     N = df.shape[0] * df.shape[1]
#     n = int(nan_rate * N)
#     nan_numb = df.isna().sum().sum()
#     assert n + 1 > nan_numb > n - 10


def test_seattle():
    df = Load.seattle()


def test_hr():
    df = Load.hr()


def test_titanic():
    df = Load.titanic()


# def tests():

#     _ = Load.boston()
#     _ = Load.iris()
