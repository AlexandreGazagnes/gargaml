import pytest

# from sklearn.datasets import load_boston, load_iris
import pandas as pd
import numpy as np
import random

from gargaml.data import Load
from gargaml.data.load import (
    _ames,
    # _boston,
    # _fashion,
    _food,
    _house,
    _hr,
    _iris,
    # _mnist,
    _seattle,
    _titanic,
    _wine,
)

import pytest


list_all = [
    _ames,
    # _boston,
    # _fashion,
    _food,
    _house,
    _hr,
    _iris,
    # _mnist,
    _seattle,
    _titanic,
    _wine,
]


class TestLoad:
    """ """


class TestBasics:
    """ """

    @pytest.mark.parametrize("funct", list_all)
    def test_load(self, funct):
        """ """

        _ = funct()


class TestSepTarget:
    """ """

    @pytest.mark.parametrize("funct", list_all)
    def test_df(self, funct):
        """test df X +y"""

        # df
        df = funct(sep_target=False)
        assert isinstance(df, pd.DataFrame)

    #     # nan rate
    #     df = funct(sep_target=False, nan_rate=nan_rate)
    #     N = df.shape[0] * df.shape[1]
    #     n = int(nan_rate * N)
    #     nan_numb = df.isna().sum().sum()
    #     assert n + 10 > nan_numb > n - 10

    @pytest.mark.parametrize("funct", list_all)
    def test_Xy(self, funct):
        """test x, y sep"""

        X, y = funct(sep_target=True)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    #     X, y = Load.funct(sep_target=True, nan_rate=nan_rate)
    #     N = X.shape[0] * X.shape[1]
    #     n = int(nan_rate * N)
    #     nan_numb = X.isna().sum().sum()
    #     assert n + 10 > nan_numb > n - 10


class TestnanRate:
    """ """

    pass
