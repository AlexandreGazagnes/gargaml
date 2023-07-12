import pytest
from typing import Callable

# from sklearn.datasets import load_boston, load_iris
import pandas as pd
import numpy as np
import random

from gargaml.data import Load
from gargaml.data.load import Load

import pytest

from ..conftest import fixt

list_all = Load.list_all()


class TestLoad:
    """ """


class TestBasics:
    """ """

    @pytest.mark.parametrize("key", list_all)
    def test_load(self, key):
        """ """

        funct = Load.dict_all(key=key)
        _ = funct()


class TestSepTarget:
    """ """

    @pytest.mark.parametrize("key", list_all)
    def test_df(self, key):
        """test df X +y"""

        # df

        funct = Load.dict_all(key=key)
        df = funct(sep_target=False)
        assert isinstance(df, pd.DataFrame)

    #     # nan rate
    #     df = funct(sep_target=False, nan_rate=nan_rate)
    #     N = df.shape[0] * df.shape[1]
    #     n = int(nan_rate * N)
    #     nan_numb = df.isna().sum().sum()
    #     assert n + 10 > nan_numb > n - 10

    @pytest.mark.parametrize("key", list_all)
    def test_Xy(self, key):
        """test x, y sep"""

        funct = Load.dict_all(key=key)
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
