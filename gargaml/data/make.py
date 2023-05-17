import os, sys, random, logging
import string, math
from sklearn.datasets import *
import pandas as pd

import numpy as np
import random


def _random(len_=10, cols=3, size=1_000):
    """ """

    _df = pd.DataFrame()
    for c in string.ascii_lowercase:
        _df[c] = np.random.random(size=size)

    return _df.iloc[:len_, :cols]


def _normal(len_=10, cols=3, loc=0, scale=1, size=1_000):
    """ """

    _df = pd.DataFrame()
    for c in string.ascii_lowercase:
        _df[c] = np.random.normal(loc=loc, scale=scale, size=size)

    return _df.iloc[:len_, :cols]


def _uniform(len_=10, cols=3, low=0.0, high=1.0, size=1_000):
    """ """

    _df = pd.DataFrame()
    for c in string.ascii_lowercase:
        _df[c] = np.random.uniform(low=low, high=high, size=size)

    return _df.iloc[:len_, :cols]


def _lognormal(len_=10, cols=3, size=1_000):
    """ """

    _df = pd.DataFrame()
    for c in string.ascii_lowercase:
        _df[c] = np.random.lognormal(size=size)

    return _df.iloc[:len_, :cols]


def _beta(len_=10, cols=3, size=1_000):
    """ """

    _df = pd.DataFrame()
    for c in string.ascii_lowercase:
        _df[c] = np.random.beta(size=size)

    return _df.iloc[:len_, :cols]


def _choice(
    len_=10,
    cols=3,
    choice=list("123456"),
    size=1_000,
    replace=True,
    p=[
        0.166,
    ]
    * 6,
):
    """ """

    _df = pd.DataFrame()
    for c in string.ascii_lowercase:
        _df[c] = np.random.choice(a=choice, size=size, replace=replace, p=p)

    return _df.iloc[:len_, :cols]


class Make:
    random = _random
    normal = _normal
    uniform = _uniform
    lognormal = _lognormal
    beta = _beta
    choice = _choice
