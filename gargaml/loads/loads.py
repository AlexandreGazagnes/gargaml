import os, sys, random, logging

from sklearn.datasets import *
import pandas as pd

import numpy as np
import random


def _load(data: str, X_y: bool, nan_rate: float):
    # data
    if data == "ames":
        data = fetch_california_housing()
    elif data == "iris":
        data = load_iris()

    # X
    X = data.data
    y = data.target
    df = pd.DataFrame(X, columns=data.feature_names)
    df.columns = [i.lower() for i in df.columns]

    # nan_rate
    if nan_rate:
        N = df.shape[0] * df.shape[1]
        nan_numb = int(nan_rate * N)

        for _ in range(nan_numb):
            x, y = random.randint(0, df.shape[0] - 1), random.randint(
                0, df.shape[1] - 1
            )
            df.iloc[x, y] = np.NaN

    # y
    df["target"] = y

    # return X_y or df
    if X_y:
        return df.drop("target", axis=1), df["target"]

    return df


def _ames(X_y: bool = True, nan_rate: float = 0.0):
    """ """

    return _load("ames", X_y=X_y, nan_rate=nan_rate)


def _iris(X_y: bool = True, nan_rate: float = 0.0):
    """ """

    return _load("iris", X_y=X_y, nan_rate=nan_rate)


def _seattle(year="2016", *args, **kwargs):
    """ """

    assert year in ["2016", "2015"]

    _2016 = "https://gist.githubusercontent.com/AlexandreGazagnes/37b2f3c19da4c4dfc8e5e81cd883169f/raw/09cb1d4d4d3adf4f5838421afe1ca9c44a4dacc2/seattle_2016.csv"

    if year == "2016":
        df = pd.read_csv(_2016)
        assert isinstance(df, pd.DataFrame)
        return df

    return None


def _titanic(
    X_y: bool = True,
    nan_rate: float = 0.1,
):
    """ """

    url = "https://gist.githubusercontent.com/AlexandreGazagnes/9018022652ba0933dd39c9df8a600292/raw/0845ef4c2df4806bb05c8c7423dc75d93e37400f/titanic_train_raw_csv"

    df = pd.read_csv(url)

    if not X_y:
        return df

    return df.drop(columns="Survived"), df.Survived


def _hr(*args, **kwargs):
    """ """

    url = "https://gist.githubusercontent.com/AlexandreGazagnes/c52899a975ab6114f6ca83eeabd563b7/raw/d539a5028e53a2fa446c4dbdf142cc10066f8ec2/hr_datascience_train.csv"

    return pd.read_csv(url)


def _house(*args, **kwargs):
    """ """

    url = "https://gist.githubusercontent.com/AlexandreGazagnes/796384619817c9e93e60a288abc188ab/raw/989be9dfbb83e40ac3374bb2d629225be6fcaa1b/dataset_house_price_raw"

    return pd.read_csv(url)
