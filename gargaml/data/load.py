import os, sys, random, logging
import string, math
from sklearn.datasets import *
import pandas as pd

import numpy as np
import random


from .utils import _preclean_seattle, _clean_columns, _add_nan, _do_sep_target


class Load:
    """Class load use to DL and load class form the internet  :

    aviablable methods :
        ames
        iris
        hr
        titanic
        food
        seattle
        house
        wine

    positionnal agrs :
        sep_target: bool = False, return df if False or X, y if True
        nan_rate: float in (0.0, 0.95)= 0.0 : forece to add nan in the df
        precleaning : bool=False : force a specific amount of ops to clean the df
        version : str=None, use a specif verion of the df
    """

    @classmethod
    def _sk_load(
        cls,
        name: str,
        sep_target: bool = False,
        nan_rate: float = 0.00,
        precleaning=False,
        version=None,
    ):
        assert name in ["ames", "iris"]

        # name / fectch
        if name == "ames":
            data = fetch_california_housing()
        elif name == "iris":
            data = load_iris()

        # df
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target

        # target name
        if name == "iris":
            replace_dict = {k: v for k, v in enumerate(data.target_names)}
            df.target = df.target.astype(int).replace(replace_dict)

        df.columns = map(_clean_columns, df.columns)

        df = _add_nan(df, nan_rate=nan_rate)

        return _do_sep_target(df, sep_target, "target")

    @classmethod
    def ames(
        cls,
        sep_target: bool = False,
        nan_rate: float = 0.0,
        precleaning=False,
        version=None,
    ):
        """ """

        return Load._sk_load("ames", sep_target=sep_target, nan_rate=nan_rate)

    @classmethod
    def iris(
        cls,
        sep_target: bool = False,
        nan_rate: float = 0.0,
        precleaning=False,
        version=None,
    ):
        """ """

        return Load.__sk_load("iris", sep_target=sep_target, nan_rate=nan_rate)

    @classmethod
    def seattle(
        cls,
        sep_target: bool = False,
        version: str = "2016",
        preclean: bool = False,
        nan_rate: float = 0.0,
        *args,
        **kwargs,
    ):
        """ """

        # check
        assert version in ["2016", "2015"]
        if nan_rate:
            raise AttributeError("not implemented : nan_rate in False, None, 0")
        if version != "2016":
            raise AttributeError("not implemented : year=2016 ")

        # dl
        _2016 = "https://gist.githubusercontent.com/AlexandreGazagnes/37b2f3c19da4c4dfc8e5e81cd883169f/raw/09cb1d4d4d3adf4f5838421afe1ca9c44a4dacc2/seattle_2016.csv"
        df = pd.read_csv(_2016)
        assert isinstance(df, pd.DataFrame)

        # clean
        df.columns = map(_clean_columns, df.columns)
        df.osebuildingid = df.osebuildingid.astype(str)

        # preclean
        if preclean:
            df = _preclean_seattle(df)

        return _do_sep_target(df, sep_target, "siteenergyuse_kbtu")

    @classmethod
    def titanic(
        cls,
        sep_target: bool = False,
        nan_rate: float = 0.0,
        precleaning=False,
        version=None,
    ):
        """ """

        url = "https://gist.githubusercontent.com/AlexandreGazagnes/9018022652ba0933dd39c9df8a600292/raw/0845ef4c2df4806bb05c8c7423dc75d93e37400f/titanic_train_raw_csv"

        df = pd.read_csv(url)

        df.columns = map(_clean_columns, df.columns)

        return _do_sep_target(df, sep_target, "survived")

    @classmethod
    def hr(
        cls,
        sep_target: bool = False,
        nan_rate: float = 0.0,
        precleaning=False,
        version=None,
    ):
        """ """

        url = "https://gist.githubusercontent.com/AlexandreGazagnes/c52899a975ab6114f6ca83eeabd563b7/raw/d539a5028e53a2fa446c4dbdf142cc10066f8ec2/hr_datascience_train.csv"

        df = pd.read_csv(url)
        df.columns = map(_clean_columns, df.columns)

        if not sep_target:
            return df

        # rename Salary as target
        target = "salary"

        return df.drop(columns=target), df[target]

    @classmethod
    def house(
        cls,
        sep_target: bool = False,
        nan_rate: float = 0.0,
        precleaning=False,
        version=None,
    ):
        """house price regression from kaggle"""

        if nan_rate:
            raise AttributeError("nan_rate invalid for this df, retry nan_rate=None")

        url = "https://gist.githubusercontent.com/AlexandreGazagnes/796384619817c9e93e60a288abc188ab/raw/989be9dfbb83e40ac3374bb2d629225be6fcaa1b/dataset_house_price_raw"

        df = pd.read_csv(url)

        df.columns = map(_clean_columns, df.columns)

        return _do_sep_target(df, sep_target, "saleprice")

    @classmethod
    def wine(
        cls,
        sep_target: bool = False,
        nan_rate: float = 0.0,
        precleaning=False,
        version=None,
    ):
        """ """

        if nan_rate:
            raise AttributeError("nan_rate invalid for this df, retry nan_rate=None")

        url = "https://gist.githubusercontent.com/AlexandreGazagnes/e3e3a6ece82363fd1cafdd0f32563fce/raw/84c63cbee273574023b48bb750e91616c5253c74/wine-quality"

        df = pd.read_csv(url)

        df.columns = map(_clean_columns, df.columns)

        return _do_sep_target(df, sep_target, "quality")

    @classmethod
    def food(
        cls,
        sep_target: bool = False,
        y_type="score",
        nan_rate=None,
        precleaning=False,
        version=None,
    ):
        """ """

        # val
        assert y_type in ("score", "grade")
        if nan_rate:
            raise AttributeError("nan_rate invalid for this df, retry nan_rate=None")

        logging.warning("Expected time : 30 sec to download")

        url = "https://gist.githubusercontent.com/AlexandreGazagnes/2477fc721596f91a9fb9da9c544949f8/raw/a27b65b141896cca35f5ab6d75b5fb3dd1b39683/food"
        df = pd.read_csv(url)

        df.columns = map(_clean_columns, df.columns)

        if not sep_target:
            return df

        # target
        target_cols = ["nutrition_grade_fr", "nutrition-score-fr_100g"]
        target = (
            "nutrition_grade_fr" if "grade" in y_type else "nutrition-score-fr_100g"
        )

        return df.drop(columns=target_cols), df[target]

    # @classmethod
    # def fashion():
    #     """ """

    #     return None

    # @classmethod
    # def mnist(cls,sep_target: bool = True, *args, **kwargs):
    #     """ """

    #     return None

    @classmethod
    def dict_all(cls):
        """key, val for all avialables methods"""

        return {
            "ames": Load.ames,
            "seattle": Load.seattle,
            "hr": Load.hr,
            "titanic": Load.titanic,
            "house": Load.house,
            # "mnist" :Load.,
            "food": Load.food,
            "wine": Load.wine,
            "iris": Load.iris,
            # "fashion" : _,
        }

    @classmethod
    def list_all(cls):
        """List of avliabable dataset methods"""

        li = Load.dict_all().keys()

        return sorted(list(li))
