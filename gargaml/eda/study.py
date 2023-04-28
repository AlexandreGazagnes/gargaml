import logging
from math import ceil

import numpy as np
import pandas as pd

import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import missingno as msno


from IPython.display import display

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.impute import KNNImputer

sns.set()


class Study:
    """corr : correlation"""

    @classmethod
    def _prepare(
        cls,
        df: pd.DataFrame,
        numeric_only: bool = True,
        nan_threshold: float = 0.5,
        force_nan_impute: bool = True,
        scale: bool = False,
        scaler: str = "minmax",
        ignore_cols: list = None,
    ):
        """ " """

        # copy
        _df = df.copy() 

        # ignore_cols
        if ignore_cols :
            cols = [i for i in ignore_cols if i in _df.columns] 
            _df =_df.drop(columns=cols)

        # nan_threshold
        tmp = _df.isna().mean()
        cols = [i for i in tmp[tmp >= nan_threshold].index if i in _df.columns]
        _df = _df.drop(                    columns=cols)

        # numeric_only
        if numeric_only :
            _df = _df.select_dtypes(include=np.number)

        # force_nan_impute
        if force_nan_impute:
            _df = pd.DataFrame(KNNImputer().fit_transform(_df), columns=_df.columns)

        # scale
        if scale:
            sca = MinMaxScaler() if "min" in scaler.lower() else StandardScaler()
            _df = pd.DataFrame(sca.fit_transform(_df), columns=_df.columns)

        return _df

    @classmethod
    def corr(
        cls,
        df: pd.DataFrame,
        figsize: int = 12,
        nan_threshold: float = 0.5,
        force_nan_impute: bool = False,
        scale: bool = False,
        ignore_cols: list = None,
    ):
        """ """

        _df = Study._prepare(
            df,
            numeric_only=True,
            nan_threshold=nan_threshold,
            force_nan_impute=force_nan_impute,
            scale=scale,
            scaler="minmax",
            ignore_cols=ignore_cols,
        )

        corr = _df.select_dtypes(include=np.number).corr()

        fig, ax = plt.subplots(1, 1, figsize=(ceil(figsize * 1.5), figsize))

        mask = np.triu(corr)
        ax = sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            mask=mask,
        )

        corr = corr.round(2).astype(str)
        _len = len(corr)
        for i in range(_len):
            corr.iloc[i, i:] = ""

        return corr

    @classmethod
    def skew(
        cls,
        df: pd.DataFrame,
        display_: bool = True,
        force_nan_impute: bool = True,
        nan_threshold: float = 0.5,
        scale: bool = False,
        ignore_cols: list = None,
    ):
        """compute skew for num cols and display log1p gain for each"""

        _df = Study._prepare(
            df,
            numeric_only=True,
            nan_threshold=nan_threshold,
            force_nan_impute=force_nan_impute,
            scale=scale,
            scaler="minmax",
            ignore_cols=ignore_cols,
        )

        cols = _df.columns
        raw_skew = [_df[c].skew() for c in cols]

        log1p = [(1 if _df[c].min() >= 0 else 0) for c in cols]
        log_vals = [
            (np.log1p(_df[c].values) if bool_ else _df[c].values)
            for c, bool_ in zip(cols, log1p)
        ]
        log_skew = [pd.Series(i).skew() for i in log_vals]
        skew = pd.DataFrame(
            {"col": cols, "raw_skew": raw_skew, "log1p": log1p, "log_skew": log_skew}
        )

        skew["gain"] = (skew["raw_skew"].values - skew["log_skew"].values) / skew[
            "raw_skew"
        ].values

        # our skew df
        skew = skew.round(2)

        if display_:
            # graphs
            _skew = skew.copy()
            _skew["raw_val"] = [_df[i].values for i in cols]
            _skew["log_val"] = log_vals

            for i, c in enumerate(_skew.col):
                # subplots
                fig = make_subplots(rows=1, cols=2)

                # normal
                x1 = _skew.loc[_skew.col == c, :].iloc[0].raw_val
                fig.add_trace(go.Histogram(x=x1, name=f"normal"), row=1, col=1)

                # log
                if _skew.loc[_skew.col == c, "log1p"].iloc[0]:
                    x2 = _skew.loc[_skew.col == c, :].iloc[0].log_val
                    fig.add_trace(go.Histogram(x=x2, name=f"log1p"), row=1, col=2)

                # fig
                fig.update_layout(
                    height=300,
                    width=800,
                    title_text=f"RAW {c} => norm skew : {_skew.loc[_skew.col == c].iloc[0].raw_skew} ==> log skew {_skew.loc[_skew.col == c].iloc[0].log_skew} ==> gain {_skew.loc[_skew.col == c, ].iloc[0].gain}",
                )
                fig.show()

        return skew

    @classmethod
    def outlier(
        cls,
        df: pd.DataFrame,
        display_: bool = True,
        model: str = "e",
        force_nan_impute: bool = True,
        nan_threshold: float = 0.5,
        scale: bool = False,
        ignore_cols: list = None,
    ):
        """apply outlier stat traeatment display desribe before / after return df with _outlier col"""


        _df = Study._prepare(
            df,
            numeric_only=True,
            nan_threshold=nan_threshold,
            force_nan_impute=force_nan_impute,
            scale=scale,
            scaler="std",
            ignore_cols=ignore_cols,
        )

        assert model in ["e", "i", "k"]

        if model == "e":
            model = EllipticEnvelope
        elif model == "i":
            model = IsolationForest
        elif model == "k":
            model = LocalOutlierFactor

        try:
            model = model()
        except:
            pass

        ee = model.fit_predict(_df)
        _df_e = _df.loc[~(ee == -1)]

        s1, s2 = _df.shape[0], _df_e.shape[0]
        r = round((s1 - s2) / s1, 2)

        # display
        display(f"shape Original : {s1} shape_cleaned {s2} => loss {r}")

        display("--------- ORIGINAL ----------")
        display(_df.describe().round(2))  # .iloc[1:]
        print()
        display("--------- CLEANED ----------")
        display(_df_e.describe().round(2))  # .iloc[1:]
        print()

        # graphs
        if display_:
            for c in _df.columns:
                # subplots
                fig = make_subplots(rows=1, cols=2)

                # normal
                x1 = _df.loc[:, c].values
                fig.add_trace(go.Histogram(x=x1, name=f"normal"), row=1, col=1)

                # log
                x2 = _df_e.loc[:, c].values
                fig.add_trace(go.Histogram(x=x2, name=f"cleaned"), row=1, col=2)

                min_mean_max = [
                    round(_df.loc[:, c].min(), 2),
                    round(_df.loc[:, c].mean(), 2),
                    round(_df.loc[:, c].max(), 2),
                ]
                _min_mean_max = [
                    round(_df_e.loc[:, c].min(), 2),
                    round(_df_e.loc[:, c].mean(), 2),
                    round(_df_e.loc[:, c].max(), 2),
                ]
                # fig
                fig.update_layout(
                    height=300,
                    width=800,
                    title_text=f"RAW {c} [ min, mean, max]\nnorm  : {min_mean_max } ==> cleaned {_min_mean_max} ",
                )
                fig.show()

        _df = df.copy()
        _df["_outlier"] = [0 if i > 0 else 1 for i in ee]

        return _df

    @classmethod
    def vif(
        cls,
        df: pd.DataFrame,
        scale: bool = False,
        force_nan_impute: bool = True,
        nan_threshold: float = 0.5,
        ignore_cols:list=None,
    ):
        """ """
      
        _df = Study._prepare(
            df,
            numeric_only=True,
            nan_threshold=nan_threshold,
            force_nan_impute=force_nan_impute,
            scale=scale,
            scaler="std",
            ignore_cols=ignore_cols,
        )

        vif_data = pd.DataFrame()
        vif_data["feature"] = _df.columns

        # calculating VIF for each feature
        vif_data["vif"] = [
            variance_inflation_factor(_df.values, i) for i in range(len(_df.columns))
        ]

        return vif_data.sort_values("vif", ascending=False).round(2)

    @classmethod
    def pairplot(
        cls,
        df: pd.DataFrame,
        nan_threshold: float = 0.5,
        force_nan_impute: bool = False, 
        scale: bool = False,
        ignore_cols: list = None,
    ):
        """ """

        # TODO code this

        _df = Study._prepare(
            df,
            numeric_only=True,
            nan_threshold=nan_threshold,
            force_nan_impute=force_nan_impute,
            scale=scale,
            scaler="std",
            ignore_cols=ignore_cols,
        )

        # update pair plot with clever filter to fast display

        # # si en dessous de 100 => frac = 0,
        # si # de 100 à 1000 = > frac 0.33
        # si 1000 à 10 000 0.25
        # si 10_000 à 100_000 > 0.15
        # si > 100_000 > 0.1

        # si 1 valeur => NON
        # si booléen  ???

    @classmethod
    def stats(cls, df: pd.DataFrame, 
        nan_threshold: float = 0.5,
        force_nan_impute: bool = False, 
        scale: bool = False,
        ignore_cols: list = None,
    ):
        """ """

        _df = Study._prepare(
            df,
            numeric_only=True,
            nan_threshold=nan_threshold,
            force_nan_impute=force_nan_impute,
            scale=scale,
            scaler="std",
            ignore_cols=ignore_cols,
        )

        # perform all test ANOVA etc etc


# class Test:
#     """ """

#     @classmethod
#     def bla(cls, txt):
#         """ """

#         txt = Test._BLABLA(txt)
#         print(f"bla dit {txt}")

#     @classmethod
#     def _BLABLA(cls, txt):

#         txt = f"_BLABLA dit {txt}"
#         return txt
