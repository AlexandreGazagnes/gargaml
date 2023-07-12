# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: env
#     language: python
#     name: python3
# ---

# # 1 Preliminaries

# ## 1.0 Context

# * About Seattle dataframe
# * Small démo of gargaml usage

# ## 1.1 System

# !pwd

# !cd ..

# !pwd

# !ls

# !uname -a

# !which python

# ## 1.2 Install

# +
# if needed :

# # !pip install -r requirements.txt

# +
# if needed :

# # !pip freeze > requirements.freeze
# -

# ## 1.3 Import

# +
# in one line :

from gargaml import *

# +
# or manually :

# import random, os, sys, warnings, datetime, time, logging
# from IPython.display import display
# # pandarallel

# import pandas as pd
# import numpy as np

# import scipy as sp
# import scipy.stats as st

# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import plotly.io as pio
# import missingno as msno


# from sklearn.base import *

# from sklearn.feature_selection import *
# from sklearn.feature_extraction import *
# from sklearn.preprocessing import *
# from sklearn.pipeline import *
# from sklearn.covariance import *
# from sklearn.decomposition import *
# from sklearn.model_selection import *
# from sklearn.impute import *
# from sklearn.metrics import *
# from sklearn.cluster import *
# from sklearn.compose import *

# from sklearn.dummy import *
# from sklearn.linear_model import *
# from sklearn.svm import *
# from sklearn.neighbors import *
# from sklearn.ensemble import *


# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline

# from xgboost import XGBRegressor, XGBRFRegressor

# ...
# -

# ## 1.4 Data

# +
# all datasets avialable :

Loads.list_all

# +
# seattle 2016 :

df = Loads.seattle(year="2016")
df.head(2)
# -

# ## 1.5 Options and Graphics

# +
# boot seaborn  :

sns.set()

# +
# usefull consts :

DISPLAY = True
FRAC = 1.0
LAZY = False

# +
# warning messages :

warnings.filterwarnings(action="once")

# or :
# warnings.filterwarnings('ignore')

# +
# if png and not fancy graphs with plotly :

# pio.renderers.default = "png"
# -

# ## 1.6 Third parties and utils

# +
# placeholder
# -

# ## 1.7 Functions and class

# +
# placeholder
# -

# # 2 First Tour

# ## 2.0 Pre-cleaning

# +
# about columns quality :

df.columns

# +
# clean columns :


def clean(txt):
    txt = txt.lower().strip()
    replace = [
        ("(s)", ""),
        ("(", "_"),
        (")", ""),
        ("/", "_"),
        ("__", "_"),
    ]

    for k, v in replace:
        txt = txt.replace(k, v)

    txt = txt.lower().strip()
    return txt


df.columns = map(clean, df.columns)
df.columns

# +
# recast osebuildingid :

df.osebuildingid = df.osebuildingid.astype(str)
df.head(2)
# -

df.steamuse_kbtu = df.steamuse_kbtu.apply(lambda i: i > 0).astype(int)
df.electricity_kbtu = df.electricity_kbtu.apply(lambda i: i > 0).astype(int)
df.naturalgas_kbtu = df.naturalgas_kbtu.apply(lambda i: i > 0).astype(int)

# +
# select columns for faster and better analysis


cols = [
    "primarypropertytype",
    "neighborhood",
    "latitude",
    "longitude",
    "yearbuilt",
    "numberofbuildings",
    "numberoffloors",
    "propertygfaparking",
    "propertygfabuilding",
    "energystarscore",
    "siteenergyuse_kbtu",
    "steamuse_kbtu",
    "electricity_kbtu",
    "naturalgas_kbtu",
]


df = df.loc[:, cols]
df.head(3)
# -

for c in df.select_dtypes(object):
    df[c] = df[c].str.lower()

# +
df = df.loc[df.propertygfabuilding.notna()]
df = df.loc[df.propertygfabuilding > 1, :]

df = df.loc[df.siteenergyuse_kbtu.notna()]
df = df.loc[df.siteenergyuse_kbtu > 1, :]

df

# +
df.numberofbuildings = df.numberofbuildings.fillna(1)
df.numberoffloors = df.numberoffloors.fillna(1)

# df.numberofbuildings = df.numberofbuildings.astype(int)
# df.numberoffloors = df.numberoffloors.astype(int)

df.numberofbuildings = df.numberofbuildings.replace({0: 1})
df.numberoffloors = df.numberoffloors.replace({0: 1})

# df.numberofbuildings = df.numberofbuildings.apply(lambda i : i if i>=1 else 1)
# df.numberoffloors = df.numberoffloors.apply(lambda i : i if i>=1 else 1)

df.propertygfaparking = df.propertygfaparking.apply(lambda i: i > 0).astype(int)
df["bool_energystarscore"] = df.energystarscore.isna().astype(int)

df["gfaperfloor"] = df.propertygfabuilding / df.numberoffloors
df["gfaperbuilding"] = df.propertygfabuilding / df.numberofbuildings

df.describe()
# -

# df.primarypropertytype.value_counts()
df.primarypropertytype = df.primarypropertytype.replace(
    {"office": "small- and mid-sized office"}
)
df.primarypropertytype.value_counts()

# ## 2.1 Display

# +
# display df :

EDA.first_tour.display(df)
# -

# ## 2.2 Structure

# +
# info :

EDA.first_tour.info(df)

# +
# usefull dtype info :

df.dtypes.value_counts()
# -

# ## 2.3 NaN and duplicated

# +
# # mnso :

# EDA.nan.viz(df)

# +
# nan rate by columns :

EDA.nan.rate(df)

# +
# filter on poor columns :

EDA.nan.rate(df, threshold=0.75)

# +
# same for lines :

EDA.nan.rate(df, axis=1, threshold=0.5)

# +
# about nan distribution by column :

EDA.nan.rate(df, axis=1, threshold=0.0).value_counts(normalize=True).round(2)

# +
# duplicated :

df.duplicated().sum()
# -

# ## 2.4 Data Inspection

# +
# describe per type :

EDA.first_tour.describe(df)

# +
# # global correlation :

# _ = EDA.study.corr(df)

# +
# # about distribution :

# EDA.study.skew(df)

# +
# # pairplot :
# # WARNING : very long computation, avoid if possible else uncomment

# frac = 0.05
# sns.pairplot(df.sample(frac=frac), corner=True)

# +
# # about outliers :

# EDA.study.outlier(df)

# +
# # multi colinearity :

# EDA.study.vif(df, scale=True)
# -

# # 2.5 ACP

# +
# _df = EDA.study.outlier(df, display_=False)
# _df = _df.loc[_df._outlier<0.5]
# _df.shape

# +
# X = _df.drop(columns="siteenergyuse_kbtu")
# y = _df.siteenergyuse_kbtu

# +
# pca = EDA.pca(X)

# +
# _ = pca.variance()

# +
# _ = pca.pcs()

# +
# pca.correlation_graph([0, 1])

# +
# pca.factorial_planes([0, 1])

# +
# pca.factorial_planes([0, 1], clusters="primarypropertytype")

# +
# pca.factorial_planes([0, 1, 2])

# +
# pca.factorial_planes([0, 1, 2], clusters="primarypropertytype")

# +
# pca.factorial_planes([0, 1, 2], clusters="neighborhood")
# -

# # 3. Modelisation

# +
## data split
# -

df

data = Data.dataclass(df, "siteenergyuse_kbtu", test_size=0.33)
data

data.X

data.y

data.train.X

data.train.y

# +
pipe = Pipeline(
    [
        ("imputer", KNNImputer()),
        ("scaler", StandardScaler()),
        ("estimator", RandomForestClassifier()),
    ]
)

param_grid = {
    "scaler": [
        StandardScaler(),
        QuantileTransformer(n_quantiles=100),
        Normalizer(),
        "passthrough",
    ],
    "estimator": [
        DummyRegressor(),
        # RandomForestRegressor() ,
        # XGBRegressor(),
        # LinearRegression(),
    ],
}

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=5,
    n_jobs=4,
    verbose=1,
    return_train_score=True,
)

grid

# +


grid.fit(data.X.train.select_dtypes(include=np.number), data.y.train)
# -

res = ML.results.resultize(grid)
res
