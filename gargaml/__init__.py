import os, sys, random, logging, warnings, time, datetime
from dataclasses import dataclass

from IPython.display import display

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno

sns.set()

import scipy.stats as stats
import statsmodels.api as sm

from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.compose import *
from sklearn.feature_selection import *
from sklearn.feature_extraction import *
from sklearn.impute import *
from sklearn.compose import *
from sklearn.decomposition import *
from sklearn.metrics import *
from sklearn.dummy import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.ensemble import *

from lightgbm import *
from xgboost import *

# import shap

from gargaml.eda import Geda
from gargaml.ml import Gml
from gargaml.loads import Loads


class SkRes(pd.DataFrame):
    """ """

    def __init__(self, dd) -> None:
        super().__init__(dd)
        self.res = "res"


class Gargaml:
    loads = Loads
    eda = Geda
    ml = Gml
