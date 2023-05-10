import os, sys, random, logging, warnings, time, datetime, secrets, math, pickle

# from math import ceil

from dataclasses import dataclass

from itertools import product, permutations
from collections.abc import Iterable

# from typing import List, Dict, Union, Tuple, Iterator, Callable, Optional


from IPython.display import display

import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper

# import dataprep
# from dataprep.datasets import load_dataset
from pandarallel import pandarallel


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import missingno as msno


import scipy.stats as st
import statsmodels.api as sm

from sklearn.base import *

from sklearn.feature_selection import *
from sklearn.feature_extraction import *
from sklearn.decomposition import *
from sklearn.model_selection import *
from sklearn.impute import *
from sklearn.preprocessing import *
from sklearn.pipeline import *
from sklearn.covariance import *
from sklearn.metrics import *
from sklearn.compose import *

from sklearn.dummy import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.cluster import *

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from xgboost import XGBRegressor, XGBRFRegressor, XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier, LGBMRegressor

import shap

# pandarallel.initialize(progress_bar=True, nb_workers=4)

sns.set()
