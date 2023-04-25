import os, sys, logging

from gargaml.imports import *

from gargaml.build_ import Build
from gargaml.data import Data
from gargaml.eda import EDA
from gargaml.dl import DL
from gargaml.loads import Loads

# from gargaml.ml import ML

# class SkRes(pd.DataFrame):
#     """ """

#     def __init__(self, dd) -> None:
#         super().__init__(dd)
#         self.res = "res"


class Gargaml:
    """ """

    build = Build
    data: Data
    eda = EDA
    dl = DL
    loads = Loads
    # ml = ML
