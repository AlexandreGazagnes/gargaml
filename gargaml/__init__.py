from .imports import * 

from .build import Build
from .data import Data
from .eda import EDA
from .dl import DL
from .loads import Loads
from .ml import ML

# class SkRes(pd.DataFrame):
#     """ """

#     def __init__(self, dd) -> None:
#         super().__init__(dd)
#         self.res = "res"


class Gargaml:
    """ """

    build = Build
    data : Data
    eda = EDA
    dl = DL
    loads = Loads
    ml = ML