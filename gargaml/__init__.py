import pandas as pd

from .eda import EDA
from .ml import ML
from .loads import Loads


class SkRes(pd.DataFrame):
    """ """

    def __init__(self, dd) -> None:
        super().__init__(dd)
        self.res = "res"


class Gargaml:
    loads = Loads
    eda = EDA
    ml = ML
