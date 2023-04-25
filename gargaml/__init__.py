from gargaml.imports import * 

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
