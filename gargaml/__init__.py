import pandas as pd


class SkRes(pd.DataFrame):
    """ """

    def __init__(self, dd) -> None:
        super().__init__(dd)
        self.res = "res"
