# from agaml import pd

import pandas as pd
from IPython.display import display


class FirstTour:
    """class first tour"""

    @classmethod
    def info(
        cls,
        data: pd.DataFrame,
        n_rand: int = 5,
    ):
        """ """

        cols = data.columns
        types = data.dtypes.values
        nan_count = data.isna().sum().values
        nan_mean = data.isna().mean().values
        uniq = data.nunique().values
        uniq_p = (data.nunique().values / len(data)).round(2)
        is_sku = data.nunique().values == len(data)

        dd = {
            "cols": cols,
            "types": types,
            "nan_sum": nan_count,
            "nan_mean": nan_mean.round(2),
            "uniq_sum": uniq,
            "uniq_rate": uniq_p,
            "is_sku": is_sku,
        }

        for i in range(n_rand):
            dd[f"val_rand_{i}"] = data.sample(1).iloc[0].to_list()

        m = data.memory_usage().sum() / 1000_000

        print(f"shape {data.shape}, memory {round(m,2)}MB")

        return pd.DataFrame(dd)

    @classmethod
    def display(
        cls,
        data: pd.DataFrame,
        n: int = 5,
        s: int = 10,
    ):
        """ """

        tmp = data.head(n).copy()
        tmp.columns.name = "----HEAD ----"
        display(tmp)

        tmp = data.sample(s).copy()
        tmp.columns.name = "----SAMP ----"
        display(tmp)

        tmp = data.tail(n).copy()
        tmp.columns.name = "----TAIL ----"
        display(tmp)

    @classmethod
    def describe(
        cls,
        data: pd.DataFrame,
    ):
        """ """

        for t in ["float", "int", "bool", "object", "datetime"]:
            tmp = data.select_dtypes(t).copy()
            tmp.columns.name = f"---- {t[:3].upper()} ----"
            if tmp.shape[1]:
                descr = tmp.describe(include=t)
                descr = descr.round(2) if t != "datetime" else descr
                display(descr)
