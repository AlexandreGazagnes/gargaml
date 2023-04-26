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

        m = data.memory_usage().sum() / 1000_000
        display(f"shape {data.shape}, memory {round(m,2)}MB")
        print()
        
        for t in ["float", "int", "bool", "object", "datetime"]:
            tmp = data.select_dtypes(t).copy()

            if tmp.shape[1]:

                print()
                display(f"---- {t[:3].upper()} ----")

                cols = tmp.columns
                types = tmp.dtypes.values
                nan_count = tmp.isna().sum().values
                nan_mean = tmp.isna().mean().values
                uniq = tmp.nunique().values
                uniq_p = (tmp.nunique().values / len(tmp)).round(2)
                is_sku = tmp.nunique().values == len(tmp)


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
                    dd[f"val_rand_{i}"] = data.sample(1).iloc[0]
                    if t in ["float", ] : 
                        dd[f"val_rand_{i}"] = dd[f"val_rand_{i}"].round(4)

                    dd[f"val_rand_{i}"]  = dd[f"val_rand_{i}"].values

                display(pd.DataFrame(dd))
        
        # return pd.DataFrame(dd)

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
        print()
        print()

        tmp = data.sample(s).copy()
        tmp.columns.name = "----SAMP ----"
        display(tmp)
        print()
        print()


        tmp = data.tail(n).copy()
        tmp.columns.name = "----TAIL ----"
        display(tmp)
        print()
        print()


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
                print()
                descr = tmp.describe(include=t)
                descr = descr.round(2) if t != "datetime" else descr
                display(descr)
                # print()
