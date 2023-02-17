# from agaml import pd

import pandas as pd


def _info(data, n_rand=5):
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


def _display(data, n=5, s=10):
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


def _describe(data):
    """ """

    for t in ["float", "int", "bool", "object", "datetime"]:

        tmp = data.select_dtypes(t).copy()
        tmp.columns.name = f"---- {t[:3].upper()} ----"
        if tmp.shape[1]:
            display(tmp.describe(include=t))


class FirstTour:
    info = _info
    desribe = _describe
    display = _display