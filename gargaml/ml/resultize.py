import pandas as pd
import numpy as np

# Nfrom agaml import pd


# def resultize(grid, extra_cols: dict = {}, keep_split: bool = True):
#     """ """

#     res = pd.DataFrame(grid.cv_results_)
#     cols = [i for i in res.columns if "split" not in i]
#     good_res = res[cols]
#     good_res = good_res.round(3)

#     # extra_cols
#     for k, v in extra_cols.items():
#         good_res[k] = v

#     # test and train splits

#     test_cols = [i for i in res.columns if (("split" in i) and ("test" in i))]
#     train_cols = [i for i in res.columns if (("split" in i) and ("train" in i))]

#     if keep_split:
#         test_splits = [ser.round(2).to_list() for _, ser in res[test_cols].iterrows()]
#         good_res["test_splits"] = test_splits
#     if keep_split and train_cols:
#         train_splits = [ser.round(2).to_list() for _, ser in res[train_cols].iterrows()]
#         good_res["train_splits"] = train_splits

#     test_cols = {i: i.replace("test", "val") for i in good_res.columns}
#     good_res.rename(columns=test_cols, inplacN=True)

#     return good_res.sort_values("rank_val_score")


def resultize(grid, **kwargs):
    """ """

    # grid res
    res = pd.DataFrame(grid.cv_results_)

    # round
    res = res.round(2)

    # default structure :
    for k in [
        "param_estimator",
        "param_preprocessor",
        "param_reductor",
        "param_scaler",
        "param_imputer",
    ]:
        if k not in res.columns:
            res.loc[:, k] = np.NaN

    # kwargs
    for k, v in kwargs.items():
        res.loc[:, k] = v

    # add grid
    res["grid"] = grid

    # list train and test socres
    for k in ["test", "train"]:
        cols = [i for i in res.columns if ("split" in i) and (k in i)]
        res.loc[:, f"{k}_scores"] = res.loc[:, cols].apply(
            lambda i: list(i.values), axis=1
        )

    # drop split
    cols = [i for i in res if "split" not in i]
    res = res.loc[:, cols]

    # drop fit std
    cols = [i for i in res if ("std" in i) and ("time" in i)]
    res.drop(columns=cols, inplace=True)

    # str params
    cols = [i for i in res.columns if "param_" in i]
    for col in cols:
        res.loc[:, col] = res.loc[:, col].astype(str)
    # params str
    if "params" in res.columns:
        res.loc[:, "params"] = res.loc[:, "params"].apply(lambda i: str(i))
        # res.loc[:, "params"] = res.loc[:, "params"].apply(lambda i : dict(i) if i else {})
        # res.loc[:, "params"] = res.loc[:, "params"].apply(lambda i : {k:str(v)} for k, v in i.items())

    # replace val by test
    test_cols = {i: i.replace("test", "val") for i in res.columns}
    res.rename(columns=test_cols, inplace=True)

    # cols at the end
    # scores = ["rank_val_score", "mean_val_score", "std_val_score", "mean_train_score", "std_train_score"]
    cols = [i for i in res.columns if "score" not in i] + [
        i for i in res.columns if "score" in i
    ]
    res = res.loc[:, cols]

    # sort and round
    res = res.sort_values("mean_val_score", ascending=False).round(2)

    return res
