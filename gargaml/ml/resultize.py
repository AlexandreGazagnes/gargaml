import os, warnings, logging, time, datetime, random, secrets
from IPython.display import display

import pandas as pd
import numpy as np

from sklearn.model_selection import *

results = pd.DataFrame()
RESULTS = pd.DataFrame()


def resultize(
    grid: GridSearchCV,
    top_only: bool = False,
    verbose: int = 1,
    **kwargs: dict,
):
    """ """

    if not isinstance(grid, GridSearchCV) : 
        raise AttributeError('GridSearchCV')

    # acces results
    try:
        global RESULTS
        _RESULTS = True
    except Exception as e:
        if verbose : 
            logging.warning("RESULTS not avialable")

    # base res
    res = pd.DataFrame(grid.cv_results_)
    cols = [i for i in res.columns if "split" not in i]
    res = res.loc[:, cols]

    # update test/val
    cols = [i.replace("test", "val") for i in res.columns]
    res.columns = cols

    # drop score time
    cols = [
        "mean_score_time",
        "std_score_time",
        "std_fit_time" ,
    ] # "rank_val_score"

    res.drop(columns=cols, inplace=True)

    # do cast
    for c in res.columns:  # [i for i in res.columns if "param" in i]
        try:
            res[c] = res[c].astype(float)
        except:
            res[c] = res[c].astype(str)
            res[c] = res[c].replace({"nan": np.NaN})

    # sort and round
    res = res.round(4).sort_values("mean_val_score", ascending=False)

    # meta data
    res["grid"] = grid
    res["best_estimator_"] = grid.best_estimator_
    res["datetime"] = str(datetime.datetime.now())[:19]
    for k, v in kwargs.items():
        res[k] = v

    # reorder cols
    end_cols = [
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
    ]
    first_cols = [
        i for i in res.columns if (i not in end_cols)
    ]  # and (i not in universal_cols)
    final_cols = first_cols + end_cols
    res = res.loc[:, final_cols]

    # update results global
    if _RESULTS:
        _res = res.copy().head(1) if top_only else res.copy()
        RESULTS = pd.concat([RESULTS, _res], ignore_index=True)
        RESULTS = RESULTS.sort_values("mean_test_score", ascending=False)

    # verbose
    if verbose >= 1 : 
        display(res.round(2).head(10))
    if verbose >=2 : 
        display(RESULTS.round(2).head(5))

    return res.round(2).head(10)



# class Results:
#     """class Results
#     resultize : return an pd.DataFrame of fancy results"""

#     @classmethod
#     def grid(
#         cls,
#         grid: GridSearchCV,
#         **kwargs,
#     ):
#         """ """

#         # grid res
#         res = pd.DataFrame(grid.cv_results_)

#         # round
#         res = res.round(2)

#         # default structure :
#         for k in [
#             "param_estimator",
#             "param_preprocessor",
#             "param_reductor",
#             "param_scaler",
#             "param_imputer",
#         ]:
#             if k not in res.columns:
#                 res.loc[:, k] = np.NaN

#         # kwargs
#         for k, v in kwargs.items():
#             res.loc[:, k] = v

#         # add grid
#         res["grid"] = grid

#         # list train and test socres
#         for k in ["test", "train"]:
#             cols = [i for i in res.columns if ("split" in i) and (k in i)]
#             res.loc[:, f"{k}_scores"] = res.loc[:, cols].apply(
#                 lambda i: list(i.values), axis=1
#             )

#         # drop split
#         cols = [i for i in res if "split" not in i]
#         res = res.loc[:, cols]

#         # drop fit std
#         cols = [i for i in res if ("std" in i) and ("time" in i)]
#         res.drop(columns=cols, inplace=True)

#         # str params
#         cols = [i for i in res.columns if "param_" in i]
#         for col in cols:
#             res.loc[:, col] = res.loc[:, col].astype(str)
#         # params str
#         if "params" in res.columns:
#             res.loc[:, "params"] = res.loc[:, "params"].apply(lambda i: str(i))
#             # res.loc[:, "params"] = res.loc[:, "params"].apply(lambda i : dict(i) if i else {})
#             # res.loc[:, "params"] = res.loc[:, "params"].apply(lambda i : {k:str(v)} for k, v in i.items())

#         # replace val by test
#         test_cols = {i: i.replace("test", "val") for i in res.columns}
#         res.rename(columns=test_cols, inplace=True)

#         # cols at the end
#         # scores = ["rank_val_score", "mean_val_score", "std_val_score", "mean_train_score", "std_train_score"]
#         cols = [i for i in res.columns if "score" not in i] + [
#             i for i in res.columns if "score" in i
#         ]
#         res = res.loc[:, cols]

#         # sort and round
#         res = res.sort_values("mean_val_score", ascending=False).round(2)

#         return res

#     @classmethod
#     def cv(cls, grid, X, y, **kwargs):
#         """ """

#         _res = pd.DataFrame({"_index": [], "y_test": [], "y_pred": [], "y_prob": []})
#         model = grid.best_estimator_

#         for i, (train_index, test_index) in enumerate(
#             StratifiedShuffleSplit(n_splits=20, test_size=0.3).split(X, y)
#         ):
#             print(f"Fold {i}:")

#             X_train, X_test = X.loc[train_index], X.loc[test_index]
#             y_train, y_test = y.loc[train_index], y.loc[test_index]

#             model.fit(X_train, y_train)

#             __res = pd.DataFrame(
#                 {
#                     "_index": X_test.index.values,
#                     "y_test": y_test.values,
#                     "y_pred": model.predict(X_test),
#                     "y_prob": np.NaN,
#                 }
#             )  # model.predict_proba(X_test)
#             _res = pd.concat([_res, __res], ignore_index=True)

#         return _res
