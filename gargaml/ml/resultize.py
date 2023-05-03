import os, warnings, logging, time, datetime, random, secrets
from IPython.display import display

import pandas as pd
import numpy as np

from sklearn.model_selection import *

import pickle

results = pd.DataFrame()
RESULTS = pd.DataFrame()


def resultize(
    grid: GridSearchCV,
    top_only: bool = True,
    verbose: int = 1,
    **kwargs: dict,
):
    """ """

    #################################

    #################################
    #################################

    if not isinstance(grid, GridSearchCV):
        raise AttributeError("GridSearchCV")

    # acces results
    try:
        global RESULTS
        _RESULTS = True
    except Exception as e:
        if verbose:
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
        "std_fit_time",
    ]  # "rank_val_score"

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
        "mean_val_score",
        "std_val_score",
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
        RESULTS = RESULTS.sort_values("mean_val_score", ascending=False)

    # verbose
    if verbose >= 1:
        display(res.round(2).head(10))
    if verbose >= 2:
        display(RESULTS.round(2).head(5))

    return res.round(2).head(10)


class Test(pd.DataFrame):  #
    """class Results
    resultize : return an pd.DataFrame of fancy results"""

    def __init__(
        self,
        name: str,
        dest_results: str = "./results/",
        dest_models: str = "./models/",
        fn: str = "",
        exp: str = "",
        run: str = "",
        columns: list = None,
        index: list = None,
        values: np.ndarray = None,
    ):
        if not (columns and index and values):
            super().__init__()

        # agrs
        self._name = name
        self._fn = fn if fn else secrets.token_hex(4)
        self._dest_results = (
            dest_results if dest_results.endswith("/") else dest_results + "/"
        )
        self._dest_models = (
            dest_models if dest_models.endswith("/") else dest_models + "/"
        )
        self._run = run if run else secrets.token_hex(4)
        self._exp = exp if exp else secrets.token_hex(4)

        # mkdir
        for fold in [dest_models, dest_results]:
            if not os.path.isdir(fold):
                os.mkdir(fold)

        # true
        self.__first = True
        self.__date = str(datetime.datetime.now())[:19]

    # def bla(self):
    #     val = {"a": range(10), "b": range(10)}
    #     df = pd.DataFrame(val)

    #     self = df.copy()


class Results:  #
    """class Results
    resultize : return an pd.DataFrame of fancy results"""

    def __init__(
        self,
        name: str,
        dest_results: str = "./results/",
        dest_models: str = "./models/",
        fn: str = "",
        exp: str = "",
        run: str = "",
        columns: list = None,
        index: list = None,
        values: np.ndarray = None,
    ):
        # agrs
        self._name = name
        self._fn = fn if fn else secrets.token_hex(4)
        self._dest_results = (
            dest_results if dest_results.endswith("/") else dest_results + "/"
        )
        self._dest_models = (
            dest_models if dest_models.endswith("/") else dest_models + "/"
        )
        self._run = run if run else secrets.token_hex(4)
        self._exp = exp if exp else secrets.token_hex(4)

        # mkdir
        for fold in [dest_models, dest_results]:
            if not os.path.isdir(fold):
                os.mkdir(fold)

        # true
        self.__first = True
        self.__date = str(datetime.datetime.now())[:19]
        self.res = pd.DataFrame()
        self.RES = pd.DataFrame()

    def update(
        self,
        grid: GridSearchCV,
        top_only: bool = True,
        verbose: int = 0,
        token: str = "",
        cell: str = "",
        **kwargs: dict,
    ):
        """ """

        # validation
        if not isinstance(grid, GridSearchCV):
            raise AttributeError("GridSearchCV")

        try:
            grid.cv_results_
        except Exception as e:
            logging.error(f"Grid not fitted ! ")
            raise e

        # token and cell
        token = token if not token else secrets.token_hex(4)
        cell = cell if not cell else secrets.token_hex(4)

        # base res
        res = pd.DataFrame(grid.cv_results_)
        cols = [i for i in res.columns if "split" not in i]
        res = res.loc[:, cols]

        # update test/val
        cols = [i.replace("test", "val") for i in res.columns]
        res.columns = cols

        # drop score time

        drop_cols = [
            "mean_score_time",
            "std_score_time",
        ]  # "rank_val_score", "std_fit_time",

        res.drop(columns=drop_cols, inplace=True)

        # do REcast
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
        res["token"] = token
        res["cell"] = cell
        k = secrets.token_hex(4)
        res["model_id"] = k
        for k, v in kwargs.items():
            res[k] = v

        # reorder cols
        end_cols = [
            "mean_val_score",
            "std_val_score",
            "mean_train_score",
            "std_train_score",
        ]
        first_cols = [
            i for i in res.columns if (i not in end_cols)
        ]  # and (i not in universal_cols)
        final_cols = first_cols + end_cols
        res = res.loc[:, final_cols]

        # update results global
        _res = res.copy().head(1) if top_only else res.copy()

        if self.__first:
            # self.drop()
            # self.columns = _res.columns
            # self.index = _res.index
            # self.values = _res.values

            # self  =_res.copy()
            # # self.columns = _res.columns
            self.res = res
            self.RES = _res
            self.__first = False

        else:
            # not working
            self.res = res.copy()
            self.RES = pd.concat([self.RES, _res], ignore_index=True)

            # # fucking uggly
            # for _, ser in _res.iterrows() :
            #     self.loc[len(self)+1] = ser.values

        # # ???
        # self.astype(str).to_csv("./results/log.csv", index=False)
        # display(self)

        self.res.sort_values("mean_val_score", ascending=False, inplace=True)
        self.RES.sort_values("mean_val_score", ascending=False, inplace=True)

        if verbose:
            display(res.round(2).head(10))

        return res.round(2).head(10)

    def save(
        self,
        dest_results: str = None,
        dest_models: str = None,
        csv: bool = True,
        models: bool = True,
        head: int = 10,
        key="mean_val_score,",
    ):
        """ """

        dest_results = dest_results if dest_results else self._dest_results
        dest_models = dest_models if dest_models else self._dest_models
        dest_results = (
            dest_results if dest_results.endswith("/") else dest_results + "/"
        )
        dest_models = dest_models if dest_models.endswith("/") else dest_models + "/"

        if csv:
            self.__save_df(dest_results=dest_results, head=head)
        if models:
            self.__save_models(dest_models=dest_models, head=head)

    def __save_df(self, dest_results="./results/", head=10, key="mean_val_score"):
        """do save df results"""

        de, na, fn, da = dest_results, self._name, self._fn, self.__date[:10]
        file_ = f"{de}{na}_{fn}_{da}.csv"

        # logging.warning(file_)
        self._strize(head=head, key=key).to_csv(file_, index=False)

    def __save_models(self, dest_models="./models/", head=10):
        """do save pk models"""

        de, na, fn, da = dest_models, self._name, self._fn, self.__date[:10]
        file_ = f"{de}{na}_{fn}_{da}__model_"
        gb = (
            self.RES.groupby("model_id")
            .agg({"best_estimator_": "first", "mean_val_score": "max"})
            .sort_values("mean_val_score", ascending=False)
            .head(head)
        )

        for k, model in gb.iterrows():
            # print(k)
            # logging.warning(file_)

            # display(model)
            with open(f"{file_}{k}.pk", "wb") as f:
                pickle.dump(model["best_estimator_"], f)

    def _strize(self, head=10, key="mean_val_score"):
        """sort, head and str a result df"""

        self.RES = self.RES.sort_values(key, ascending=False).head(head)

        self.RES["run"] = self._run
        self.RES["exp"] = self._exp
        self.RES["date"] = self.__date

        # self.__date = str(datetime.datetime.now())[:19]
        # self._run = run if run else secrets.token_hex(4)
        # self._exp = exp if exp else secrets.token_hex(4)

        return self.RES.astype(str)

    @classmethod
    def load_csv(fn):
        """ """

        return pd.read_csv(fn)

    @classmethod
    def load_model(fn):
        """ """

        with open(fn, "rb") as f:
            return pickle.load(f)


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


# class Results:
#     resultize = resultize
#     data = None
