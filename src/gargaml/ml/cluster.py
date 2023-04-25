import os, logging, warnings, sys, random
from IPython.display import display

from sklearn.cluster import *
from sklearn.metrics import *
from sklearn.preprocessing import *

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


from ..eda.pca import Pca


class Cluster:
    """ """

    def __init__(
        self,
        X,
        k=None,
        k_max=5,
        estimator=KMeans,
        customer_id_col="customer_unique_id",
    ) -> None:
        """ """

        self.id_column = X.loc[:, customer_id_col].values

        self.pca = Pca(X.copy(), customer_id_col=customer_id_col)

        self.X = X.select_dtypes(np.number).copy()

        self.k = k
        self.k_max = k_max
        self._estimator = estimator

        # self.estimator = estimator(n_clusters=k_max)
        self.k_best = self.__evaluate_k_best()
        self.X_cluster = self.__build_clusters()

        self.features = [i for i in X.columns if i != customer_id_col]
        self.k_cols = [i for i in self.X_cluster if i.startswith("k_")]

    def __evaluate_k_best(self):
        eval_df = []

        # main loop
        for k in range(2, self.k_max):
            # k
            # print(k)

            # self.estimator fit and predict
            self.estimator = self._estimator(n_clusters=k)

            # display(self.X)
            y = self.estimator.fit_predict(self.X)

            # scores
            var = {
                "k": k,
                "intertia": self.estimator.inertia_,
                "db_score": davies_bouldin_score(self.X, y),
                "silhouette": silhouette_score(self.X, y),
                "labels": y,
            }

            # update avl_df
            eval_df.append(var)

        # df
        eval_df = pd.DataFrame(eval_df)

        # all good score are higher
        eval_df["db_score"] = -eval_df["db_score"]
        eval_df["intertia"] = -eval_df["intertia"]

        return eval_df

    def __build_clusters(self):
        """ """

        _X = self.X.copy()

        _min = 2
        _max = self.k_max

        if self.k:
            _min = self.k
            _max = self.k + 1

        # build clusters for clusters
        for k in range(_min, _max):
            print(k)
            _X[f"k_{k}"] = self.k_best.loc[self.k_best["k"] == k, "labels"].values[0]

        # force k as str
        cols = [i for i in _X.columns if i.startswith("k_")]
        for col in cols:
            _X[col] = _X[col].astype(str)

        return _X

    def inertia(self):
        """ """

        # just show inertia
        fig = px.line(self.k_best, x="k", y="intertia")
        fig.show()

    @property
    def score(self):
        """ """

        n_cols = self.k_best.select_dtypes(include=float).columns
        _df = pd.DataFrame(
            MinMaxScaler().fit_transform(self.k_best.loc[:, n_cols]), columns=n_cols
        )
        _df["intertia"] /= 2
        _df["_score"] = _df.loc[:, ["intertia", "db_score", "silhouette"]].sum(axis=1)
        _df["k"] = self.k_best.k
        _df = _df.loc[
            :,
            [
                "k",
                "intertia",
                "db_score",
                "silhouette",
                "_score",
            ],
        ]
        _df = _df.sort_values("_score", ascending=False).round(1)
        return _df

    def X_cluster_scaled(self, scaler=MinMaxScaler):
        """ """

        if scaler == True:
            scaler = MinMaxScaler()

        try:
            scaler = scaler()
        except:
            pass

        _df = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)
        for k in self.k_cols:
            _df[k] = self.X_cluster[k].values

        return _df

    def plot_k(self, size=5, scaler=False):
        """ """

        # px.scatter(eval_df, x="intertia", y="db_score", color="k")
        # px.scatter(eval_df, x="intertia", y="db_score", color="k")

        fig = px.scatter_3d(
            self.k_best, x="intertia", y="db_score", color="k", z="silhouette"
        )
        fig.update_traces(marker={"size": size})
        fig.show()

    def plot_clusters(self, x_i=0, y_i=1, z_i=2, scaler=False, size=2):
        """ """

        _df = (
            self.X_cluster.copy()
            if not scaler
            else self.X_cluster_scaled(scaler).copy()
        )

        for k in self.k_cols:
            fig = px.scatter_3d(
                _df,
                x=self.features[x_i],
                y=self.features[y_i],
                z=self.features[z_i],
                color=k,
            )
            fig.update_traces(marker={"size": size})
            fig.show()

    def box_clusters(self, scaler=False):
        """ """

        for k in self.k_cols:
            _df = (
                self.X_cluster.copy()
                if not scaler
                else self.X_cluster_scaled(scaler).copy()
            )

            cols = list(self.features) + [k]
            _df = _df.loc[:, cols]

            fig = px.box(_df, color=k)
            # fig.update_traces(marker={'title' : k})

            fig.show()

    def radar_clusters(self, scaler=False, indicator="mean"):
        """ """

        for k in self.k_cols:
            _df = (
                self.X_cluster.copy()
                if not scaler
                else self.X_cluster_scaled(scaler).copy()
            )

            _df = (
                _df.groupby(k).mean()
                if indicator == "mean"
                else _df.groupby(k).median()
            )
            _df = _df.reset_index()
            melt = _df.melt(id_vars=[k])

            # # fig
            fig = px.line_polar(
                melt,
                r="value",
                theta="variable",
                color=k,
                line_close=True,
            )  #  title=f"cluster {cluster_val}"
            fig.show()

    def polar_clusters(self, scaler=False, size=5):
        """ """

        for k in self.k_cols:
            _df = (
                self.X_cluster.copy()
                if not scaler
                else self.X_cluster_scaled(scaler).copy()
            )

            cols = list(self.features) + [
                k,
            ]
            _df = _df.loc[:, cols]
            melt = _df.melt(id_vars=[k])

            fig = px.scatter_polar(melt, r="value", theta="variable", color=k)
            fig.update_traces(marker={"size": size})

            fig.show()

    def value_counts(self, normalize=False):
        """ """

        for k in self.k_cols:
            _df = self.X_cluster[k].value_counts(normalize=normalize)
            _df = _df if not normalize else _df.round(2)
            display(_df)
