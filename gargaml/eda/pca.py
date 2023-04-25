import os, sys, logging, warnings, secrets
from IPython.display import display

import numpy as np
import pandas as pd

from sklearn.preprocessing import * 
from sklearn.decomposition import * 

import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px



class Pca:
    """attrs :

    X : original X
    X_scaled : standardscale of X
    pca : pca instance
    pcs : components
    variance : vairaince and cumvariance of pcs

    """

    def __init__(
        self,
        _df,
        customer_id_col="customer_unique_id",
    ) -> None:
        """ """

        # X
        self.X = _df.copy()

        # display(_df)

        self.id_column = _df.loc[:, customer_id_col].values
        X_num = _df.select_dtypes(include=np.number).copy()

        # X_scaled
        X_scaled = pd.DataFrame(
            StandardScaler().fit_transform(X_num), columns=X_num.columns
        )
        self.X_scaled = X_scaled

        # X_pca
        pca = PCA()
        self.pca = pca
        self.pca.fit(X_scaled)
        X_pca = self.pca.transform(X_scaled)
        X_pca = pd.DataFrame(X_pca)
        X_pca.columns = [f"PC_{i+1}" for i, _ in enumerate(X_pca.columns)]
        self.X_pca = X_pca

        # pcs
        pcs = self.pca.components_
        pcs = pd.DataFrame(pcs, index=X_pca.columns, columns=X_scaled.columns)
        pcs = pcs.round(2)
        self.pcs = pcs

        # variance :
        variance = pca.explained_variance_ratio_
        variance_cum = np.cumsum(pca.explained_variance_ratio_)

        self.variance = pd.DataFrame(
            {"variance": variance, "variance_cum": variance_cum},
            index=[f"PC_{i+1}" for i, _ in enumerate(X_pca.columns)],
        )
        self.variance = self.variance.round(2)

    def heatmap(self):
        fig = sns.heatmap(
            self.pcs.T, vmin=-1, vmax=1, cmap="coolwarm", fmt=".2f", annot=True
        )
        # fig.show()