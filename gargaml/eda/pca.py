import os, sys, logging, warnings, secrets
from IPython.display import display

import numpy as np
import pandas as pd

from sklearn.preprocessing import *
from sklearn.decomposition import *

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


from sklearn.impute import KNNImputer


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
        n_components: int = None,
        id_col=None,
        kernel=False,
        force_nan_impute=True,
    ) -> None:
        """ """

        # X
        self.X = _df.copy()

        # display(_df)

        self.id_column = self.X.loc[:, id_col].values if id_col else self.X.index.values
        X_num = self.X.select_dtypes(include=np.number).copy()

        if force_nan_impute:
            X_num = pd.DataFrame(
                KNNImputer().fit_transform(X_num), columns=X_num.columns
            )

        self.kernel = kernel

        self.n_components = n_components if n_components else X_num.shape[1]

        # X_scaled
        X_scaled = pd.DataFrame(
            StandardScaler().fit_transform(X_num), columns=X_num.columns
        )
        self.X_scaled = X_scaled

        if not kernel:
            pca = PCA(n_components=self.n_components)
        else:
            pca = KernelPCA(
                n_components=self.n_components, kernel=self.kernel, gamma=10
            )

        self.pca = pca
        self.pca.fit(X_scaled)

    @property
    def _variance(self):
        # variance :
        variance = self.pca.explained_variance_ratio_
        variance_cum = np.cumsum(self.pca.explained_variance_ratio_)

        _variance = pd.DataFrame(
            {"variance": variance, "variance_cum": variance_cum},
            index=[f"PC_{i+1}" for i, _ in enumerate(self.X_proj.columns)],
        )

        return _variance.round(2)

    @property
    def _pcs(self):
        # pcs
        _pcs = self.pca.components_
        _pcs = pd.DataFrame(
            _pcs, index=self.X_proj.columns, columns=self.X_scaled.columns
        )
        _pcs = _pcs.round(2)

        return _pcs

    @property
    def X_proj(self):
        X_proj = self.pca.transform(self.X_scaled)
        X_proj = pd.DataFrame(X_proj)
        X_proj.columns = [f"PC_{i+1}" for i, _ in enumerate(X_proj.columns)]

        return X_proj

    @property
    def variance(self, display_=True):
        """ """
        scree = (self.pca.explained_variance_ratio_ * 100).round(2)
        scree_cum = scree.cumsum().round()
        x_list = range(1, self.n_components + 1)

        if display_:
            plt.bar(x_list, scree)
            plt.plot(x_list, scree_cum, c="red", marker="o")
            plt.xlabel("rang de l'axe d'inertie")
            plt.ylabel("pourcentage d'inertie")
            plt.title("Eboulis des valeurs propres")
            plt.show(block=False)

        return self._variance

    @property
    def pcs(self):
        """ """
        fig = sns.heatmap(
            self._pcs.T, vmin=-1, vmax=1, cmap="coolwarm", fmt=".2f", annot=True
        )
        # fig.show()

        return self._pcs.T

    def correlation_graph(
        self,
        x_y,
    ):
        """Affiche le graphe des correlations

        Positional arguments :
        -----------------------------------
        x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
        """

        # Extrait x et y
        x, y = x_y

        # features
        features = self.X_scaled.columns

        # Taille de l'image (en inches)
        fig, ax = plt.subplots(figsize=(10, 9))

        # Pour chaque composante :
        for i in range(0, self.pca.components_.shape[1]):
            # Les flèches
            ax.arrow(
                0,
                0,
                self.pca.components_[x, i],
                self.pca.components_[y, i],
                head_width=0.07,
                head_length=0.07,
                width=0.02,
            )

            # Les labels
            plt.text(
                self.pca.components_[x, i] + 0.05,
                self.pca.components_[y, i] + 0.05,
                features[i],
            )

        # Affichage des lignes horizontales et verticales
        plt.plot([-1, 1], [0, 0], color="grey", ls="--")
        plt.plot([0, 0], [-1, 1], color="grey", ls="--")

        # Nom des axes, avec le pourcentage d'inertie expliqué
        plt.xlabel(
            "F{} ({}%)".format(
                x + 1, round(100 * self.pca.explained_variance_ratio_[x], 1)
            )
        )
        plt.ylabel(
            "F{} ({}%)".format(
                y + 1, round(100 * self.pca.explained_variance_ratio_[y], 1)
            )
        )

        # J'ai copié collé le code sans le lire
        plt.title("Cercle des corrélations (F{} et F{})".format(x + 1, y + 1))

        # Le cercle
        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

        # Axes et display
        plt.axis("equal")
        plt.show(block=False)

    def factorial_planes(
        self,
        x_y,
        labels: str = None,
        clusters: str = None,
        alpha: float = 1,
        figsize: list = [10, 8],
        marker: str = ".",
    ):
        """
        Affiche la projection des individus

        Positional arguments :
        -------------------------------------
        x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

        Optional arguments :
        -------------------------------------
        labels : str, list/tuple : les labels des individus à projeter, default = None
        si str on va chercher la colonne du df, si list on ajoute ex nihilo
        clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
        si str on va chercher la colonne du df, si list on ajoute ex nihilo
        alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
        figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8]
        marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
        """

        # Transforme self.X_proj en np.array
        X_ = np.array(self.X_proj)

        # On définit la forme de la figure si elle n'a pas été donnée
        if not figsize:
            figsize = (7, 6)

        # On gère les labels

        if (
            isinstance(labels, [list, tuple, pd.Series]) # np.ndarray

        ):
            labels = labels
        elif labels in [None, "", 0, False, []]:
            labels = []
        elif isinstance(labels, str):
            labels = self.X.loc[:, labels].values

        try:
            len(labels)
        except Exception as e:
            raise e

        # idem clusters
        # On vérifie s'il y a des clusters ou non
        if clusters in [None, "", 0, False, []]:
            clusters = []
        elif isinstance(clusters, str):
            clusters = self.X.loc[:, clusters].values
        elif (
            isinstance(clusters, list)
            or isinstance(clusters, tuple)
            or isinstance(clusters, pd.Series)
            or isinstance(clusters, np.array)
        ):
            clusters = clusters
        try:
            len(clusters)
        except Exception as e:
            raise e

        # On vérifie la variable axis
        if not len(x_y) == 2:
            raise AttributeError("2 axes sont demandées")
        if max(x_y) >= X_.shape[1]:
            raise AttributeError("la variable axis n'est pas bonne")

        # on définit x et y
        x, y = x_y

        # Initialisation de la figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Les points
        # plt.scatter(   X_[:, x], X_[:, y], alpha=alpha,
        #                     c=c, cmap="Set1", marker=marker)
        if len(clusters):
            sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=clusters)
        else:
            sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y])

        # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe
        v1 = str(round(100 * self.pca.explained_variance_ratio_[x])) + " %"
        v2 = str(round(100 * self.pca.explained_variance_ratio_[y])) + " %"

        # Nom des axes, avec le pourcentage d'inertie expliqué
        ax.set_xlabel(f"F{x+1} {v1}")
        ax.set_ylabel(f"F{y+1} {v2}")

        # Valeur x max et y max
        x_max = np.abs(X_[:, x]).max() * 1.1
        y_max = np.abs(X_[:, y]).max() * 1.1

        # On borne x et y
        ax.set_xlim(left=-x_max, right=x_max)
        ax.set_ylim(bottom=-y_max, top=y_max)

        # Affichage des lignes horizontales et verticales
        plt.plot([-x_max, x_max], [0, 0], color="grey", alpha=0.8)
        plt.plot([0, 0], [-y_max, y_max], color="grey", alpha=0.8)

        # Affichage des labels des points
        if len(labels):
            # j'ai copié collé la fonction sans la lire
            for i, (_x, _y) in enumerate(X_[:, [x, y]]):
                plt.text(
                    _x, _y + 0.05, labels[i], fontsize="14", ha="center", va="center"
                )

        # Titre et display
        plt.title(f"Projection des individus (sur F{x+1} et F{y+1})")
        plt.show()
