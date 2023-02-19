from sklearn.pipeline import Pipeline

from sklearn.dummy import *
from sklearn.impute import *
from sklearn.preprocessing import *


class Pipe:
    """class Pipe
    - pipeline : return pipeline instancianted with default object
    - param_grid : return a dict of paramg rid"""

    @classmethod
    def pipeline(
        cls,
        is_regression: bool,
        txt_pipe: str = "ise",
        default_est=None,
    ):
        """ """

        pst = "passthrough"

        if default_est:
            try:
                default_est = default_est()
            except Exception as e:
                pass

        if not default_est:
            default_est = DummyRegressor() if is_regression else DummyClassifier()

        p_dict = {
            "b": ("balancer", pst),
            "p": ("preprocessor", pst),
            "i": ("imputer", pst),
            "s": ("scaler", pst),
            "r": ("reductor", pst),
            "e": ("estimator", default_est),
        }

        pipe = [p_dict[t] for t in txt_pipe]

        return Pipeline(pipe)

    @classmethod
    def param_grid(
        cls,
        pipe: Pipeline,
        imputer_level: int = 2,
        scaler_level: int = 2,
    ):
        """ """

        pst = "passthrough"

        if scaler_level == 1:
            scaler = [
                pst,
                StandardScaler(),
            ]
        elif scaler_level == 2:
            scaler = [
                pst,
                StandardScaler(),
                QuantileTransformer(n_quantiles=100),
                Normalizer(),
            ]
        elif scaler_level == 3:
            scaler = [
                pst,
                StandardScaler(),
                RobustScaler(),
                Normalizer(),
                QuantileTransformer(n_quantiles=100),
                MinMaxScaler(),
                MaxAbsScaler(),
            ]

        if imputer_level == 1:
            imputer = [
                pst,
            ]
        elif imputer_level == 2:
            imputer = [
                pst,
                SimpleImputer(strategy="median"),
                KNNImputer(n_neighbors=5),
            ]
        elif imputer_level == 3:
            imputer = [
                pst,
                KNNImputer(n_neighbors=3),
                KNNImputer(n_neighbors=5),
                KNNImputer(n_neighbors=10),
                KNNImputer(n_neighbors=15),
                KNNImputer(n_neighbors=20),
                SimpleImputer(strategy="median"),
            ]

        param = {
            i: [
                j,
            ]
            for i, j in pipe.steps
        }

        param["imputer"] = imputer
        param["scaler"] = scaler

        return param
