import pytest
import shap

from sklearn_pandas import DataFrameMapper

from gargaml import *


class TestTitanic:
    """ """

    def test_load(self):
        """ """

        data = Data.Load.titanic(False)
        X, y = Data.Load.titanic(True)
        X, y = Data.Load.titanic(True, precleaning=True)

    def test_simple_model(self):
        """ """

        # load
        X, y = Data.Load.titanic(sep_target=True, precleaning=True)

        X = X.select_dtypes(include=np.number)
        # X = pd.get_dummies(X)
        pipe = Pipeline(
            [
                ("imputer", "passthrough"),
                ("scaler", "passthrough"),
                ("estimator", LogisticRegression()),
            ]
        )

        param_grid = {
            "imputer": [KNNImputer(), SimpleImputer()],
            "scaler": [QuantileTransformer(n_quantiles=30), StandardScaler()],
            "estimator": [
                DummyClassifier(),
                LogisticRegression(),
                KNeighborsClassifier(),
            ],
        }

        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=3,
            n_jobs=1,
            return_train_score=True,
            verbose=2,
        )
        grid.fit(X, y)

    def test_medium_model(self):
        """ """

        # load
        X, y = Data.Load.titanic(sep_target=True, precleaning=True)

        # X = X.select_dtypes(include=np.number)
        X = pd.get_dummies(X)

        pipe = Pipeline(
            [
                ("imputer", "passthrough"),
                ("scaler", "passthrough"),
                ("estimator", LogisticRegression()),
            ]
        )

        param_grid = {
            "imputer": [KNNImputer()],
            "scaler": [QuantileTransformer(n_quantiles=30), StandardScaler()],
            "estimator": [LogisticRegression(), KNeighborsClassifier()],
        }

        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=3,
            n_jobs=1,
            return_train_score=True,
            verbose=2,
        )
        grid.fit(X, y)

    def test_shap(self):
        """ """

        # load
        X, y = Data.Load.titanic(sep_target=True, precleaning=True)

        # X = X.select_dtypes(include=np.number)
        X = pd.get_dummies(X)

        pipe = Pipeline(
            [
                ("imputer", "passthrough"),
                ("scaler", "passthrough"),
                ("estimator", RandomForestClassifier()),
            ]
        )

        param_grid = {
            "imputer": [KNNImputer()],
            "scaler": [
                StandardScaler(),
            ],
            "estimator": [RandomForestClassifier(), XGBClassifier()],
        }

        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=3,
            n_jobs=1,
            return_train_score=True,
            verbose=2,
        )
        grid.fit(X, y)

        estimator = grid.best_estimator_["estimator"]
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X.iloc[:10])
        shap_values


def TestSkLearnPandas():
    """ """

    def test_sklearn_pandas(self):
        """ """

        data = pd.DataFrame(
            {
                "pet": ["cat", "dog", "dog", "fish", "cat", "dog", "cat", "fish"],
                "children": [4.0, 6, 3, 3, 2, 3, 5, 4],
                "salary": [90.0, 24, 44, 27, 32, 59, 36, 27],
            }
        )

    mapper = DataFrameMapper(
        [("pet", LabelBinarizer()), (["children"], StandardScaler())]
    )

    np.round(mapper.fit_transform(data.copy()), 2)
