import pytest

from gargaml import *


class TestTitanic:
    """ """

    def test_load(self):
        """ """

        data = Data.Load.titanic(False)
        X, y = Data.Load.titanic(True)

    def test_simple_model(self):
        """ """

        
        X, y = Data.Load.titanic(True)

        X = pd.get_dummies(X)
        pipe = Pipeline(
            [
                ("imputer", "passthrough"),
                ("scaler", "passthrough"),
                ("estimator", LogisticRegression()),
            ]
        )

        param_grid = {
            "imputer": [KNeighborsClassifier(), SimpleImputer()],
            "estimator" : [DummyClassifier(), LogisticRegression(), KNeighborsClassifier()]
            }
        

        grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=1, return_train_score=True, verbose=2,)
        grid.fit(X, y)
