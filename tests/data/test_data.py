import pytest

from gargaml import Loads
from gargaml.data import DataClass


def test_init_dataClass():


    df = Loads.ames(X_y=False)

    print(df.columns)

    data = DataClass(df, y_name="target")

    print(len(df))
    print(data.X.shape)
    print(data.X.train.shape)
    print(data.train.X.shape)

    print(data)
