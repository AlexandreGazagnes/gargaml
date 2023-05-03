import pytest

from gargaml.data import Load
from gargaml.data import DataClass


def test_init_dataClass():
    """ """

    df = Load.ames(sep_target=False)
    print(df.columns)

    data = DataClass(df, y_name="target")

    print(len(df))
    print(data.X.shape)
    print(data.X.train.shape)
    print(data.train.X.shape)

    print(data)


def test_train_sep_target():
    """ """

    df = Load.ames(sep_target=False)
    data = DataClass(df, y_name="target")

    assert len(data.train.X) > 0
    assert len(data.train.y) > 0
    assert data.train.X.shape[1] > 0


def test_test_sep_target():
    """ """

    df = Load.ames(sep_target=False)
    data = DataClass(df, y_name="target")

    assert len(data.test.X) > 0
    assert len(data.test.y) > 0
    assert data.test.X.shape[1] > 0


def test_sep_target():
    """ """

    df = Load.ames(sep_target=False)
    data = DataClass(df, y_name="target")

    assert len(data.X) > 0
    assert len(data.y) > 0

    assert len(data.X.shape) > 0


def test_sep_target_train():
    """ """

    df = Load.ames(sep_target=False)
    data = DataClass(df, y_name="target")

    assert len(data.X.train) > 0
    assert len(data.y.train) > 0

    assert len(data.X.train.shape) > 0


def test_sep_target_test():
    """ """

    df = Load.ames(sep_target=False)
    data = DataClass(df, y_name="target")

    assert len(data.X.test) > 0
    assert len(data.y.test) > 0

    assert len(data.X.test.shape) > 0
