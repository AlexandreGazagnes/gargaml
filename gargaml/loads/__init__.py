from .loads import (
    _ames,
    _iris,
    _seattle,
    _hr,
    _house,
    _titanic,
    _food,
    _fashion,
    _wine,
    _minst,
)


class Loads:
    """ """

    boston = _ames
    ames = _ames
    iris = _iris
    seattle = _seattle
    hr = _hr
    titanic = _titanic
    house = _house
    minst = _minst
    food = _food
    wine = _wine
    fashion = _fashion

    list_all = [
        "boston",
        "ames",
        "seattle",
        "hr",
        "titanic",
        "house",
        "mnist",
        "food",
        "wine",
        "iris",
        "fashion",
    ]
    list_regression = []
    list_classification = []
    list_image = []
    list_nlp = []
    list_exploration = []
