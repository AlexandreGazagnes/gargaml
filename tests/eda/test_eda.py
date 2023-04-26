from gargaml import *


# from gargaml import Loads
# from gargaml.data import DataClass


def test_first_tour():


    from gargaml import Loads, EDA

    df = Loads.ames(X_y=False)

    EDA.first_tour.display(df)
    EDA.first_tour.info(df)
    EDA.first_tour.describe(df)
