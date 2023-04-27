from gargaml import *
from gargaml import Loads, Data, EDA, ML, DL


def _import():
    display("from gargaml import * ")
    display("# from gargaml import Build")
    display("from gargaml import Loads, Data, EDA, ML, DL, Walkthrough ")


def _display():
    display("## x.1 Display")
    display("EDA.first_tour.display(_)")


def _structure():
    display("## x.2 Structure")
    display("EDA.first_tour.info(_)")


def _nan():
    display("## x.3 Nan & Duplicated")
    display("EDA.nan.count(_)")
    display("EDA.nan.rate(_)")
    display("EDA.nan.cols(_)")
    display("EDA.nan.lines(_)")
    display("EDA.nan.viz(_)")


def _data_inspection():
    display("## x.4 Data Inspection")


def _pca():
    display("## x.5 fast PCA")


def _first_tour():
    display("# x First Tour")

    _display()
    _structure()
    _nan()
    _data_inspection()
    _pca()


def _all():
    _import()
    _first_tour()


class Walkthrough:
    import_ = _import
    first_tour = _first_tour
    all = _all
