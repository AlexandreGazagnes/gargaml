import os, sys, logging, warnings
from pathlib import Path
from .path import README_list, data_list, INIT_list, notebooks_list, assets_list

def create_files() : 
    """ """

    path = os.getcwd()
    logging.warning(f"path is {path}")

    # readme files
    for dir in README_list : 
        Path(f"{dir}/README.md").touch()

    # data/
    for dir in data_list : 
        Path(f"data/{dir}/README.md").touch()

    # assets/
    for dir in data_list : 
        Path(f"assets/{dir}/README.md").touch()

    # __init__.py
    for dir in INIT_list : 
        Path(f"{dir}/__init__.py").touch()

    # notebooks
    for f in notebooks_list : 
        Path(f"notebooks/{f}.ipynb").touch()
