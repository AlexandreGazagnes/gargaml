import os, sys, logging, warnings
from pathlib import Path
from .path import README_list, data_list, INIT_list, notebooks_list, assets_list


def create_files():
    """ """

    path = os.getcwd()
    # logging.warning(f"path is {path}")

    fn_list = []

    # readme files
    for dir in README_list:
        fn_list.append(f"{path}/{dir}/README.md")

    # __init__.py
    for dir in INIT_list:
        fn_list.append(f"{path}/{dir}/__init__.py")

    # data/
    for dir in data_list:
        fn_list.append(f"{path}/data/{dir}/README.md")

    # assets/
    for dir in assets_list:
        fn_list.append(f"{path}/assets/{dir}/README.md")

    # notebooks
    for f in notebooks_list:
        fn_list.append(f"{path}/notebooks/{f}.ipynb")

    # do touch my tralala
    for fn in fn_list:
        try:
            Path(fn).touch()
        except Exception as e:
            logging.error(f"{fn} => {e}")
