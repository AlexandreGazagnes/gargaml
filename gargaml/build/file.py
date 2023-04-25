import os, sys, logging, warnings
from pathlib import Path
from .path import README_list, data_list, INIT_list, notebooks_list, assets_list

def create_files() : 
    """ """

    path = os.getcwd()
    # logging.warning(f"path is {path}")

    # readme files
    for dir in README_list : 
        try : 
            Path(f"{path}/{dir}/README.md").touch()
        except Exception as e : 
            logging.error(e)

    # data/
    for dir in data_list : 
        try
            Path(f"{path}/data/{dir}/README.md").touch()
        except Exception as e : 
            logging.error(e)

    # assets/
    for dir in data_list : 
        try
            Path(f"{path}/assets/{dir}/README.md").touch()
        except Exception as e : 
            logging.error(e)

    # __init__.py
    for dir in INIT_list : 
        try
            Path(f"{path}/{dir}/__init__.py").touch()
        except Exception as e : 
            logging.error(e)

    # notebooks
    for f in notebooks_list : 
        try
            Path(f"{path}/notebooks/{f}.ipynb").touch()
        except Exception as e : 
            logging.error(e)
