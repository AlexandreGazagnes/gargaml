import os, sys, logging, warnings
from pathlib import Path

from .path import README_list, data_list, INIT_list, notebooks_list


def create_folders() : 
    """ """
    
    path = os.getcwd()
    # logging.warning(f"path is {path}")

    # all folders
    for dir in README_list + INIT_list:
        try :   
            os.mkdir(f"{path}/{dir}/")
        except Exception as e : 
            logging.error(e)
    # data/
    for dir in data_list : 
        try :   
            os.mkdir(f"{path}/data/{dir}/")
        except Exception as e : 
            logging.error(e)
    
    # github
    try : 
        os.mkdir(f"{path}/.github/")
        os.mkdir(f"{path}/.github/workflows/")
    except Exception as e : 
            logging.error(e)

